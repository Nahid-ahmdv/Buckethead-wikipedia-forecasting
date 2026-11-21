"""Behavioral checks for packaging failures and information-boundary safeguards."""
import ast
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import integrity
import verify_saved
from research_adapter import FAILURE_POLICY, UnexpectedForecastError, research_layout


class DeliveryTests(unittest.TestCase):
    def test_guard_rejects_missing_and_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'provenance').mkdir()
            (root / 'required.csv').write_text('a\n1\n')
            (root / 'provenance/manifest.json').write_text(json.dumps({'files': {'required.csv': integrity.sha(root / 'required.csv')}}))
            with patch.object(integrity, 'ROOT', root), patch.object(integrity, 'HERE', root / 'provenance'):
                (root / 'required.csv').write_text('a\n2\n')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    integrity.verify_current()
                (root / 'required.csv').unlink()
                with self.assertRaisesRegex(FileNotFoundError, 'missing'):
                    integrity.verify_current()

    def test_single_notebook_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'provenance').mkdir()
            (root / 'provenance/manifest.json').write_text('{"files": {}}')
            (root / 'buckethead_wikipedia_forecasting.ipynb').touch()
            (root / 'extra.ipynb').touch()
            with patch.object(integrity, 'ROOT', root), patch.object(integrity, 'HERE', root / 'provenance'):
                with self.assertRaisesRegex(ValueError, 'one notebook'):
                    integrity.verify_current()

    def test_explicit_author_backup_is_ignored_but_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'provenance').mkdir()
            (root / 'provenance/manifest.json').write_text('{"files": {}}')
            (root / 'buckethead_wikipedia_forecasting.ipynb').touch()
            backup = root / 'buckethead_wikipedia_forecasting copy.ipynb'
            backup.write_bytes(b'author backup')
            with patch.object(integrity, 'ROOT', root), patch.object(integrity, 'HERE', root / 'provenance'), \
                 patch.object(integrity, 'source_map', return_value={}):
                result = integrity.verify_current()
            self.assertEqual(result['notebooks'], ['buckethead_wikipedia_forecasting.ipynb'])
            self.assertEqual(backup.read_bytes(), b'author backup')

    def test_weekday_baseline_ignores_unavailable_and_future_values(self):
        dates = pd.date_range('2023-01-01', '2024-02-01')
        y = pd.Series(np.arange(len(dates), dtype=float), index=dates)
        issue = pd.Timestamp('2024-01-01')
        baseline = verify_saved.weekday_forecast(y, issue)
        perturbed = y.copy()
        perturbed.loc[issue - pd.Timedelta(days=1):] = 1e12
        np.testing.assert_array_equal(baseline, verify_saved.weekday_forecast(perturbed, issue))
        perturbed.loc[issue - pd.Timedelta(days=2)] += 56
        self.assertFalse(np.array_equal(baseline, verify_saved.weekday_forecast(perturbed, issue)))

    def test_mathematical_helpers_preserved(self):
        original = json.loads(gzip.decompress((integrity.ROOT / 'provenance/original_notebook.json.gz').read_bytes()))
        # The illustrative baseline remains the same mathematical implementation.
        current = json.loads((integrity.ROOT / 'buckethead_wikipedia_forecasting.ipynb').read_text())
        old_functions = [node for cell in original['cells'] if cell['cell_type'] == 'code'
                         for node in ast.parse(''.join(cell['source'])).body if isinstance(node, ast.FunctionDef)]
        new_functions = [node for cell in current['cells'] if cell['cell_type'] == 'code'
                         for node in ast.parse(''.join(cell['source'])).body if isinstance(node, ast.FunctionDef)]
        self.assertEqual([ast.dump(node) for node in old_functions], [ast.dump(node) for node in new_functions])

    def test_original_source_contracts_and_mature_training_labels(self):
        with research_layout() as (root, forecasting, final, engine):
            # This exercises the original source/configuration lock checks.
            final.approved_contract()
            lock = json.loads((engine.HERE / 'implementation_lock.json').read_text())
            for relative, expected in lock['files'].items():
                self.assertEqual(engine.digest(root / relative), expected)
            y = engine.load_y()
            origin = engine.at_origin(y, '2025-11-02')
            x, labels, metadata = forecasting.supervised_paths(origin, 'expanding')
            self.assertLessEqual(metadata.target_date.max(), origin.cutoff)
            self.assertTrue((metadata.feature_cutoff < metadata.training_issue).all())
            self.assertEqual(len(x), len(labels))

    def test_unexpected_errors_stop_each_supported_runner_without_retry(self):
        with research_layout() as (_, forecasting, final, engine):
            y = engine.load_y()
            original_spec = final.roster()[0]
            retrospective_spec = next(s for s in engine.roster() if s['family'] == 'sarimax')
            cases = [
                (forecasting, forecasting, '2024-11-03', original_spec),
                (final, forecasting, '2025-11-02', original_spec),
                (engine, engine, '2025-11-02', retrospective_spec),
            ]
            for runner, model_module, issue, spec in cases:
                for error_type in (ValueError, RuntimeError):
                    with self.subTest(runner=runner.__name__, error=error_type.__name__):
                        with patch.object(model_module, 'fit_sarimax',
                                          side_effect=error_type('invalid configuration')) as fit:
                            with self.assertRaises(UnexpectedForecastError) as caught:
                                runner.predict(y, issue, spec)
                            self.assertEqual(caught.exception.details['error_type'], error_type.__name__)
                            self.assertIn('invalid configuration', str(caught.exception))
                            self.assertEqual(fit.call_count, 1)
                with patch.object(model_module, 'fit_sarimax', side_effect=TypeError('programming error')):
                    with self.assertRaisesRegex(TypeError, 'programming error'):
                        runner.predict(y, issue, spec)

    def test_explicit_numerical_failures_keep_the_logged_naive_fallback(self):
        with research_layout() as (_, forecasting, final, engine):
            y = engine.load_y()
            original_spec = final.roster()[0]
            retrospective_spec = next(s for s in engine.roster() if s['family'] == 'sarimax')
            cases = [
                (forecasting, forecasting, '2024-11-03', original_spec),
                (final, forecasting, '2025-11-02', original_spec),
                (engine, engine, '2025-11-02', retrospective_spec),
            ]
            for runner, model_module, issue, spec in cases:
                for error_type in (forecasting.ModelFailure, np.linalg.LinAlgError):
                    with self.subTest(runner=runner.__name__, error=error_type.__name__):
                        with patch.object(model_module, 'fit_sarimax',
                                          side_effect=error_type('numerical failure')) as fit:
                            prediction, details = runner.predict(y, issue, spec)
                        expected = forecasting.weekly_naive(engine.at_origin(y, issue, scheduled=False))
                        np.testing.assert_array_equal(np.asarray(prediction), expected)
                        self.assertTrue(details['fallback'])
                        self.assertEqual(details['error_type'], error_type.__name__)
                        self.assertEqual(fit.call_count, 1)

    def test_successful_prediction_and_adapter_cleanup_are_unchanged(self):
        self.assertEqual(FAILURE_POLICY, 'explicit-numerical-failures-only-v1')
        with research_layout() as (_, forecasting, final, engine):
            y = engine.load_y()
            spec = next(s for s in final.roster() if s['family'] == 'weekday_mean')
            originals = {module: module.predict.__wrapped__ for module in (forecasting, final, engine)}
            actual, details = final.predict(y, '2025-11-02', spec)
            expected, original_details = originals[final](y, '2025-11-02', spec)
            np.testing.assert_array_equal(actual, expected)
            for key, value in original_details.items():
                if key != 'runtime_seconds':
                    self.assertEqual(details[key], value)
        for module, original in originals.items():
            self.assertIs(module.predict, original)


if __name__ == '__main__':
    unittest.main()
