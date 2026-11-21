"""Information-boundary and scoring tests, requiring no model downloads."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import foundation_analysis as fm


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.y = pd.Series(np.arange(2200, dtype=float) + 10,
                           index=pd.date_range('2020-01-01', periods=2200))
        self.issue = pd.Timestamp('2025-01-01')
        self.fold = {'issue': self.issue, 'target_end': self.issue + pd.Timedelta(days=27),
                     'stage': 'selection', 'origin_id': 'example'}

    def test_context_excludes_unpublished_day_and_all_future_targets(self):
        first = fm.context_for(self.y, self.issue)
        changed = self.y.copy()
        changed.loc[self.issue - pd.Timedelta(days=1):] = 1e10
        second = fm.context_for(changed, self.issue)
        pd.testing.assert_series_equal(first, second)
        self.assertEqual(fm.context_digest(first), fm.context_digest(second))
        changed.loc[self.issue - pd.Timedelta(days=2)] += 1
        self.assertNotEqual(fm.context_digest(first),
                            fm.context_digest(fm.context_for(changed, self.issue)))

    def test_bridge_drops_only_the_unobserved_intervening_day(self):
        raw = np.column_stack([np.arange(29), np.arange(29) + 1, np.arange(29) + 2])
        rows, repairs = fm.bridge_rows(self.y, self.fold, raw, 'model')
        scored = rows.loc[rows.scored]
        self.assertEqual(len(scored), 28)
        self.assertEqual(rows.iloc[0].date, '2024-12-31')
        self.assertEqual(scored.iloc[0].date, '2025-01-01')
        self.assertEqual(scored.iloc[-1].date, '2025-01-28')
        np.testing.assert_array_equal(scored.prediction, raw[1:, 1])
        np.testing.assert_array_equal(scored.horizon, np.arange(1, 29))
        self.assertEqual(repairs, {'crossing_rows': 0, 'negative_quantiles': 0})

    def test_invalid_outputs_stop_and_quantile_repairs_are_counted(self):
        raw = np.tile([3, -1, 2], (29, 1))
        q, repairs = fm.normalize_quantiles(raw)
        np.testing.assert_array_equal(q, np.tile([0, 2, 3], (29, 1)))
        self.assertEqual(repairs, {'crossing_rows': 29, 'negative_quantiles': 29})
        for invalid in [np.ones((28, 3)), np.full((29, 3), np.nan)]:
            with self.assertRaises(ValueError):
                fm.normalize_quantiles(invalid)

    def test_masked_scores_pool_days_and_do_not_change_predictions(self):
        dates = pd.date_range('2025-11-02', periods=56)
        reference = pd.DataFrame({
            'model_id': 'weekday_mean', 'stage': 'retrospective',
            'issue': ['2025-11-02'] * 28 + ['2025-11-30'] * 28,
            'date': dates.strftime('%Y-%m-%d'), 'horizon': list(range(1, 29)) * 2,
            'actual': 10., 'prediction': 12., 'mase_scale': 2.,
            'lower': np.nan, 'upper': np.nan,
        })
        new = reference.copy()
        new['model_id'] = 'model'
        new['origin_id'] = ['first'] * 28 + ['second'] * 28
        new['scored'] = True
        new['prediction'] = np.arange(56) + 10.
        new['lower'], new['upper'] = 0., 100.
        before = new.copy(deep=True)
        with patch.object(fm, 'reference_forecasts', return_value=reference):
            result = fm.assemble_tables(new)['comparison']
        row = result.loc[(result.model_id == 'model') &
                         (result.population == 'exclude_november_2025')].iloc[0]
        self.assertEqual(row.days, 27)
        self.assertEqual(row.paths, 1)
        self.assertEqual(row.mae, np.arange(29, 56).mean())
        pd.testing.assert_frame_equal(new, before)


if __name__ == '__main__':
    unittest.main()
