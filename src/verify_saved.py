"""Read-only checks of saved predictions and all numbers used by the notebook.

No statistical/boosting engine imports, optimizer calls or research writes.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from integrity import ROOT, verify_current, historical_status


RESULTS = {'phase1': ROOT / 'results/development',
           'phase2': ROOT / 'results/original_evaluation',
           'phase3': ROOT / 'results/diagnostics',
           'phase4': ROOT / 'results/retrospective_followup'}


def metrics(g):
    error = g.prediction.to_numpy() - g.actual.to_numpy()
    return {'days': len(g), 'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt(np.mean(error ** 2))),
            'mase': float(np.mean(np.abs(error) / g.mase_scale.to_numpy())),
            'mean_signed_error': float(error.mean()),
            'overprediction_days': int((error > 0).sum())}


def interval_metrics(g):
    below = g.actual < g.lower
    above = g.actual > g.upper
    width = g.upper - g.lower
    score = width + 10 * np.maximum(g.lower - g.actual, 0) + 10 * np.maximum(g.actual - g.upper, 0)
    return {'days': len(g), 'coverage': float((~(below | above)).mean()),
            'mean_width': float(width.mean()), 'below': int(below.sum()),
            'above': int(above.sum()), 'interval_score80': float(score.mean())}


def weekday_forecast(y, issue):
    cutoff = pd.Timestamp(issue) - pd.Timedelta(days=2)
    recent = y.loc[:cutoff].iloc[-56:]
    assert len(recent) == 56 and recent.index[-1] == cutoff
    targets = pd.date_range(issue, periods=28)
    means = recent.groupby(recent.index.dayofweek).mean()
    return means.reindex(targets.dayofweek).to_numpy()


def read_results():
    verify_current()
    y = pd.read_csv(ROOT / 'data/processed/buckethead_daily.csv', parse_dates=['date']).set_index('date')['count']
    raw = json.loads((ROOT / 'data/raw/pageviews_all-access.json').read_text())['items']
    raw_y = pd.Series([r['views'] for r in raw], index=pd.to_datetime([r['timestamp'] for r in raw], format='%Y%m%d%H')).sort_index()
    assert y.index.equals(pd.date_range('2015-07-01', '2026-10-31')) and len(y) == 4141
    np.testing.assert_array_equal(y, raw_y)
    dev = pd.read_csv(ROOT / 'results/development/development_forecasts.csv')
    original = pd.read_csv(ROOT / 'results/original_evaluation/final_forecasts.csv')
    extension = pd.read_csv(ROOT / 'results/retrospective_followup/forecasts.csv')
    return y, dev, original, extension


def verify():
    integrity = verify_current()
    history = historical_status()
    y, dev, original, extension = read_results()
    records = 0
    # Match every CSV prediction to its immutable per-origin cache.
    for phase, frame, audit_name, key_columns in (
        ('phase1', dev, 'origin_metrics.csv', ['stage', 'origin_id', 'procedure_id']),
        ('phase2', original, 'fit_audits.json', ['origin_id', 'procedure_id']),
        ('phase4', extension, 'fit_audits.json', ['origin_id', 'id']),
    ):
        audit_path = RESULTS[phase] / audit_name
        audits = pd.read_csv(audit_path).to_dict('records') if phase == 'phase1' else json.loads(audit_path.read_text())
        for audit in audits:
            mask = np.ones(len(frame), dtype=bool)
            for key in key_columns:
                mask &= frame[key].eq(audit[key]).to_numpy()
            g = frame.loc[mask]
            assert len(g) == 28 and list(g.horizon) == list(range(1, 29))
            key = audit['cache_key'] if phase != 'phase2' else audit['origin_id'] + '__' + audit['procedure_id']
            cached = json.loads((RESULTS[phase] / 'cache' / (key + '.json')).read_text())
            dates = cached['target_dates'] if phase == 'phase1' else cached['dates']
            assert g.date.tolist() == dates
            np.testing.assert_allclose(g.prediction, cached['prediction'], rtol=1e-13)
            np.testing.assert_array_equal(g.actual, y.loc[pd.to_datetime(g.date)])
            cutoff = pd.Timestamp(cached['info']['cutoff'])
            assert pd.Timestamp(dates[0]) - cutoff == pd.Timedelta(days=2)
            scale_y = y.loc[(y.index > cutoff - pd.DateOffset(years=3)) & (y.index <= cutoff)].to_numpy()
            np.testing.assert_allclose(g.mase_scale, np.abs(scale_y[7:] - scale_y[:-7]).mean(), rtol=1e-13)
            assert not cached['info']['fallback']
            if 'label_end' in cached['info']:
                assert pd.Timestamp(cached['info']['label_end']) <= cutoff
            records += 1
        error = frame.prediction - frame.actual
        # Decimal CSV round-trips can differ at ~1e-12 after subtraction.
        np.testing.assert_allclose(frame.absolute_error, np.abs(error), rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(frame.squared_error, error ** 2, rtol=1e-12, atol=1e-8)

    reconciled = 0
    for phase, frame, populations, model_key in (
        ('phase1', dev, None, 'procedure_id'),
        ('phase2', original, ('primary', 'exclude_november_2025'), 'family'),
        ('phase4', extension, ('all', 'exclude_november_2025'), 'id'),
    ):
        names = ['tuning_summary', 'selection_summary'] if phase == 'phase1' else ['summary']
        for name in names:
            saved = pd.read_csv(RESULTS[phase] / (name + '.csv'))
            for _, row in saved.iterrows():
                g = frame.loc[frame[model_key] == row[model_key]]
                if phase == 'phase1':
                    g = g.loc[g.stage == ('tuning' if name.startswith('tuning') else 'selection')]
                elif phase == 'phase4':
                    g = g.loc[g.stage == row.stage]
                if phase != 'phase1' and row.population == 'exclude_november_2025':
                    g = g.loc[~g.date.str.startswith('2025-11-')]
                m = metrics(g)
                np.testing.assert_allclose([m[k] for k in ('mae', 'rmse', 'mase')],
                                           [row[k] for k in ('mae', 'rmse', 'mase')], rtol=1e-11)
                reconciled += 1

    # Independently reconstruct the original fixed interval and the later signed-error rule.
    config = json.loads((ROOT / 'results/development/proposed_final_configuration.json').read_text())
    calibration = pd.read_csv(ROOT / 'results/development/calibration_errors.csv')
    q = config['interval']['half_width']
    np.testing.assert_allclose(q, np.sort(calibration.absolute_error)[269], rtol=1e-13)
    sarima = original.loc[original.family == 'sarimax']
    np.testing.assert_allclose(sarima.lower, np.maximum(0, sarima.prediction - q))
    np.testing.assert_allclose(sarima.upper, sarima.prediction + q)
    selected = json.loads((ROOT / 'results/retrospective_followup/point_selection.json').read_text())['selected_id']
    assert selected == 'weekday_mean'
    offsets = json.loads((ROOT / 'results/retrospective_followup/interval_calibration.json').read_text())['procedures']
    assert set(offsets) == {selected}
    p4base = extension.loc[(extension.stage == 'retrospective') & (extension.id == selected)]
    cal = extension.loc[(extension.stage == 'calibration') & (extension.id == selected)]
    residual = np.sort(cal.actual - cal.prediction)
    np.testing.assert_allclose([offsets[selected]['lower_offset'], offsets[selected]['upper_offset']], residual[[10, 101]])
    np.testing.assert_allclose(p4base.lower, np.maximum(0, p4base.prediction + residual[10]))
    np.testing.assert_allclose(p4base.upper, np.maximum(0, p4base.prediction + residual[101]))
    for phase, frame, old in [('phase2', sarima, True), ('phase4', p4base, False)]:
        saved = pd.read_csv(RESULTS[phase] / 'interval_metrics.csv')
        saved = saved.loc[saved.origin_id == 'all'] if old else saved.loc[(saved.group_type == 'origin') & (saved['group'] == 'all')]
        for _, row in saved.iterrows():
            g = frame if row.population in {'primary', 'all'} else frame.loc[~frame.date.str.startswith('2025-11-')]
            keys = ['coverage', 'mean_width', 'below', 'above'] + ([] if old else ['interval_score80'])
            m = interval_metrics(g)
            np.testing.assert_allclose([m[k] for k in keys], [row[k] for k in keys], rtol=1e-12)

    # Reconstruct both delayed baselines for all 36 Phase 4 issues.
    for issue, g in extension.loc[extension.id == 'weekday_mean'].groupby('issue', sort=False):
        np.testing.assert_allclose(g.prediction, weekday_forecast(y, issue), rtol=1e-13)
        naive = extension.loc[(extension.issue == issue) & (extension.id == 'weekly_naive')]
        hist = y.loc[:pd.Timestamp(issue) - pd.Timedelta(days=2)]
        expected = [hist.loc[hist.index.dayofweek == d.dayofweek].iloc[-1] for d in pd.date_range(issue, periods=28)]
        np.testing.assert_allclose(naive.prediction, expected, rtol=1e-13)
    old_base = original.loc[original.family == 'weekday_mean']
    np.testing.assert_allclose(old_base.prediction, p4base.prediction, rtol=1e-13)
    assert old_base.date.tolist() == p4base.date.tolist()

    diagnostic = json.loads((ROOT / 'results/diagnostics/diagnostic_summary.json').read_text())
    m = metrics(sarima)
    assert m['overprediction_days'] == diagnostic['selected']['overprediction_days'] == 326
    np.testing.assert_allclose(m['mean_signed_error'], diagnostic['selected']['bias_forecast_minus_actual'])
    top10 = np.sort(np.abs(sarima.prediction - sarima.actual))[-10:].sum() / np.abs(sarima.prediction - sarima.actual).sum()
    np.testing.assert_allclose(top10, diagnostic['top10_error_share'])
    levels = pd.read_csv(ROOT / 'results/diagnostics/path_level_diagnostics.csv').loc[lambda d: d.family == 'sarimax']
    for _, row in levels.iterrows():
        cutoff = pd.Timestamp(row.issue) - pd.Timedelta(days=2)
        np.testing.assert_allclose(row.past56_mean, y.loc[:cutoff].iloc[-56:].mean())
    closer = int((np.abs(levels.past56_mean - levels.actual_mean) < np.abs(levels.prediction_mean - levels.actual_mean)).sum())
    assert closer == 12
    challenger = extension.loc[(extension.stage == 'retrospective') & (extension.id == 'lgbm_correction')]
    improvement = 100 * (1 - metrics(challenger)['mae'] / metrics(p4base)['mae'])
    masked = lambda g: g.loc[~g.date.str.startswith('2025-11-')]
    sensitivity = 100 * (1 - metrics(masked(challenger))['mae'] / metrics(masked(p4base))['mae'])
    assert round(improvement, 2) == .92 and round(sensitivity, 2) == .06
    figures = json.loads((ROOT / 'provenance/notebook_figure_sources.json').read_text())['figures']
    for fig in figures:
        assert (ROOT / fig['file']).is_file()
        for source, digest in fig['source_sha256'].items():
            from integrity import sha
            assert sha(ROOT / source) == digest
    return {'status': 'passed', 'integrity': integrity, 'historical_manifest_reconciliation': history,
            'cached_origin_records_verified': records, 'daily_predictions_verified': len(dev) + len(original) + len(extension),
            'summary_rows_recomputed': reconciled, 'delayed_baseline_paths_reconstructed': 72,
            'original_overprediction_days': 326, 'original_mean_signed_error': m['mean_signed_error'],
            'top10_absolute_error_share': float(top10), 'recent_level_closer_paths': closer,
            'correction_improvement_percent': improvement, 'masked_improvement_percent': sensitivity,
            'original_intervals': interval_metrics(sarima), 'phase4_intervals': interval_metrics(p4base),
            'new_optimizer_fits': 0, 'fresh_future_paths': 0}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
