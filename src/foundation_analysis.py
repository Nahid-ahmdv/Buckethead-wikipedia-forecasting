"""Timing, scoring and read-only checks for the bounded foundation-model extension.

Model libraries are deliberately imported only by the separate inference script.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/foundation_models'
PROTOCOL = ROOT / 'provenance/foundation_models/protocol.json'
LABELS = {
    'chronos2': 'Chronos-2, zero-shot',
    'timesfm25': 'TimesFM 2.5, zero-shot',
    'weekday_mean': 'Eight-week weekday mean',
    'weekly_naive': 'Weekly naive',
    'original_sarimax': 'Original selected SARIMAX',
    'lgbm_correction': 'Retrospective LightGBM correction',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_target(root=ROOT):
    target = pd.read_csv(Path(root) / 'data/processed/buckethead_daily.csv',
                         parse_dates=['date']).set_index('date')['count'].astype(float)
    assert target.index.equals(pd.date_range('2015-07-01', '2026-10-31'))
    assert target.notna().all() and (target >= 0).all()
    return target


def schedule(root=ROOT):
    frame = pd.read_csv(Path(root) / 'provenance/protocols/original/split_schedule.csv')
    frame = frame.loc[~frame.boundary_observation_block.astype(bool)].copy()
    frame['stage'] = frame['split'].replace({'final': 'retrospective'})
    frame['issue'] = pd.to_datetime(frame.target_start)
    frame['cutoff'] = pd.to_datetime(frame.information_cutoff_date)
    frame['target_end'] = pd.to_datetime(frame.target_end)
    assert frame.groupby('stage').size().to_dict() == {
        'retrospective': 13, 'selection': 12, 'tuning': 12}
    assert (frame.issue - frame.cutoff == pd.Timedelta(days=2)).all()
    assert (frame.target_end - frame.issue == pd.Timedelta(days=27)).all()
    return frame[['stage', 'origin_id', 'issue', 'cutoff', 'target_end']].reset_index(drop=True)


def context_for(target, issue, context_length=1024):
    """The only observed target values an inference adapter may receive."""
    issue = pd.Timestamp(issue)
    cutoff = issue - pd.Timedelta(days=2)
    context = target.loc[:cutoff].iloc[-context_length:]
    assert len(context) == context_length and context.index[-1] == cutoff
    assert context.index.equals(pd.date_range(context.index[0], cutoff))
    assert np.isfinite(context).all()
    return context


def context_digest(context):
    """Stable values/date digest, independent of CSV text formatting."""
    payload = {'dates': context.index.strftime('%Y-%m-%d').tolist(),
               'values': context.to_numpy(dtype=float).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def scale_for(target, cutoff):
    cutoff = pd.Timestamp(cutoff)
    history = target.loc[(target.index > cutoff - pd.DateOffset(years=3)) &
                         (target.index <= cutoff)].to_numpy(dtype=float)
    scale = float(np.abs(history[7:] - history[:-7]).mean())
    assert scale > 0
    return scale


def normalize_quantiles(raw):
    """Predeclared monotone projection, then nonnegative count clipping."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (29, 3) or not np.isfinite(raw).all():
        raise ValueError(f'Expected 29 by 3 finite quantiles, received {raw.shape}')
    repaired = np.sort(raw, axis=1)
    info = {'crossing_rows': int(np.any(np.diff(raw, axis=1) < 0, axis=1).sum()),
            'negative_quantiles': int((repaired < 0).sum())}
    return np.maximum(repaired, 0), info


def bridge_rows(target, fold, quantiles, model_id):
    """Drop O-1 explicitly; attach evaluation outcomes only after prediction."""
    issue = pd.Timestamp(fold['issue'])
    cutoff = issue - pd.Timedelta(days=2)
    q, repairs = normalize_quantiles(quantiles)
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=29)
    assert dates[1] == issue and dates[-1] == pd.Timestamp(fold['target_end'])
    rows = pd.DataFrame({
        'model_id': model_id, 'stage': fold['stage'], 'origin_id': fold['origin_id'],
        'issue': issue.strftime('%Y-%m-%d'), 'cutoff': cutoff.strftime('%Y-%m-%d'),
        'date': dates.strftime('%Y-%m-%d'), 'bridge_step': np.arange(1, 30),
        'horizon': np.arange(0, 29), 'scored': np.arange(29) > 0,
        'lower': q[:, 0], 'prediction': q[:, 1], 'upper': q[:, 2],
        'raw_q10': np.asarray(quantiles)[:, 0],
        'raw_q50': np.asarray(quantiles)[:, 1],
        'raw_q90': np.asarray(quantiles)[:, 2],
        'mase_scale': scale_for(target, cutoff),
    })
    rows['actual'] = target.reindex(dates).to_numpy()
    assert rows.actual.notna().all()
    return rows, repairs


def point_metrics(frame):
    e = frame.prediction.to_numpy(float) - frame.actual.to_numpy(float)
    return {'paths': int(frame.issue.nunique()), 'days': len(frame),
            'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt(np.mean(e ** 2))),
            'mase': float(np.mean(np.abs(e) / frame.mase_scale.to_numpy(float))),
            'signed_error': float(e.mean()), 'overprediction_days': int((e > 0).sum())}


def interval_metrics(frame):
    actual, lower, upper = [frame[c].to_numpy(float) for c in ['actual', 'lower', 'upper']]
    below, above = actual < lower, actual > upper
    width = upper - lower
    return {'coverage': float((~(below | above)).mean()), 'mean_width': float(width.mean()),
            'below': int(below.sum()), 'above': int(above.sum()),
            'interval_score80': float(np.mean(width + 10 * np.maximum(lower - actual, 0)
                                             + 10 * np.maximum(actual - upper, 0)))}


def reference_forecasts(root=ROOT):
    """Load existing predictions without fitting or changing original procedures."""
    root = Path(root)
    original = pd.read_csv(root / 'results/original_evaluation/final_forecasts.csv')
    original['model_id'] = original.family.replace({'sarimax': 'original_sarimax'})
    original['stage'] = 'retrospective'
    followup = pd.read_csv(root / 'results/retrospective_followup/forecasts.csv')
    # Phase 4's baseline point forecasts match the original baseline exactly.
    # Attach its separately documented residual bands, without lending them to
    # the correction model, whose uncertainty was never evaluated.
    baseline_bands = followup.loc[(followup.stage == 'retrospective') &
                                  (followup.id == 'weekday_mean')].set_index('date')
    baseline_mask = original.model_id == 'weekday_mean'
    baseline_dates = original.loc[baseline_mask, 'date']
    np.testing.assert_allclose(original.loc[baseline_mask, 'prediction'],
                               baseline_bands.loc[baseline_dates, 'prediction'], rtol=1e-13)
    for bound in ['lower', 'upper']:
        original.loc[baseline_mask, bound] = baseline_bands.loc[baseline_dates, bound].to_numpy()
    correction = followup.loc[(followup.stage == 'retrospective') &
                              (followup.id == 'lgbm_correction')].copy()
    correction['model_id'] = correction['id']
    keep = ['model_id', 'stage', 'issue', 'date', 'horizon', 'actual', 'prediction',
            'mase_scale', 'lower', 'upper']
    return pd.concat([original[keep], correction[keep]], ignore_index=True)


def assemble_tables(predictions, root=ROOT):
    """One scoring population for all models; native FM intervals stay separate."""
    scored = predictions.loc[predictions.scored.astype(bool)].copy()
    assert scored.groupby(['model_id', 'origin_id']).size().eq(28).all()
    stage = []
    for (model, period), frame in scored.groupby(['model_id', 'stage']):
        stage.append({'model_id': model, 'stage': period, **point_metrics(frame),
                      **interval_metrics(frame)})
    retrospective = pd.concat([
        scored.loc[scored.stage == 'retrospective'], reference_forecasts(root)
    ], ignore_index=True)
    summary, intervals, paired = [], [], []
    for population in ['all', 'exclude_november_2025']:
        selected = retrospective if population == 'all' else retrospective.loc[
            ~retrospective.date.str.startswith('2025-11-')]
        base = selected.loc[selected.model_id == 'weekday_mean']
        base_loss = base.assign(loss=(base.prediction - base.actual).abs()).groupby('issue').loss.mean()
        for model, frame in selected.groupby('model_id'):
            joined = frame.merge(base[['date', 'actual', 'mase_scale']], on='date',
                                 suffixes=('', '_baseline'), validate='one_to_one')
            assert len(joined) == len(frame) == len(base)
            np.testing.assert_allclose(joined.actual, joined.actual_baseline, rtol=0, atol=0)
            np.testing.assert_allclose(joined.mase_scale, joined.mase_scale_baseline, rtol=1e-12)
            summary.append({'model_id': model, 'population': population, **point_metrics(frame),
                            'mae_improvement_percent': 100 * (1 - point_metrics(frame)['mae'] /
                                                               point_metrics(base)['mae'])})
            if frame[['lower', 'upper']].notna().all().all():
                intervals.append({'model_id': model, 'population': population,
                                  **point_metrics(frame), **interval_metrics(frame)})
            losses = frame.assign(loss=(frame.prediction - frame.actual).abs()).groupby('issue').loss.mean()
            for issue, loss in losses.items():
                paired.append({'model_id': model, 'population': population, 'issue': issue,
                               'path_mae': loss, 'baseline_path_mae': base_loss.loc[issue],
                               'difference': loss - base_loss.loc[issue]})
    return {'stage_metrics': pd.DataFrame(stage), 'comparison': pd.DataFrame(summary),
            'interval_metrics': pd.DataFrame(intervals), 'paired_paths': pd.DataFrame(paired)}


def verify_foundation(root=ROOT):
    """Independently reconstruct all scores from cached quantiles and fixed inputs."""
    root = Path(root)
    folder = root / 'results/foundation_models'
    protocol = json.loads((root / 'provenance/foundation_models/protocol.json').read_text())
    lock = json.loads((root / 'provenance/foundation_models/protocol_lock.json').read_text())
    for relative, expected in lock['files'].items():
        assert digest(root / relative) == expected, relative
    target = read_target(root)
    predictions = pd.read_csv(folder / 'forecasts.csv')
    audits = json.loads((folder / 'inference_audit.json').read_text())
    assert len(audits) == 74 and len(predictions) == 2146
    rebuilt = []
    folds = schedule(root).set_index('origin_id')
    for audit in audits:
        model = audit['model_id']
        assert audit['revision'] == protocol['models'][model]['revision']
        cache = json.loads((folder / 'cache' / (model + '__' + audit['origin_id'] + '.json')).read_text())
        assert cache['protocol_sha256'] == digest(root / 'provenance/foundation_models/protocol.json')
        fold = folds.loc[audit['origin_id']].to_dict()
        fold['origin_id'] = audit['origin_id']
        context = context_for(target, fold['issue'], protocol['context_length'])
        assert context_digest(context) == audit['context_sha256'] == cache['context_sha256']
        assert audit['context_end'] == str(context.index[-1].date())
        rows, repairs = bridge_rows(target, fold, cache['raw_quantiles'], model)
        assert repairs == audit['repairs']
        rebuilt.append(rows)
    rebuilt = pd.concat(rebuilt, ignore_index=True)
    keys = ['model_id', 'origin_id', 'date']
    a, b = [f.sort_values(keys).reset_index(drop=True) for f in [predictions, rebuilt]]
    assert a[keys].equals(b[keys])
    for column in ['actual', 'prediction', 'lower', 'upper', 'mase_scale', 'raw_q10', 'raw_q50', 'raw_q90']:
        np.testing.assert_allclose(a[column], b[column], rtol=1e-12, atol=1e-10)
    tables = assemble_tables(rebuilt, root)
    for name, table in tables.items():
        saved = pd.read_csv(folder / (name + '.csv'))
        assert list(table.columns) == list(saved.columns) and len(table) == len(saved)
        for column in table:
            if pd.api.types.is_numeric_dtype(table[column]):
                np.testing.assert_allclose(table[column], saved[column], rtol=1e-11, atol=1e-9)
            else:
                assert table[column].tolist() == saved[column].tolist()
    return {'status': 'passed', 'procedure_origin_records': len(audits),
            'bridge_predictions': len(predictions), 'scored_predictions': int(predictions.scored.sum()),
            'models': list(protocol['models']), 'optimizer_fits': 0, 'fine_tuning': False,
            'new_future_paths': 0, 'verification_inference_calls': 0}
