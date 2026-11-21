#!/usr/bin/env python3
"""Replay fixed research recipes; default plan mode performs no fitting."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import contextlib
import datetime
import json
from pathlib import Path
import shutil
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'buckethead-delivery-mpl'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from integrity import verify_current
from research_adapter import FAILURE_POLICY, research_layout, load_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['development', 'original-evaluation', 'retrospective-followup'], required=True)
    parser.add_argument('--mode', choices=['plan', 'cached', 'refit'], default='plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verify_current()
    phase = {'development': 'phase1', 'original-evaluation': 'phase2', 'retrospective-followup': 'phase4'}[args.phase]
    group = {'phase1': 'development', 'phase2': 'original_evaluation', 'phase4': 'retrospective_followup'}[phase]
    report = {'phase': args.phase, 'mode': args.mode, 'fixed_recipe': True,
              'expected_origin_records': {'phase1': 324, 'phase2': 39, 'phase4': 288}[phase],
              'new_model_variants': 0,
              'adapter_failure_policy': FAILURE_POLICY}
    if args.mode == 'plan':
        print(json.dumps(report, indent=2))
        return
    if args.output is None:
        parser.error('--output is required for cached/refit mode')
    destination = args.output.resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error('Use a NEW output directory outside this repository')
    destination.mkdir(parents=True, exist_ok=False)
    historical = ROOT / 'results' / group
    if args.mode == 'cached':
        shutil.copytree(historical / 'cache', destination / 'cache')
    else:
        (destination / 'cache').mkdir()
    if phase == 'phase2':
        shutil.copyfile(historical / 'preflight.json', destination / 'preflight.json')
    with research_layout() as (root, forecasting, final, engine):
        if phase == 'phase1':
            runner = load_module('delivery_development_runner', root / 'phase1/run_development.py')
            runner.RESULTS, runner.CACHE = destination, destination / 'cache'
            predictor = forecasting
        elif phase == 'phase2':
            runner = final
            runner.RESULTS = destination
            predictor = final
        else:
            runner = load_module('delivery_retrospective_runner', root / 'phase4/run_experiment.py')
            engine.RESULTS = destination
            predictor = engine
        previous = predictor.predict
        if args.mode == 'cached':
            def forbidden(*unused_args, **unused_kwargs):
                raise RuntimeError('Cache replay attempted fitting or prediction')
            predictor.predict = forbidden
        try:
            with (destination / 'replay.log').open('w') as log, contextlib.redirect_stdout(log):
                runner.main()
        finally:
            predictor.predict = previous
    import numpy as np
    import pandas as pd
    filename = {'phase1': 'development_forecasts.csv', 'phase2': 'final_forecasts.csv', 'phase4': 'forecasts.csv'}[phase]
    old, new = [pd.read_csv(folder / filename) for folder in (historical, destination)]
    keys = [key for key in ['stage', 'origin_id', 'procedure_id', 'id', 'date'] if key in old]
    joined = old.merge(new, on=keys, suffixes=('_historical', '_replay'), validate='one_to_one')
    assert len(joined) == len(old) == len(new)
    difference = float(np.abs(joined.prediction_historical - joined.prediction_replay).max())
    if args.mode == 'cached':
        np.testing.assert_array_equal(joined.prediction_historical, joined.prediction_replay)
    report.update({'status': 'passed', 'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'maximum_prediction_difference': difference, 'daily_predictions': len(joined),
                   'prediction_calls_forbidden': args.mode == 'cached',
                   'historical_preflight_copied_as_evidence_only': phase == 'phase2'})
    (destination / 'reproduction_record.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
