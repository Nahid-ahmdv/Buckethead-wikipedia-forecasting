#!/usr/bin/env python3
"""Run the frozen inference-only extension or verify its saved results.

Default plan mode does not download weights, fit models, or overwrite results.
Each infer invocation loads one model; use separate processes to bound memory.
"""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import foundation_analysis as fm


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_protocol():
    protocol = json.loads(fm.PROTOCOL.read_text())
    lock = json.loads((fm.PROTOCOL.parent / 'protocol_lock.json').read_text())
    for name, expected in lock['files'].items():
        if fm.digest(ROOT / name) != expected:
            raise ValueError('Frozen extension input or implementation changed: ' + name)
    return protocol


def checkpoint(model_id, protocol, local_root=None):
    spec = protocol['models'][model_id]
    if local_root:
        directory = Path(local_root) / model_id
    else:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(spec['repository'], revision=spec['revision'],
                                          allow_patterns=['config.json', 'model.safetensors']))
    for filename, expected in spec['files'].items():
        if fm.digest(directory / filename) != expected:
            raise ValueError(f'Checkpoint content mismatch: {model_id}/{filename}')
    return directory


def load_predictor(model_id, directory, protocol):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(protocol['seed'])
    torch.use_deterministic_algorithms(True)
    if model_id == 'chronos2':
        from chronos import Chronos2Pipeline
        model = Chronos2Pipeline.from_pretrained(str(directory), device_map='cpu',
                                                 torch_dtype=torch.float32,
                                                 local_files_only=True)

        def predict(values):
            quantiles, _ = model.predict_quantiles(
                [np.asarray(values, dtype=np.float32)], prediction_length=29,
                quantile_levels=[0.1, 0.5, 0.9], context_length=1024,
                batch_size=1, cross_learning=False)
            return quantiles[0].squeeze(0).detach().cpu().numpy()
    else:
        import timesfm
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            str(directory), torch_compile=False, local_files_only=True)
        model.compile(timesfm.ForecastConfig(**protocol['models'][model_id]['forecast_config']))

        def predict(values):
            _, quantiles = model.forecast(horizon=29, inputs=[np.asarray(values, dtype=np.float32)])
            return quantiles[0, :29, :][:, [1, 5, 9]]
    return predict


def infer(model_id, output, local_root):
    protocol = check_protocol()
    for package, expected in protocol['essential_package_versions'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'Unexpected {package} version; use the recorded environment.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = protocol['models'][model_id]
    directory = checkpoint(model_id, protocol, local_root)
    load_started = time.perf_counter()
    predict = load_predictor(model_id, directory, protocol)
    loading_seconds = time.perf_counter() - load_started
    target = fm.read_target()
    audits = []
    protocol_sha = fm.digest(fm.PROTOCOL)
    for index, fold in enumerate(fm.schedule().to_dict('records'), start=1):
        context = fm.context_for(target, fold['issue'], protocol['context_length'])
        cache_path = output / 'cache' / (model_id + '__' + fold['origin_id'] + '.json')
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
            assert cache['protocol_sha256'] == protocol_sha
            assert cache['context_sha256'] == fm.context_digest(context)
            audits.append(cache['audit'])
            continue
        started = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                raw = np.asarray(predict(context.to_numpy()), dtype=float)
            _, repairs = fm.normalize_quantiles(raw)
        except Exception as error:
            write_json(output / (model_id + '_failure.json'), {
                'model_id': model_id, 'origin_id': fold['origin_id'],
                'error_type': type(error).__name__, 'error': str(error),
                'traceback': traceback.format_exc(), 'fallback': False})
            raise
        audit = {
            'model_id': model_id, 'origin_id': fold['origin_id'], 'stage': fold['stage'],
            'issue': str(fold['issue'].date()), 'context_start': str(context.index[0].date()),
            'context_end': str(context.index[-1].date()), 'context_rows': len(context),
            'context_sha256': fm.context_digest(context), 'revision': spec['revision'],
            'loading_seconds': loading_seconds, 'inference_seconds': time.perf_counter() - started,
            'repairs': repairs, 'fallback': False, 'optimizer_fits': 0,
            'warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in captured],
        }
        write_json(cache_path, {'protocol_sha256': protocol_sha, 'model_id': model_id,
                               'context_sha256': audit['context_sha256'],
                               'raw_quantiles': raw.tolist(), 'audit': audit})
        audits.append(audit)
        print(json.dumps({'model': model_id, 'completed': index, 'total': 37,
                          'issue': audit['issue'], 'seconds': audit['inference_seconds']}), flush=True)
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != 'Darwin':
        peak *= 1024
    write_json(output / (model_id + '_execution.json'), {
        'status': 'passed', 'model_id': model_id, 'revision': spec['revision'],
        'python': sys.version, 'platform': platform.platform(), 'packages': packages,
        'device': 'cpu', 'dtype': 'float32', 'threads': 1, 'seed': protocol['seed'],
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'origin_records': len(audits), 'new_optimizer_fits': 0, 'fine_tuning': False,
        'peak_process_memory_bytes': peak, 'total_inference_seconds': sum(a['inference_seconds'] for a in audits),
    })


def summarize(output):
    protocol = check_protocol()
    output = Path(output)
    target = fm.read_target()
    rows, audits = [], []
    for model_id in protocol['models']:
        for fold in fm.schedule().to_dict('records'):
            cache = json.loads((output / 'cache' / (model_id + '__' + fold['origin_id'] + '.json')).read_text())
            assert cache['protocol_sha256'] == fm.digest(fm.PROTOCOL)
            frame, _ = fm.bridge_rows(target, fold, cache['raw_quantiles'], model_id)
            rows.append(frame)
            audits.append(cache['audit'])
    forecasts = pd.concat(rows, ignore_index=True)
    forecasts.to_csv(output / 'forecasts.csv', index=False)
    write_json(output / 'inference_audit.json', audits)
    for name, frame in fm.assemble_tables(forecasts).items():
        frame.to_csv(output / (name + '.csv'), index=False)
    write_json(output / 'run_summary.json', {
        'status': 'passed', 'models': list(protocol['models']), 'origin_records': len(audits),
        'bridge_predictions': len(forecasts), 'scored_predictions': int(forecasts.scored.sum()),
        'optimizer_fits': 0, 'fallbacks': 0, 'new_future_paths': 0,
        'context_length_search': False, 'fine_tuning': False,
        'protocol_sha256': fm.digest(fm.PROTOCOL),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    print('Summarized', len(audits), 'procedure-origin records.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['plan', 'infer', 'summarize', 'verify'], default='plan')
    parser.add_argument('--model', choices=['chronos2', 'timesfm25'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--weights-root', type=Path)
    args = parser.parse_args()
    if args.mode == 'plan':
        protocol = check_protocol()
        print(json.dumps(protocol, indent=2))
    elif args.mode == 'verify':
        print(json.dumps(fm.verify_foundation(), indent=2))
    elif args.mode == 'infer':
        if not args.model or args.output is None:
            parser.error('infer requires --model and --output')
        infer(args.model, args.output, args.weights_root)
    else:
        if args.output is None:
            parser.error('summarize requires --output')
        summarize(args.output)


if __name__ == '__main__':
    main()
