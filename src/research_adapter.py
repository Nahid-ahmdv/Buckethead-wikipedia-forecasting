"""Use byte-preserved research implementations in a temporary historical layout.

Broad presentation-freeze callbacks use the current delivery guard. Prediction
entry points additionally reject unexpected errors recorded as fallbacks by the
historical runners. Original source bytes and configuration contracts stay
unchanged. No file from the author's working archive is used.
"""
from contextlib import contextmanager
from functools import wraps
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from integrity import ROOT, source_map, verify_current


FAILURE_POLICY = 'explicit-numerical-failures-only-v1'
_NUMERICAL_FAILURES = frozenset({'ModelFailure', 'LinAlgError'})


class UnexpectedForecastError(Exception):
    """A historical runner caught an error that must stop current execution."""

    def __init__(self, details):
        self.details = dict(details)
        super().__init__(
            f"Unexpected {details.get('error_type', 'unclassified error')} at "
            f"{details.get('issue', 'unknown issue')}: {details.get('error', '')}. "
            "The returned fallback was rejected before scoring or caching."
        )


def _strict_prediction(predict):
    """Keep successful results and explicit numerical fallbacks unchanged."""
    @wraps(predict)
    def checked(*args, **kwargs):
        prediction, details = predict(*args, **kwargs)
        if details.get('fallback') and details.get('error_type') not in _NUMERICAL_FAILURES:
            raise UnexpectedForecastError(details)
        return prediction, details
    return checked


def current_guard():
    return verify_current()['delivery_files_verified']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


@contextmanager
def research_layout():
    """Reconstruct original relative paths from recorded, unchanged file bytes."""
    current_guard()
    old_path = list(sys.path)
    names = ['forecasting', 'final_experiment', 'engine']
    previous_modules = {name: sys.modules.get(name) for name in names}
    with tempfile.TemporaryDirectory(prefix='buckethead-research-') as temporary:
        root = Path(temporary).resolve()
        for original, record in source_map().items():
            # Modified reader documentation and the descriptive helper are never
            # inputs to the preserved model implementations or source locks.
            if record['source_sha256'] != record['delivery_sha256']:
                continue
            target = root / original
            if not target.resolve().is_relative_to(root):
                raise ValueError(f'Invalid archived path: {original}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / record['path'], target)
        for name in names:
            sys.modules.pop(name, None)
        sys.path[:0] = [str(root / 'phase1/src'), str(root / 'phase2'), str(root / 'phase4')]
        forecasting = load_module('forecasting', root / 'phase1/src/forecasting.py')
        final = load_module('final_experiment', root / 'phase2/final_experiment.py')
        engine = load_module('engine', root / 'phase4/engine.py')
        callbacks = [(forecasting, 'phase0_integrity'), (final, 'original_integrity'), (engine, 'protect')]
        old_callbacks = [(module, attr, getattr(module, attr)) for module, attr in callbacks]
        for module, attr in callbacks:
            setattr(module, attr, current_guard)
        predictors = [(module, module.predict) for module in (forecasting, final, engine)]
        for module, predict in predictors:
            module.predict = _strict_prediction(predict)
        try:
            yield root, forecasting, final, engine
        finally:
            for module, predict in predictors:
                module.predict = predict
            for module, attr, value in old_callbacks:
                setattr(module, attr, value)
            sys.path[:] = old_path
            for name in names:
                sys.modules.pop(name, None)
                if previous_modules[name] is not None:
                    sys.modules[name] = previous_modules[name]
            current_guard()
