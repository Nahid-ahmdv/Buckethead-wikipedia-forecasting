"""Verify saved scientific results, rendering inputs and local documentation links."""
import base64
import gzip
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
import nbformat
import numpy as np
import pandas as pd
from integrity import ROOT, sha, verify_current, historical_status, is_private_publication
import verify_saved

FIGURES = [
    'figures/development/01_attention_history.png',
    'figures/development/03_dependence_variability.png',
    'figures/historical_context/interview_case.png',
    'figures/development/05_development_comparison.png',
    'figures/original_evaluation/06_final_forecasts.png',
    'figures/diagnostics/09_level_adaptation.png',
    'figures/retrospective_followup/03_differencing_and_bias.png',
    'figures/retrospective_followup/04_correction_paired_paths.png',
    'figures/retrospective_followup/05_interval_diagnostics.png',
]

FOUNDATION_FIGURES = [
    'figures/foundation_models/01_point_comparison.png',
    'figures/foundation_models/02_paired_and_horizon.png',
    'figures/foundation_models/03_native_intervals.png',
    'figures/foundation_models/04_first_three_paths.png',
]


def check_links(text, base):
    count = 0
    links = re.findall(r'\]\(([^\s)]+)\)', text) + re.findall(r'(?:href|src)=["\']([^"\']+)', text)
    for link in links:
        parts = urlsplit(link)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        path = (base / unquote(parts.path)).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f'Link leaves this delivery: {link}')
        if not path.exists():
            raise FileNotFoundError(f'Broken local link from {base.relative_to(ROOT)}: {link}')
        count += 1
    return count


def notebook_checks(nb):
    nbformat.validate(nb)
    code = [cell for cell in nb.cells if cell.cell_type == 'code']
    assert len(nb.cells) == 63 and len(code) == 30
    assert [cell.execution_count for cell in code] == list(range(1, 31))
    assert not [out for cell in code for out in cell.outputs if out.output_type == 'error']
    images = [out.data['image/png'] for cell in code for out in cell.outputs if 'image/png' in out.get('data', {})]
    assert len(images) == len(FIGURES) + len(FOUNDATION_FIGURES) == 13
    for encoded, path in zip(images, FIGURES + FOUNDATION_FIGURES):
        assert hashlib.sha256(base64.b64decode(encoded)).hexdigest() == sha(ROOT / path), path
    original = json.loads(gzip.decompress((ROOT / 'provenance/original_notebook.json.gz').read_bytes()))
    original_images = [out['data']['image/png'] for cell in original['cells'] if cell['cell_type'] == 'code'
                       for out in cell['outputs'] if 'image/png' in out.get('data', {})]
    assert images[:9] == original_images
    source = '\n'.join(cell.source for cell in nb.cells)
    for phrase in ['326 of 364', 'retrospective post-evaluation', 'not a new untouched test',
                   'CC0 1.0 Universal', 'no future forecasts', 'does not refit models']:
        assert phrase in source, phrase
    return source


def verify(include_publication=False):
    current = verify_current()
    saved = verify_saved.verify()
    notebook = nbformat.read(ROOT / 'buckethead_wikipedia_forecasting.ipynb', as_version=4)
    source = notebook_checks(notebook)
    links = check_links(source, ROOT)
    for path in ROOT.rglob('*.md'):
        if any(part in {'.venv', '.git'} for part in path.relative_to(ROOT).parts):
            continue
        if not include_publication and is_private_publication(path.relative_to(ROOT)):
            continue
        links += check_links(path.read_text(), path.parent)
    if include_publication:
        article = (ROOT / 'publication/article.md').read_text()
        table = pd.read_csv(ROOT / 'results/publication/phase4_retrospective.csv')
        for label, rest in re.findall(r'^\| ([^|]+) \| ([^\n]+)$', article, flags=re.M):
            if label == 'LightGBM correction':
                numbers = [float(value.replace('*', '').strip()) for value in rest.strip('| ').split('|')]
                row = table.loc[table.id == 'lgbm_correction'].iloc[0]
                np.testing.assert_allclose(numbers, [round(row.mae, 2), round(row.rmse, 2), round(row.mase, 4),
                                                     round(row.mae_excluding_november_2025, 2)])
        assert len(re.findall(r'!\[[^\]]*\]\(([^)]+)\)', article)) == 5
    from foundation_analysis import verify_foundation
    foundation = verify_foundation()
    return {'status': 'passed', 'integrity': current, 'code_cells': 30, 'embedded_figures_verified': 13,
            'local_links_checked': links, 'private_publication_checked': include_publication,
            'article_figures': 5 if include_publication else None,
            'scientific_checks': {key: saved[key] for key in [
                'cached_origin_records_verified', 'daily_predictions_verified', 'summary_rows_recomputed',
                'delayed_baseline_paths_reconstructed', 'original_overprediction_days',
                'original_mean_signed_error', 'correction_improvement_percent',
                'masked_improvement_percent', 'original_intervals', 'phase4_intervals']},
            'foundation_checks': foundation,
            'new_optimizer_fits': 0, 'historical_archive_required': False}
