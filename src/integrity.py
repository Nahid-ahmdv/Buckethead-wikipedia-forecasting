"""Integrity checks for this selected GitHub delivery, independent of its archive."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'provenance'

PRIVATE_PUBLICATION_DIRECTORIES = (
    'publication', 'figures/publication', 'results/publication',
    'src/research/publication', 'provenance/publication', 'provenance/claims',
)
PRIVATE_PUBLICATION_FILES = {
    'provenance/visual_manifest.json',
    'provenance/historical/publication__freeze_manifest.json',
}


def is_private_publication(path):
    """Article-only paths excluded from the public delivery and its guard."""
    relative = Path(path).as_posix()
    return relative in PRIVATE_PUBLICATION_FILES or any(
        relative == folder or relative.startswith(folder + '/')
        for folder in PRIVATE_PUBLICATION_DIRECTORIES)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_map():
    return json.loads((HERE / 'source_map.json').read_text())['files']


def resolve_source(original_path):
    """Translate a preserved research path without depending on the old folder."""
    return ROOT / source_map()[str(original_path)]['path']


def verify_current():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f'Required delivery file missing: {name}')
        if sha(path) != expected:
            raise ValueError(f'Delivery file changed: {name}')
    # The author's explicitly named pre-extension backup is a private local
    # artifact. It is never executed, edited, or treated as a reader entry point.
    notebooks = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.ipynb')
                       if p.relative_to(ROOT).as_posix() != 'buckethead_wikipedia_forecasting copy.ipynb'
                       and not any(x in {'.venv', '__pycache__', '.ipynb_checkpoints', '.git', '.venv-foundation'}
                                  for x in p.relative_to(ROOT).parts))
    if notebooks != ['buckethead_wikipedia_forecasting.ipynb']:
        raise ValueError(f'Expected one notebook, found {notebooks}')
    records = source_map()
    for old, record in records.items():
        if sha(ROOT / record['path']) != record['delivery_sha256']:
            raise ValueError(f'Source mapping changed: {old}')
    return {'status': 'passed', 'delivery_files_verified': len(manifest['files']),
            'mapped_source_artifacts_verified': len(records), 'notebooks': notebooks,
            'scope': 'Included delivery files; omitted working-archive files are not verified'}


def historical_status():
    """Report explicit partial coverage of immutable historical manifests."""
    mapping = source_map()
    policy_path = resolve_source('consolidation/research_manifest_v1.json')
    policy = json.loads(policy_path.read_text())
    revised = set(policy['superseded_presentation_paths'])
    rows = []
    for original, base in [('protocol/freeze_manifest.json', ''),
                           ('phase1/freeze_manifest.json', 'phase1'),
                           ('phase2/freeze_manifest.json', ''),
                           ('phase3/freeze_manifest.json', 'phase3'),
                           ('phase4/freeze_manifest.json', 'phase4')]:
        manifest = json.loads(resolve_source(original).read_text())
        same = relocated = superseded = omitted = 0
        for relative, expected in manifest['files'].items():
            old = str(Path(base) / relative)
            if old not in mapping:
                omitted += 1
                continue
            record = mapping[old]
            if record['source_sha256'] != expected:
                if old not in revised:
                    raise ValueError(f'Unexplained historical source change: {old}')
                superseded += 1
            elif sha(ROOT / record['path']) == expected:
                same += 1
            else:
                if record['change'] not in {'local Markdown links updated only',
                    'same describe function; figure-writing entry point omitted'}:
                    raise ValueError(f'Unexplained delivery transformation: {old}')
                relocated += 1
        rows.append({'historical_manifest': original, 'retained_bytes_verified': same,
                     'documented_delivery_changes': relocated,
                     'previously_superseded_presentations': superseded,
                     'omitted_archive_files_not_verified': omitted,
                     'full_historical_archive_verified': False})
    return rows
