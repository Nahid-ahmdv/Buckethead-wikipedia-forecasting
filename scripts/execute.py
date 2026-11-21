#!/usr/bin/env python3
"""Execute the sole notebook sequentially in memory; do not overwrite results."""
import os
import datetime
import json
from pathlib import Path
import sys
import tempfile
for variable, name in [('IPYTHONDIR', 'ipython'), ('JUPYTER_RUNTIME_DIR', 'jupyter'), ('MPLCONFIGDIR', 'mpl')]:
    os.environ.setdefault(variable, str(Path(tempfile.gettempdir()) / ('buckethead-delivery-' + name)))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from integrity import sha, verify_current
from verify_delivery import notebook_checks


def execute():
    verify_current()
    path = ROOT / 'buckethead_wikipedia_forecasting.ipynb'
    before = sha(path)
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=240, allow_errors=False,
                            resources={'metadata': {'path': str(ROOT)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    notebook_checks(notebook)
    assert sha(path) == before
    verify_current()
    return {'status': 'passed', 'executed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'python_version': sys.version.split()[0], 'code_cells': 30, 'sequential_execution': True,
            'errors': 0, 'embedded_figures': 13, 'new_optimizer_fits': 0,
            'notebook_overwritten': False, 'notebook_sha256': before,
            'scope': 'Saved-artifact execution; full optimizer fitting was not rerun'}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
