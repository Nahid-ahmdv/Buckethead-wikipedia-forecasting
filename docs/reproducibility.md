# Reproduce the recorded work

Run commands from the repository root. All documented execution inputs are
included here. The original working folder and the author's other projects are
not required. Network access is needed to install packages; saved-data execution
and cache replay use the preserved local snapshot.

## Environment

The tested numerical environment is CPython **3.10.18, macOS arm64**.
[requirements.txt](../requirements.txt) is the single setup file. It preserves
the 90-package combined environment used for the consolidated notebook and its
optional fitting tools, with the 26 additional foundation-model pins.
It includes native modeling dependencies even though notebook execution imports
no statistical or boosting engine. Native LightGBM/OpenMP is a host dependency;
new OS and cross-platform optimizer equivalence remain untested.

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

## Verify and execute saved results

```sh
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/execute.py
.venv/bin/python -m unittest discover -s tests -v
```

Verification checks every file in the delivery manifest, raw-versus-processed
counts, predictions against all 651 origin caches, 73 summary rows, delayed
baselines, original and retrospective interval rules, figures and local
Markdown links. Execution uses this interpreter for the Jupyter kernel and
runs all **30 code cells** sequentially in memory, showing **13 figures**.
It never overwrites the notebook or calls either foundation model. The kernel uses local loopback sockets.

Private Towards Data Science drafts, article-only figures and article table
exports are ignored by Git and are not required by these commands. Two figures
shared with the notebook are retained under `figures/retrospective_followup/`.
When the private files are available locally, their additional checks can be run
with `.venv/bin/python scripts/verify.py --include-publication`.

These are saved-artifact checks, with **zero optimizer fits**. They do not
establish independent confirmation on unseen data. Research snapshot:
November 5, 2026; observations end October 31, 2026.

## Independently reconstruct numerical results

```sh
.venv/bin/python scripts/replay_numerics.py
```

The unchanged independent verifiers reconstruct 13 original SARIMAX paths and
144 follow-up ETS/SARIMAX paths from saved parameters, and reconstruct historical
labels and baseline offsets for 36 matched LightGBM training pairs. No optimizer
is fitted and LightGBM is not retrained. Output is printed; historical reports
are never overwritten. [Verification evidence](../provenance/checks/README.md)
separates new packaging checks from dated research evidence.

## Replay complete fixed recipes

The [preserved research implementations](../src/research/README.md) are unchanged
source files. [The adapter](../src/research_adapter.py) copies their mapped
inputs into a temporary historical layout. It replaces broad historical
presentation-freeze callbacks with the current delivery guard and wraps the
prediction entry points with the failure policy below. Original source bytes,
execution contracts, candidate locks and chronology checks stay active. The
temporary copy is deleted after the run.

### Numerical fallback and unexpected errors

The preserved historical runners caught `ValueError` and `RuntimeError` as well
as explicit numerical failures. Those broad classes can also indicate invalid
configuration or programming errors. Every recorded real-data run had zero
fallbacks, so this distinction does not change any saved forecast or score.

The current adapter permits only logged `ModelFailure` and `LinAlgError`
fallbacks to reach scoring. A different recorded error, including a generic
`ValueError` or `RuntimeError`, raises `UnexpectedForecastError`; the rejected
prediction is not returned for scoring or caching. Its original error details
and traceback remain available on the exception. Errors that already propagate
from the historical runner continue to propagate. No optimizer retry is added.

The supported entry points apply this policy without rewriting the archived
engines. A reproduction record identifies the adapter policy explicitly;
cached replay retains the original predictions. The correction release record
is [preserved separately](../provenance/correction_release.json).

Inspect the three plans without fitting or writing result files:

```sh
.venv/bin/python scripts/reproduce.py --phase development
.venv/bin/python scripts/reproduce.py --phase original-evaluation
.venv/bin/python scripts/reproduce.py --phase retrospective-followup
```

To replay the complete runners using existing caches, select `cached` and a
new output directory **outside** this repository. Examples use relative paths
that are portable across working directories:

```sh
.venv/bin/python scripts/reproduce.py --phase development --mode cached --output ../buckethead-development-replay
.venv/bin/python scripts/reproduce.py --phase original-evaluation --mode cached --output ../buckethead-original-replay
.venv/bin/python scripts/reproduce.py --phase retrospective-followup --mode cached --output ../buckethead-followup-replay
```

Each directory must not exist beforehand. Cached mode forbids model prediction
calls and requires identical saved predictions. It executes 324 development,
39 original-evaluation and 288 follow-up origin records, respectively. The
original final preflight is copied as historical evidence, not presented as a
new preflight on the reader's machine.

For an intentional full reproduction of the fixed model-fitting recipes,
replace `--mode cached` with `--mode refit` and use different new output
directories. Refit mode uses empty caches, preserves the original settings,
and records differences from historical predictions. It does not replace the
source of truth, retune models or authorize new variants. The entire optimizer
pipeline was **not rerun during GitHub packaging**; numerical equivalence across
library builds, CPUs and operating systems is not guaranteed by the cache test.

## Integrity and historical scope

[manifest.json](../provenance/manifest.json) is the new delivery inventory.
[source_map.json](../provenance/source_map.json) records original relative paths,
source hashes, delivered paths and any documented transformations. All data,
prediction CSVs, fit caches, model implementations and historical contracts
retain their original bytes. Selected reader documents update local links.
The notebook changes its paths and reproduction notes; its scientific story,
figures, predictions and reported findings remain intact.

Historical manifests are preserved as dated evidence with explicitly partial
coverage in this selected package. Missing archive files are not described as
verified. The [archive guide](research_archive.md) documents the boundary. Hashes
are local integrity checks, not signatures, external preregistration or proof of
analyst blindness. Editing a sealed file requires a new documented release.

The manifest excludes itself, the private publication paths listed in
`.gitignore`, and `provenance/checks/`, which contains dated
execution evidence bound to the notebook and delivery manifest hashes. Default
verification commands are read-only and print their results.
The original GitHub delivery manifest and source map are preserved under
`provenance/historical/`; the current manifest records their successor.

The root dependency files were consolidated into `requirements.txt` with all
64 version pins unchanged. The former direct-dependency list is retained only
as a historical record. Source mapping reconstructs both original dependency
files for archived workflows; readers install from the single root file.


## Foundation-model extension

The [foundation-model guide](foundation_models.md) documents the separately
frozen inference-only comparison, checkpoint hashes, exposure limitations and
optional replay. Normal notebook execution reads those saved results without
loading PyTorch, Chronos or TimesFM. Default verification now checks 74
additional model-origin caches, 2,146 bridge predictions and 2,072 scored
predictions, alongside the unchanged original evidence.

The combined root `requirements.txt` was installed successfully into a new
isolated Python 3.10.18 environment on this macOS host; `pip check` passed.
This does not establish clean-OS or cross-platform numerical equivalence.
The author's explicitly named notebook copy is ignored by delivery checks and
Git. Other unexpected notebooks still fail the single-main-notebook check.
