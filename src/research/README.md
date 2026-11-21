# Preserved research implementations

These files retain the original implementation bytes so that candidate grids,
features, information boundaries, estimator settings and source-lock signatures
remain inspectable. `development/` contains the original forecasting helpers and
runner; `original_evaluation/` contains the frozen final experiment;
`retrospective_followup/` contains the bounded extension.

Use [scripts/reproduce.py](../../scripts/reproduce.py) and
[scripts/replay_numerics.py](../../scripts/replay_numerics.py), following the
[reproduction guide](../../docs/reproducibility.md). The original scripts expect
their historical directory layout; the wrapper reconstructs that layout in a
temporary directory. Running those archived source files directly from their
new locations is not the supported entry point.

The adapter replaces broad historical presentation-freeze callbacks and applies
the current failure policy at prediction entry points. Explicit numerical
fallbacks remain eligible; unexpected errors stop execution. Original source
bytes, configuration locks and numerical methods remain intact. See the
[failure-policy explanation](../../docs/reproducibility.md#numerical-fallback-and-unexpected-errors).
All new fitting outputs must go to a fresh directory outside this repository.
