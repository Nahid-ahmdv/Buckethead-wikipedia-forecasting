# Phase 1 implementation record — before development fitting

Authority: the user's Phase 1 brief, reviewed November 5, 2026. Phase 0 v1,
datasets, event ledger, schedule, and hashes remain unchanged. Final evaluation
remains 2025-11-02–2026-10-31 and is not run in this phase.

## Consistency review and clarifications

The two-day observation cutoff and the two development boundary blocks are
consistent: twelve tuning and twelve selection paths are scored. The thirteenth
block in each stage supplies later training observations only as they become
available. It is not a thirteenth scored development path. No additional model
family, parameter candidate, target transformation, or event feature is added.

The following underspecified implementation details are resolved before fitting:

1. LightGBM training uses **complete mature 28-label paths only**, S+27 <= O-2.
   Every path's origin S lies inside the selected history window. This resolves
   the ambiguous treatment of partially observed paths without importing any
   future label. Past features can use observations preceding the window.
2. Past rolling standard deviations use ddof=0. Rolling summaries cover **7 and
   28 days**, both ending at S-2; counts at offsets 0,1,6,7,13,27,55 are included.
   Training origins are Sundays spaced seven days apart, as are evaluation
   origins (spaced 28 days). The annual basis uses period 365.25 and fixed epoch
   2015-01-01. Features and parameters are recorded with every fit.
3. Annual SARIMAX regressors are fixed sine/cosine terms K=2. Fit uses statsmodels
   L-BFGS, maxiter=200, stationarity/invertibility enforcement, no trend, default
   state-space differencing. A nonconverged or nonfinite fit is rejected. Prophet
   uses MAP estimation with seed=42 and uncertainty_samples=0 (its unused native
   intervals are not simulated); documented additive seasonality orders and
   priors remain fixed. Its optimizer's explicit convergence result is checked.
   LightGBM uses deterministic CPU behavior, one thread and force_col_wise=True;
   other installed estimator defaults are logged. No early stopping.
4. Tuning configurations are compared on all twelve scheduled paths with logged
   naive fallback errors included if a numerical fit fails. Configuration ties
   use lexicographic JSON parameter order. Selection compares the resulting
   eight procedures; any failed selection fit makes that procedure ineligible,
   as Phase 0 requires. All unsuccessful results remain in the artifacts.
5. Fixed-origin features and scores use only the development CSV. The runner
   refuses final origins and rejects final rows in its loader. Default fitting
   functions receive already truncated histories, never complete future targets.
   Development EDA is restricted further to information available before the
   selection stage: through 2024-11-01. No model decision is based on final data.
6. Phase 0's verifier contains a phase-specific assertion that the entire project
   has no notebook. It is preserved. Phase 1 instead verifies every file in the
   original freeze manifest; adding an authorized Phase 1 notebook does not
   modify or invalidate the original record.

The prior status update's wording about standard deviations referred to the
7- and 28-day rolling summaries, not seven separate standard deviations.

## Fixed stopping point

Implement and run development only; select the procedure; calibrate the fixed
80% pooled absolute-error interval from its 336 selection-era errors. Report
that this calibration block also selected the procedure. Native model intervals
are not compared. No final-period forecast, error, ranking, event response, or
interval coverage is computed. Freeze the proposed final configuration for user
review. Future final refits may use earlier final observations only after their
O-2 availability cutoff, under the unchanged Phase 0 policy.

These are pre-fit clarifications, not corrections driven by observed performance.
An implementation/input hash in every cache key prevents accidental reuse after
a source change. Unanticipated failures are reported; programming errors must
be fixed transparently rather than represented as scientific model failures.
