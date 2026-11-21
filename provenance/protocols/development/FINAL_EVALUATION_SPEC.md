# Proposed frozen final-evaluation specification

Status: development complete; **user review is required before final evaluation**.
This file and `results/proposed_final_configuration.json` describe the proposal;
there are no final predictions or errors. The original Phase 0 record is preserved.

## Selected procedure

- Library: statsmodels 0.14.5.
- Family: SARIMAX, order (1,0,1), seasonal order (0,1,1,7), trend=None.
- Endogenous target: raw user-classified canonical-title daily counts.
- History: (C minus three calendar years, C], C=issue date minus two days.
- Exogenous regressors: sin/cos annual Fourier j=1,2; period 365.25 days;
  fixed epoch 2015-01-01. No event regressors or fitted calendar encodings.
- Fit: fresh fit at every origin, L-BFGS, maxiter=200, stationarity and
  invertibility enforcement, default full state-space differencing. No target
  transformation, warm-start search, or final retuning. Native threads limited to one.
- Prediction: 29 dates C+1 through O+27, discard O-1, retain O through O+27.
  Clip point predictions at zero. No integer rounding before scoring.
- Failure: nonconvergence/nonfinite outputs trigger a recorded weekly-naive
  fallback for that origin; retain its outcome in the selected procedure's score.

Companion baselines: most recent available weekday value repeated through the
whole path, and each weekday's average over the latest 56 available days.
Both obey the identical O-2 cutoff. No new baselines or candidate rankings.

## Uncertainty and scoring

- Fixed nominal 80% interval half-width q=408.286528545692.
  Calibration uses rank 270 of 336 absolute errors from the selected procedure
  on the twelve selection paths; the block also selected that procedure.
- Bounds: [max(0,p-q),p+q] after clipping p. No recalibration on final actuals.
- Primary MAE; pooled RMSE; common training-only weekly MASE scale per origin,
  constructed from valid pairs fully inside the same three-year reference.
- Empirical daily interval coverage and mean width; describe nominal versus
  observed coverage. No distribution-free or operational coverage guarantee.
- Thirteen nonoverlapping 28-day paths, 2025-11-02–2026-10-31, exactly as the
  original schedule. Later issues may use earlier final observations only after
  their O-2 cutoff. Configurations and the interval half-width remain fixed.
- Primary population: all 364 days. Sensitivity: remove November 2025 target
  dates from identical saved forecasts, leaving 335 daily-weighted dates. Do not
  refit for the sensitivity or treat it as corrected human-traffic truth.

## Implementation controls

The Phase 1 implementation SHA-256 is
`e4e7a0eda67cfe800209a56fc8b4f7a10841e959abbb2231b5b4785a9520460e`.
The scored development predictions SHA-256 is
`4b6cdd3753f52a19455c64c96919c6c0412260fdb96ba5a6548b1cda669e2a0d`.
The Phase 1 freeze manifest protects the source, environment, caches, notebook,
tests and proposal. Original Phase 0 inputs remain separately protected.

Phase 1 deliberately rejects final origins. A separately authorized final entry
point will be needed; do not edit the preserved Phase 1 implementation or
silently remove its guard. Its chronological behavior must match the verified
development contract. There is no final-actual loader or evaluation command in
this delivered phase.

## Numerical caveat carried forward

All selection fits converged, but the selected seasonal MA coefficient is near
-1. The coefficient audit and two exact unchanged replay checks are preserved
in `results/selected_coefficient_audit.csv` and `results/numerical_replay.json`.
They motivate caution about identification and numerical portability, not a
post-hoc method replacement. Keep the proposed configuration and failure rule;
any further change requires a recorded pre-final amendment and user review.
