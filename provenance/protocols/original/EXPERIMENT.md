# Experimental protocol v1 — frozen after Phase 0 integrity checks

Status: fixed before any forecasting model or final outcome exploration.
This file specifies the next phase; it does not authorize execution during Phase 0.
The freeze manifest timestamps and hashes the exact specification and inputs.
Any later amendment must be recorded before fitting, with its reason and outcome
exposure. Preserve this version and its manifest.

## Target, cutoff, and chronology

Forecast 28 daily canonical-title pageview counts, issued at 00:00 UTC every
28 days. Target starts at the issue date. Snapshot cutoff is 2026-10-31, chosen
as the last complete calendar month before client date 2026-11-05, independently
of outcomes. Full collection begins 2015-07-01.

| Role | Target dates inclusive | 28-day blocks | Days |
|---|---|---:|---:|
| Earlier history | 2015-07-01–2023-11-04 | Training context only | 3,049 |
| Hyperparameter tuning | 2023-11-05–2024-11-02 | 12 scored + 1 boundary | 364 |
| Procedure/window selection | 2024-11-03–2025-11-01 | 12 scored + 1 boundary | 364 |
| Final evaluation | 2025-11-02–2026-10-31 | 13 | 364 |

Every origin, target end, and information cutoff is enumerated in
`split_schedule.csv`. There are 26 development paths, not 26 paths plus a
separate hidden tuning dataset. Earlier history is available for training.

**Publication-delay assumption:** at issue date O, historical target and event
observations may extend only through calendar date **O-2**. Yesterday's complete
count is not assumed available at midnight. A forecast spans day O through
O+27, corresponding to steps 2–29 after the last observed day. Statistical models
must forecast the intervening day O-1 too and discard it for scoring. This
24-hour publication buffer is an explicit assumption, not verified historical
availability: no as-of API snapshots exist here. Calendar covariates are known.
Date-only announcement evidence becomes usable from the following UTC day.

The final tuning origin's labels and the final selection origin's labels must
also satisfy this latency. To avoid using the last target day before it could
have been published, **tuning and selection summaries use the first 12 paths
in their respective blocks**. The 13th path of each block is a predeclared
boundary observation block, with no required scored forecast. Its observations
may enter subsequent training only when available. Thus tuning decisions are available before selection begins;
all selection and calibration decisions are available before final evaluation.
The selection boundary path ending 2025-11-01 is not used to select or calibrate
the procedure issued 2025-11-02. See the uncertainty plan.

## Primary and sensitivity populations

Primary scoring retains all 364 final days exactly as observed in the retrieved
snapshot. November 2025 has an unrepaired global incident; article-specific
influence is unknown. No percentage correction or spike deletion is justified.

Predefined sensitivity: rescore **the identical saved forecasts** after removing
target dates 2025-11-01–2025-11-30. The final period loses 29 dates, retaining
335 dates across 12 contributing paths, one partial. Compute pooled daily MAE,
RMSE, scaled absolute errors, interval coverage and mean width over retained
dates; do not average unequal-length path means equally. This is a scoring
sensitivity, not a refit, an estimate of corrected human traffic, or a cure for
November contamination in subsequent training data. Report both populations;
the sensitivity cannot select the winning method or change the primary target.
Corrected historical periods remain in the primary data with provenance flags.

## Candidate procedures and compact tuning

All predictions are finite and clipped at zero before scoring. No test-driven
rounding or transformation. Scores remain on the original count scale.

| Family | Candidate configurations per training window |
|---|---|
| Weekly seasonal naive | Most recent observed value for each target weekday; repeat that historical week, respecting O-2 |
| Recent weekday mean | Mean for each weekday over W in {4,8} completed weeks ending at O-2 |
| SARIMAX | orders {(1,0,0),(1,1,0),(1,0,1),(1,1,1)}, seasonal order (0,1,1,7); fixed annual Fourier K=2 as deterministic regressors |
| Prophet | weekly Fourier order 3, yearly Fourier order 3, daily seasonality off, additive mode, changepoint prior scale in {0.01,0.1}; other defaults recorded at implementation |
| LightGBM | raw target, objective regression_l1, num_leaves in {7,15}, 300 trees, learning_rate=.05, min_child_samples=50, seed=42, one CPU thread; other defaults recorded |

For SARIMAX, Prophet, and LightGBM compare expanding history with the trailing
three calendar years ending at O-2. Baselines have their explicit fixed history
rules; they are not duplicated as nominal expanding/three-year candidates.
LightGBM may use earlier observations solely to construct past features at the
window's boundary; eligible training labels/origins are restricted to the chosen
window. SARIMAX uses `enforce_stationarity=True`, `enforce_invertibility=True`,
`maxiter=200`; save warnings, convergence, parameters, and likelihood diagnostics.
Reject nonfinite or nonconverged fits; no absolute AIC cutoff. Do not compare
raw-scale and transformed-scale AIC. There is no transformation-search arm in v1.

For each fitted family/window, choose hyperparameters on the first 12 tuning
paths by mean path MAE. Fix weekday-mean W there too. On the first 12 selection
paths compare the two baselines and six tuned family/window procedures. Choose
lowest mean path MAE. Numerical ties within 1e-9 use: weekly naive, weekday
mean, SARIMAX, Prophet, LightGBM; within a family prefer trailing three years,
then lexicographic parameter order. Different families have explicitly different
small budgets; no claim of optimally tuned family superiority is supported.

Require complete valid forecasts on every selection path. A fitting failure
uses weekly naive as an explicitly logged fallback for that origin; a procedure
with any such failure is ineligible for primary selection. If all fitted
procedures fail, select a baseline. Once selected, refit with frozen settings
at each final origin. A final fitting failure uses the same logged fallback;
retain its errors and report failure counts, without replacing the procedure.

MAE rewards conditional medians, whereas some statistical procedures supply
conditional means. This is a comparison of specified forecasting procedures
under MAE, not a claim that every method targets the identical optimal functional.

## Genuine fixed-origin LightGBM

Use one horizon-conditioned direct model with h in {1,...,28}. Historical
training issue dates are spaced seven days apart, aligned to the same weekday
as evaluation origins. For each training issue date S, construct one row per h
with features calculated from observations no later than S-2. Training rows at
current issue O must have their target dates no later than O-2; exclude immature
labels. Keep the full 28-label path together when making validation splits.

Past features: count at cutoff and cutoff offsets {1,6,7,13,27,55}; past 7- and
28-day mean and standard deviation; known target weekday and annual sine/cosine
terms K=2; horizon h. All rolling statistics end at the issue's information
cutoff, not at the target date. No supplied future counts, backward filling,
or rolling statistics computed on the full dataset. No feature selection on
final outcomes. Event features are permitted only by the separate event policy.

Required verification before fitting: changing any count on or after O-1 must
leave every feature and predicted path issued at O unchanged. In particular,
day-20 features cannot use actual day-13 counts. Verify label maturity too.
For example, at the first final issue, 2025-11-02, historical values stop at
2025-10-31. Day 20 is 2025-11-21 and actual day 13 is 2025-11-14; neither future
count can enter any feature for that issue. This is a design assertion;
executable feature checks remain an implementation prerequisite, not a completed
modeling test in Phase 0.

## Metrics and common MASE scale

Primary MAE is the mean absolute error over the complete paths. Also report
pooled RMSE = sqrt(mean(all squared errors)), the 13 path MAEs, and weekly MASE.
Primary complete paths have equal lengths, so pooled MAE equals mean path MAE.
MAE selects; RMSE and MASE never break an inconvenient MAE result.

At issue O let C=O-2 and R be dates in (C minus three calendar years, C]. Define
q(O) = mean(|y[t]-y[t-7]|) over pairs for which both dates lie in R and both
values are observed. This **same q(O)** scales every model and both training
windows at that origin, including baselines. It is independent of each model's
training-window choice and contains no forecast targets. Final MASE is the
mean of |error|/q(O) across scored days. Log reference dates, valid pair count,
and denominator at implementation. If q is zero or undefined, MASE is missing
with an explicit reason; MAE and RMSE remain valid. Do not silently substitute 1.

Thirteen final origins give a limited description of one period. Report effect
sizes and path variability; no new final leaderboard search, p-value campaign,
or general superiority claim. Event associations are observational.

## Final roster, exposure, and stopping rule

Final reporting contains the selected pipeline plus the two frozen baselines
(deduplicate if a baseline wins). Add one matched event-aware counterpart only
if the pre-model event gate passes; current ledger fails that gate. Its settings
must be frozen before final evaluation. Do not score all unselected fitted
families on the final period as another selection contest.

Phase 0 read full values into memory for mechanical checks and wrote the clean
dataset. Only date/schema/validity/zero and access-reconciliation results were
examined for final dates. Development screening was limited to dates before
2025-11-02 and did not drive event selection. Earlier discussion browsed current
biographical and event pages, without Buckethead pageview outcomes. No model,
forecast error, final trend, or final event-response interpretation exists.

The experiment remains a retrospective evaluation using a revised snapshot.
**Stop after Phase 0.** Implementation, fitting, the main notebook, final
performance inspection, and the article belong to a subsequent authorized phase.
