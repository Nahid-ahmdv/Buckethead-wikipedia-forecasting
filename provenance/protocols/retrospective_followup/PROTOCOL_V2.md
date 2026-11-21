# Phase 4 v2 — Targeted forecasting improvements

Active version under AMENDMENT_01.md; v1 remains preserved in PROTOCOL.md.

Authority: the user's approved six-procedure Phase 4 brief, November 5, 2026.
This protocol and its machine-readable schedule/configuration are locked before
any Phase 4 candidate fitting. Local hashes record integrity, not external
preregistration, analyst blindness or a restored untouched historical test set.

## Question and scope

Can an adaptive procedure outperform the eight-week weekday average? Exactly
six candidates and two fixed baselines; no hyperparameter grid, event features,
new families, added history windows or performance-driven search expansion.
The 10% relative MAE aspiration is descriptive, not a stopping/acceptance rule.
All Phases 0–3 and raw data remain unchanged; all new files live in phase4/.
The only target is the frozen canonical-title, all-access, user-classified
daily English Wikipedia series through 2026-10-31. Revised-history and assumed
historical publication availability limitations carry forward.

## Exact chronology

All historical issues are Sundays at 00:00 UTC, spaced 28 days. At issue O,
C=O−2; no observation later than C enters a feature, label, fitted parameter,
level estimate or MASE denominator. Statistical models predict C+1 through
O+27 (29 dates), discard O−1, then score O through O+27. Both baselines repeat
only information available at C. Later origins can use earlier observed targets
after maturity; this never permits daily refreshing inside an issued path.

| Role | Issues inclusive (every 28 days) | Target dates inclusive | Paths |
|---|---|---|---:|
| Earlier development diagnostics | 2023-11-05 through 2024-09-08 | 2023-11-05 through 2024-10-05 | 12 |
| Point-procedure selection | 2024-11-03 through 2025-04-20 | 2024-11-03 through 2025-05-17 | 7 |
| Interval calibration only | 2025-06-15 through 2025-09-07 | 2025-06-15 through 2025-10-04 | 4 |
| Retrospective post-evaluation comparison | 2025-11-02 through 2026-10-04 | 2025-11-02 through 2026-10-31 | 13 |

The 2025-05-18–2025-06-14 path is an additional unscored boundary observation
block: its last label is unavailable at the first calibration issue's cutoff.
Original boundary observation blocks (2024-10-06–11-02 and
2025-10-05–11-01) remain unscored. They can enter later training through C.
Six candidates and both baselines receive predictions on all 36 listed paths:
288 procedure-origin records, 8,064 daily predictions, 216 statistical/ML fits
plus 72 baseline paths. A fit failure produces a recorded baseline fallback,
not a missing score. No observed real-series fit is a preflight smoke test.

The earlier 12 paths describe behavior, not a parameter search. Select the
point procedure using only the seven selection paths, before calculating
calibration or retrospective candidate scores. Primary selection is lowest
pooled MAE among all eight procedures with zero selection fitting failures.
Tie tolerance 1e-9; fixed priority weekday mean, weekly naive, ETS level,
ETS damped, SARIMAX D0, SARIMAX D1, LightGBM raw, LightGBM correction.
Save the chosen identity/source/selection-score hash in a once-only record.

All historical material was already studied in earlier phases. The new
chronological separation prevents Phase 4 calibration errors being used in its
point-selection rule; it does not make these errors independent of all prior
research choices. All later-year Phase 4 performance is post-evaluation
exploratory evidence. No confirmatory p-value or universal superiority claim.

## Exactly six candidates

1. **ets_level:** ETS(A,N,A), additive error and weekly additive seasonality,
   no trend, seasonal_periods=7. Latest three calendar years (C−3 years,C].
2. **ets_damped:** ETS(A,Ad,A), same error/weekly seasonality/history, with an
   additive damped trend. This is a separate trend hypothesis, not a synonym
   for level adaptation. Both use statsmodels ETSModel, estimated initialization,
   default admissible parameter bounds and one L-BFGS fit, maxiter=1000.
   Divide training counts by their training mean before fitting and multiply
   all 29 forecasts back. The positive scale is estimated solely through C.
   Both seasonality specifications are fixed now; no annual ETS terms.
3. **sarimax_d0:** order=(1,0,1), seasonal_order=(0,0,1,7).
4. **sarimax_d1:** order=(1,0,1), seasonal_order=(0,1,1,7).
   Both use identical trailing three-year histories, annual Fourier K=2,
   period=365.25 and epoch=2015-01-01, trend=None, stationarity/invertibility
   enforcement and full state-space differencing. Fit one L-BFGS, maxiter=500.
   On each training history calculate m=mean(y) and s=population std(y); fit
   z=(y−m)/s and invert forecasts as m+s*z_hat. If s=0 use s=1. This supplies
   D0 with a nonzero origin-available level and applies exactly the same affine
   treatment to D1. It introduces no unidentified constant regression in the
   integrated model and no drift term in one arm only. Compare this matched
   pair; neither is an exact replay of the original unscaled SARIMAX. Preserve
   the original saved forecasts as a historical reference, not a seventh fit.
5. **lgbm_raw:** original expanding-history LightGBM specification (the better
   original LightGBM window): regression_l1, num_leaves=7, n_estimators=300,
   learning_rate=.05, min_child_samples=50, seed=42, n_jobs=1,
   deterministic=True, force_col_wise=True, verbosity=-1. No early stopping.
6. **lgbm_correction:** same estimator, expanding history, rows and features;
   predict y[target]−b_S[target_weekday], then add b_O at evaluation. Here b_S
   is the eight-week weekday mean constructed from the 56 days ending S−2,
   separately for each historical training issue S. Never construct training
   corrections using a baseline from the later evaluation origin or future
   target observations. Residual labels can be negative and are not clipped.

Both LightGBM variants use the original frozen feature construction: counts at
cutoff offsets 0,1,6,7,13,27,55; means/std(ddof=0) over 7 and 28 past days;
target weekday, horizon 1–28 and the same annual Fourier K=2. Weekly training
issues are Sundays and only full mature paths S+27<=C are included. Training
paths overlap; rows are not independent observations for inference. Features,
row order, X hash and hyperparameters must match within each pair. Only target
representation/add-back differs. This is one coherent engineering hypothesis.

ETS/SARIMAX history is three years; LightGBM retains the original expanding
control's history. This is a comparison of specified procedures, not a pure
cross-family history intervention. No alternative window is searched.

## Baselines, numerical policy and metrics

Fixed baselines: last available value for each weekday, and each weekday's
mean across the latest 56 available days. Same O−2 cutoff. All point forecasts
are clipped at zero, without rounding, after inverse transforms/add-back.
Numerical exceptions, nonconvergence, nonfinite likelihood/parameters/forecasts
or invalid forecast shape trigger a single logged weekly-naive fallback.
No optimizer retries, changed starts or post-result tolerances. Programming
errors propagate and require a documented correctness fix, not silent fallback.
Any selection failure makes a procedure ineligible for point selection; all
later errors/fallback paths remain in the scored result. Near-boundary finite
invertible fits are retained and disclosed. Save parameters, warnings,
optimizer return values, AR/MA roots, training dates and transformations.

Primary MAE; pooled RMSE=sqrt(mean daily squared error); common weekly MASE
scale at each issue computed from pairs fully inside (C−3 years,C]. Zero or
undefined scale yields missing MASE with explicit counts, never a substitute.
Signed error is prediction−actual; report mean signed error, overprediction
frequency, path/horizon scores, paired MAE differences and relative improvement
against both baselines. Describe path wins, median differences and worst paths;
no assumption of independent origins or extra significance tests.

Primary retrospective scoring retains all 364 days. Also carry forward the
predefined score-only November 2025 sensitivity: same forecasts, no refits,
335 retained dates, daily weighting, first path omitted and second 27 days.
No threshold-based exclusions or article-specific traffic correction.

## One frozen 80% interval construction

Apply one signed-error method to the development-selected point procedure and
the eight-week weekday baseline (deduplicate if that baseline is selected).
Use their 112 calibration predictions on 2025-06-15–2025-10-04, each produced
with its own historical cutoff. Calibration completes before the first
retrospective issue. Use r=y−p after point clipping, sorted ascending.
Minimum required sample is all 112 finite errors; if unavailable, report
calibration failure rather than shrinking n or substituting errors.

For n=112, lower rank floor((n+1)*.10)=11, upper rank
ceil((n+1)*.90)=102, one-indexed, with no interpolation. Bounds for every
retrospective point p are [max(0,p+r_(11)), max(0,p+r_(102))]. Upper clipping
also enforces nonnegative support if both signed offsets are negative.
Signed offsets can yield asymmetric, shifted bands; they do not alter the
reported point forecasts. Freeze these two offsets across all 13 retrospective
issues: no online updates, horizon-specific quantiles, radius search or
post-evaluation bias correction. This pooling is deliberate given only four
calibration paths and does not assert horizon-conditional coverage.

Report inclusive daily/pointwise coverage, average width, lower/upper misses,
per-path and per-horizon coverage, and all-28-days simultaneous path coverage
as a separate descriptive proportion, never as an 80% path guarantee.
Interval score at alpha=.2:
IS=(upper−lower)+10*(lower−y) if y<lower, or
(upper−lower)+10*(y−upper) if y>upper, otherwise width.
Pool daily scores; report the same score-mask sensitivity. No native model
interval comparison or unsupported distribution-free guarantee. Selection
reuse in earlier research, dependent errors, small n and changing conditions
remain limitations despite the new calibration separation.

## Future nomination and stopping

The development-selected identity is not replaced after seeing retrospective
scores, and its interval evaluation remains attached to it. Separately nominate
the lowest retrospective pooled-MAE procedure among the eight with zero
retrospective fit failures for future observation, using the same tie priority.
This nomination explicitly uses the already-examined year for exploratory
selection; it is not an independently validated winner. If the nominee differs
from the development selection, its interval performance has not been evaluated
under this two-procedure interval comparison and must not be borrowed from it.
Retain the weekday baseline as the prospective reference in all cases.

Planned genuinely future issues: 2026-11-08, 2026-12-06, 2027-01-03,
2027-01-31; respective 28-day targets end 2026-12-05, 2027-01-02,
2027-01-30, 2027-02-27. They preserve Sunday issues and 28-day spacing, using
O−2 data. They are a separate prospective schedule, not an extension claimed
to have been issued at the past 2026-11-01 origin. This phase does not acquire
future values, invent forecasts, schedule jobs or claim any completed new path.
The frozen snapshot cannot supply a complete post-October path; confirmation
is pending. Forecasts must be saved before their future targets become known.

Stop after these six candidates, the two baselines, fixed calibration and
diagnostics. No new model/feature/window/interval variant if 10% is not reached.
No TDS article draft or alteration to the original notebook.

## Reproduction and implementation lock

Use a new isolated Python 3.10 environment with requirements.txt, record the
resolved dependency lock and imported/distribution version equality. Library
defaults and random seeds are recorded. Native computation uses one thread.
If clean installation is externally blocked, record it and use the existing
environment only with an explicit pre-fit amendment, never for a better score.

Protocol freeze precedes even synthetic fitting. Then implement/test temporal
cutoffs, affine level treatment, matched ML rows/corrections, 29-date bridge,
metrics and interval quantiles on synthetic data. A separate implementation
lock, with source/config/input/environment hashes and passing preflight, must
exist before real candidate fitting. Fit caches are source/input/issue/spec
keyed and immutable. Correctness fixes require an explicit departure record
with outcome exposure, old/new hashes and rerun scope; no undocumented retries.

Technical references: statsmodels ETSModel documentation and the installed
statsmodels source; Forecasting: Principles and Practice, interval scoring.
https://www.statsmodels.org/v0.14.4/generated/statsmodels.tsa.exponential_smoothing.ets.ETSModel.html
https://otexts.com/fpp3/toolbox.html#evaluating-distributional-forecast-accuracy
