# Methodology and research chronology

Forecast **28 daily user-classified pageview counts** for one canonical English
Wikipedia article. At issue O, use observations only through O−2. Statistical
models predict a 29-day bridge and discard the unavailable intervening day.
Feature lags and rolling summaries use the issue cutoff; each historical ML
training path must have all 28 labels available by that cutoff.

| Research role | Preserved results | Interpretation |
|---|---|---|
| Original development | [Development results](../results/development/) | Twelve tuning paths followed by twelve procedure-selection paths |
| Original final evaluation | [Original evaluation](../results/original_evaluation/) | Thirteen paths, November 2, 2025–October 31, 2026; fixed selected procedure and interval width |
| Failure diagnostics | [Diagnostics](../results/diagnostics/) | Post-evaluation study of the original predictions and historical context |
| Bounded follow-up | [Retrospective follow-up](../results/retrospective_followup/) | Six candidates plus two baselines; seven selection paths, four calibration paths; later comparison reuses the examined final year |

The weekday reference averages the last eight available values for each weekday.
The original winner is SARIMAX (1,0,1)(0,1,1)[7] with annual Fourier K=2,
three-year history, raw counts and maxiter=200. Its seasonal MA estimate near −1
motivates investigation; convergence and root distance do not establish useful
forecasts or identify why it overpredicts.

The follow-up compares ETS level and damped-trend arms, a matched standardized
SARIMAX D=0/D=1 pair, and matched raw-target/correction LightGBM procedures. The
correction label subtracts each historical origin's own weekday baseline. Model
settings, comparisons and stopping rules remain the recorded ones.

MAE selects procedures. RMSE highlights larger misses, and MASE uses a common
training-only weekly scale. Daily pooled errors determine aggregate metrics;
paired paths share history and are dependent. Prediction intervals are judged
by coverage, width and interval score together. The original interval reuses
selection errors; the follow-up separates selection and calibration but remains
post-evaluation research. Neither establishes reliable nominal coverage or a
distribution-free guarantee.

## Protocols and implementation

- [Original experiment](../provenance/protocols/original/EXPERIMENT.md), [split schedule](../provenance/protocols/original/split_schedule.csv), [uncertainty rule](../provenance/protocols/original/UNCERTAINTY.md), [event eligibility](../provenance/protocols/original/EVENT_POLICY.md)
- [Development implementation](../provenance/protocols/development/PROTOCOL_IMPLEMENTATION.md) and [original final specification](../provenance/protocols/development/FINAL_EVALUATION_SPEC.md)
- [Original execution contract](../provenance/protocols/original_evaluation/execution_contract.json)
- [Follow-up active protocol](../provenance/protocols/retrospective_followup/PROTOCOL_V2.md), [chronology amendment](../provenance/protocols/retrospective_followup/AMENDMENT_01.md), [implementation lock](../provenance/protocols/retrospective_followup/implementation_lock.json)
- [Failure analysis](research/diagnostics/FORECAST_FAILURE_ANALYSIS.md) and [follow-up findings](research/retrospective_followup/PHASE4_FINDINGS.md)
- [Scientific verification implementation](../src/verify_saved.py), [numerical replay](../scripts/replay_numerics.py), [reproduction guide](reproducibility.md)

No musical event features were used: verified advance-known musical episodes
were insufficient in the scored development period. Historical interview and
release windows are descriptive, selected after observing history, and do not
establish causal effects. No genuinely unseen future forecasts have been
evaluated or issued in this research snapshot.
