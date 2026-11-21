# Pre-fit amendment 01 — selection/calibration availability

The initial preflight caught a one-day information conflict: the eighth
selection path ended 2025-06-14, but the first calibration issue 2025-06-15
can use observations only through 2025-06-13. Point selection using that path
could not have been completed before calibration forecasting began.

No real-series Phase 4 model had been fitted and no real candidate score was
computed. The synthetic engine tests passed; one chronology test failed. The
original protocol freeze, schedule, failed test output and implementation
sources are preserved. The test assertion is also clarified to allow equality
with a cutoff: the proper requirement is last available target <= cutoff.

Correction: use seven selection paths through 2025-05-17; treat the May 18–
June 14 path as an unscored boundary observation block. Keep all four
calibration paths and all thirteen retrospective paths unchanged. There are
now 36 scored origins, 288 procedure-origin records, 8,064 daily predictions,
216 real candidate fit attempts and 72 baseline paths. Exactly six candidates
and the two baselines remain unchanged. This is a chronology correction,
not a performance-driven revision.

The active protocol/schedule are PROTOCOL_V2.md and schedule_v2.csv. This
amendment is authorized by the user's instruction to resolve and report
necessary correctness departures before fitting. The active implementation
is separately locked only after a corrected clean-environment preflight.
