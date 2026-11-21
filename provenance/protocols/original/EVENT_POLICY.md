# Event-feature policy v1

This policy was written before fitting and without consulting event responses
or final-period pageview patterns. Event selection uses musical relevance,
coverage across eras, and source quality. It does not use the development flag
list or the largest pageview days. The 23 entries are a selective working ledger,
not an exhaustive release census or a denominator for release-rate claims.

## Availability and dating

- **A:** a dated contemporary source verifies the advertised event in advance,
  or the feature is a deterministic calendar date. Eligibility is conditional on
  the particular origin, not on the A label alone.
- **B:** an event/edition date is verified, but advance availability is not.
  Interpretation only. Current artist-listed release dates are not announcement
  timestamps or proof of the first appearance of every edition.
- **C:** unresolved date, edition, or event information. Research notes only;
  exclude from event-response calculations until resolved by sources.

For A announcements with a date but no defensible UTC timestamp, availability
begins at 00:00 UTC on the following day. At issue O, require that timestamp
to be no later than the issue. Calendar rules are explicitly marked and need
no invented announcement date. Current artist-site ordering, indexing dates,
fan comments, and Pike numbering do not establish historical availability.

The ledger separates the advertised schedule from actual occurrence. The four
historical tour rows verify that a future schedule was public, not that every
concert occurred. Announced-event features describe plans known at the origin;
they may not retrospectively remove cancellations. For retrospective discussion
of a performance itself, verify actual occurrence separately.

Use the release edition actually documented: digital, live-recording release,
preview, physical reissue, or streaming edition. A live-album release date is
not its recording date. `earliest_verified_public_announcement_date` means the
earliest source established in this selective research, not proof that nothing
was announced earlier. Conflicting dates remain visible.

## One small controlled experiment, conditional on feasibility

Proposed features are only:

1. Indicator that a verified, previously announced musical event is scheduled
   on the target date.
2. Indicator that such an event is scheduled in the preceding seven days of
   the target date, based solely on the schedule known at issue O.
3. One predeclared October 31 calendar indicator, if the musical-event gate
   passes; do not search arbitrary anniversary windows or tune event widths.

Scheduled events after the issue may populate these features only when their
announcement is already eligible. A B/C row never becomes a training covariate
merely because its event date is now known. Construct features separately at
each historical training issue S as well as evaluation issue O. Otherwise
current catalog knowledge would leak into historical supervised training.

Zero means **no eligible event is documented in this ledger as of this issue**.
It does not mean there was no release or performance. Research coverage is
incomplete and probably uneven; report that limitation. Do not estimate missing
release counts, use the ledger as a complete activity measure, or add final
events after viewing final errors.

Pre-model feasibility gate, decided using source dates and the schedule only:
at least six distinct advance-known musical episodes in the scored development
paths, with at least two in tuning and two in selection, spanning at least two
calendar quarters. Count one tour or coordinated release campaign as one
episode; recurring holidays do not count as musical episodes. Require at least
two documented musical episodes eligible at final issue dates for the intended
final comparison. This is a pragmatic scope gate, not a statistical power
guarantee. Even if it passes, report how few event-exposed paths exist.

Current ledger: four A musical episodes, all dated 2016–2019, and one A calendar
entry in 2024. There are **zero eligible A musical episodes in development**.
The gate fails. The frozen v1 plan therefore uses the ledger for interpretation
and does not force event-aware predictive testing. This is a limitation of this
bounded research ledger, not proof that usable contemporary announcements do
not exist. Any expanded announcement research requires a recorded pre-model
protocol amendment and another source-only gate assessment, without target-based
event selection. Do not move the evaluation window to make existing events fit.

If a pre-model amendment legitimately passes the gate: select family, window,
and hyperparameters from event-free development results; add the two fixed
musical-event features and fixed calendar indicator without further tuning.
Match training rows, origins, settings, clipping and fallback rules. Compare
paired path MAE and pooled metrics. Freeze that pair and two baselines before
final performance inspection. This estimates the addition of these documented
features to that configuration, not the value of all Buckethead expertise.
If a baseline wins selection, keep it as the selected forecasting procedure and
restrict events to interpretation; no extra family is added to manufacture a pair.

Retrospective explanations may be researched after the protocol is frozen, with
the frozen ledger retained and new interpretive sources clearly distinguished.
Coincident dates do not establish causality. An unexplained spike may remain
unexplained. Original fan-research insights should be checked against sources.
