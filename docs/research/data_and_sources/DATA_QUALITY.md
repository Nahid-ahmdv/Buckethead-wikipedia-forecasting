# Phase 0 data-quality report

Snapshot acquired November 5, 2026. The cutoff, October 31, is a calendar rule,
not a date chosen to avoid an inconvenient outcome. Acquisition metadata are in
`data/raw/acquisition_manifest.json`; source response bytes are preserved.

| Mechanical check | Result |
|---|---|
| Requested calendar | 2015-07-01–2026-10-31, 4,141 days |
| Canonical series observed | 4,141 dates |
| Missing dates / duplicates | 0 / 0 |
| Negative, noninteger, null counts | 0 |
| Zero-count days | 0 |
| Access-specific date coverage | Each of desktop, mobile-web, mobile-app has 4,141 dates |
| Desktop + mobile-web + mobile-app = all-access | Exact equality on every date |
| Target identity | English article `Buckethead`, current page ID 21244303 |
| Current redirect aliases | 19; not included in target |

The full machine-readable audit is `mechanical_audit.json`. Validation checks
every raw item's project, title, agent, access, granularity, timestamp, and count
type. Canonical counts are retained unchanged. No correction, interpolation,
zero deletion, or outlier exclusion has been applied. Integrity is not evidence
that all counts represent human readers.

The legacy-format export was also loaded with pandas in the existing local
environment: all seven named columns aligned correctly, the exported index was
retained, all ranks were missing, and dates/counts matched the canonical table
after accounting for the index. No package installation was needed.

## Redirect definition

Wikimedia distinguishes visits to a redirect from visits to its destination;
the canonical-title series does not automatically include every redirect.
[Official definition](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/concepts/page-views.html).

The current alias list includes biographical spellings but also albums and
related bands. Adding all current redirects would silently broaden the target
and apply today's article relationships retrospectively. Primary modeling uses
the canonical title alone throughout.

Two recognizable biographical aliases were retrieved for scope/coverage checks:

| Current redirect | Returned daily records | Absent dates |
|---|---:|---:|
| Brian Patrick Carroll | 2,891 | 1,250 |
| Brian Carroll (musician) | 864 | 3,277 |

Absence is not assigned zero and alias counts are not merged. Historic creation,
move, merge, and redirect membership were not reconstructed; a present redirect
list cannot provide them. Coverage is documented without interpreting alias
spikes or comparing final-period traffic levels.

## Development-only unusual-count screening

A descriptive upper-tail screen compared each count with the preceding 28-day
median and MAD: flag if y > median + max(8*1.4826*MAD, 2*max(median,1)). It found
21 flagged dates before 2025-11-02, saved in `development_integrity_flags.csv`.
The rule was not fitted to final outcomes. No final spike screen, extrema,
trend chart, quantile table, or outcome averages were computed. Flags do not
identify bots or establish an event response. All counts remain included. The
event ledger was selected from musical/source considerations, not these flags.

## Outcome exposure and protection

Raw final observations were downloaded, parsed in memory, validated, reconciled
across access methods, hashed, and written to the clean CSV. The analyst saw
date coverage, absence/invalid/zero counts, identity metadata, and equality-check
results. Redirect missing-date lists were also mechanically produced/displayed.
This is limited outcome-related exposure, not a claim of pristine blindness.

No final count values, peaks, means, trend summaries, musical-event responses,
forecasts, or forecasting errors were examined. The existing Machine Learning
assignment results concern another article and are not Buckethead outcomes.
Current Buckethead event pages, including a 2026 collaboration's release date,
were read independently of pageview outcomes and disclosed in the ledger.

Use `development_daily.csv` for subsequent EDA; it excludes every final row.
Per-origin publication cutoffs still apply inside that file. Protect final
outcomes procedurally: the full dataset is local and readable, not encrypted
or inaccessible. Record every later opening and any protocol amendment.

## What passed and what remains uncertain

Calendar coverage and mechanical consistency support the proposed experiment.
They cannot establish exact human readership, original historical availability,
the absence of automated traffic, or a complete measure of artist attention.
The incident policy is defined separately before fitting. Further source-quality
questions may justify a recorded amendment, never outcome-dependent deletions.
