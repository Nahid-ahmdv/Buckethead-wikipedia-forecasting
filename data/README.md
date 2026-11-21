# Frozen data snapshot

Source: Wikimedia Foundation, [Wikimedia Analytics API](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html).
Article: **Buckethead**, English Wikipedia. Period: **July 1, 2015–October 31, 2026**,
4,141 complete UTC dates. Request parameters: `en.wikipedia.org`, `all-access`,
`user`, `daily`.

| Path | Contents |
|---|---|
| `raw/` | Preserved API response bodies, request metadata, acquisition manifest and initial HTTP headers |
| `processed/buckethead_daily.csv` | Canonical daily counts, metadata, partitions and measurement flag |
| `processed/development_daily.csv` | Development subset excluding the original final year |
| `processed/wiki_buckethead.csv` | Compatibility export with the original assignment-style schema |
| `research/event_ledger.csv` | Frozen, selective historical-event ledger with source and availability assessments |

The [data dictionary](../docs/data_dictionary.md) defines columns. Canonical-title
traffic excludes alternate-title requests; current redirect diagnostics cannot
reconstruct historical redirects. Pageviews are requests, not unique people.
The stored all-access counts reconcile with the three access-method responses.
All canonical observations remain in the target.

The November 2025 sensitivity excludes those target dates from an alternative
score population. It neither replaces observations nor removes November history
from later model training. Retrieved histories can be revised by Wikimedia;
fetching again is a new data vintage and need not reproduce this snapshot.

## License

The Wikimedia Analytics API data are **[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)**,
permitting reuse, modification and redistribution, including commercially.
[Official Wikimedia policy](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/documentation/access-policy.html).
The research is independent and is not affiliated with or endorsed by Wikimedia.
The source ledger and third-party references have their own applicable terms;
CC0 for the API data does not automatically license project code or creative
materials. The [main license statement](../README.md#data-source-and-license)
explains the scope.
