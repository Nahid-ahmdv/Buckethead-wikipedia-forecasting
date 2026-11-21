# Dataset definition and files

The target is the API's daily `views` count for the literal article title
`Buckethead`, project parameter `en.wikipedia.org` (response `en.wikipedia`),
access `all-access`, agent `user`. A day is a UTC day. Current page identity is
page ID **21244303**, canonical URL <https://en.wikipedia.org/wiki/Buckethead>.
The Action API response records the current revision and nineteen redirects;
this does not establish historical redirect membership or a stable page ID for
every historical title. The pageview API is title-based.

`data/processed/buckethead_daily.csv` is the authoritative working table:

| Column | Definition |
|---|---|
| `date` | ISO date, one complete UTC day |
| `count` | Nonnegative integer user-classified pageviews; missing source observations would remain null |
| `lang` | `en` |
| `page`, `title` | `Buckethead` |
| `rank` | Empty; not provided by this API and unused in the models |
| `month` | `YYYYMM` derived from `date` |
| `partition` | `historical_training`, `development`, or `final`; subdivision appears in the schedule |
| `source_missing` | 1 if absent from source; every current observation is 0 |
| `measurement_flag` | Global incident/revision context; never proof of article-specific contamination |

`development_daily.csv` contains all pre-final observations, including the
earlier training history. It ends on 2025-11-01. Per-origin information cutoffs
are stricter than file membership; see the protocol's publication-delay rule.

`wiki_buckethead.csv` reproduces the existing assignment file's unusual
R-export-style structure: seven named space-separated columns, plus an extra
quoted unnamed row index in each data row. Strings are quoted, dates and counts
are unquoted, and the missing rank is `""`. Load it with:

```python
df = pd.read_csv("data/processed/wiki_buckethead.csv", sep=" ")
```

The index is a new sequential export index, not a Wikipedia page identifier.
No Machine Learning ranks or row IDs were copied. Canonical rows retain the full
daily calendar. No zero filtering, interpolation, spike removal, smoothing, or
bot correction has been applied.

Raw JSON and request metadata live in `data/raw/`. Metadata record response
bytes, SHA-256, URL, request identity, and retrieval time. The first response's
completion time comes from its saved file modification time after successful
curl execution; its HTTP headers are separately preserved. Later responses
record explicit request start/completion timestamps and response headers.

Two current biographical redirect titles were retrieved diagnostically. Their
absent observations remain absent; they are neither filled with zero nor added
to the target. This project measures visits to the canonical title, not the sum
of every present or past alias, album, band, and associated Wikipedia page.
