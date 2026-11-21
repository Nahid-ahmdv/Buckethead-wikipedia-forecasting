# From Pikes to Pageviews: Forecasting Buckethead's Wikipedia Traffic

![Buckethead performing on stage with a white guitar, mask, and bucket hat](figures/Buckethead.jpg)

In this project, I forecast the next **28 days of daily views to Buckethead's
English Wikipedia page** and investigate how much different models add to a
simple average of matching weekdays. The original development comparison
selected SARIMAX, but on the following year it persistently overpredicted
traffic. That failure led me to investigate changing traffic levels, test
LightGBM corrections around the baseline, and later compare TimesFM 2.5 and
Chronos-2.

- **Full analysis:** [Executed forecasting notebook](buckethead_wikipedia_forecasting.ipynb)
- **Towards Data Science article:** Coming soon.

Saved notebook outputs can be read on GitHub without running the analysis.

## The forecasting task

The frozen snapshot contains **4,141 consecutive daily observations**, from
July 1, 2015, through October 31, 2026. The target is user-classified pageviews
of the canonical English article, across desktop, mobile web, and mobile app.
These are recorded page requests, not unique readers, fan counts, or a measure
of overall popularity.

Each forecast is treated as issued at midnight UTC, with observations available
only through **two days before the issue date**. All 28 target-day predictions
are made at once. Methods forecasting forward from the last available count
produce 29 steps: the first bridges the unobserved intervening day and is
excluded from scoring. Forecast issues are 28 days apart.

The original experiment used **12 tuning paths**, followed by **12 later
model-selection paths**. The selected configuration and two seasonal baselines
were then evaluated on **13 paths covering November 2, 2025, through
October 31, 2026: 364 target days**. Models were refitted at each origin using
eligible earlier observations, while selected settings stayed fixed.

The eight-week weekday average was selected from four- and eight-week windows
during tuning. Musical events were investigated for historical context, but
none entered the forecasting features because there were too few eligible
events with sufficiently established advance information.

## Main results

The following procedures were evaluated on the same 364 later-year targets.
The LightGBM correction and foundation-model experiments took place after that
year had already been examined.

| Procedure | Evaluation role | MAE, views/day |
|:---|:---|---:|
| TimesFM 2.5 (200M) | Retrospective foundation-model extension | **199.399** |
| Chronos-2 | Retrospective foundation-model extension | 211.380 |
| LightGBM correction | Bounded retrospective follow-up | 236.533 |
| Eight-week weekday average | Original baseline, retained in later comparisons | 238.733 |
| Selected SARIMAX | Original final evaluation | 437.776 |

TimesFM and Chronos-2 reduced observed MAE relative to the weekday baseline by
**16.48%** and **11.46%**, respectively. Both used a fixed **1,024-day context**
with no task-specific fine-tuning. Their advantages remained under the existing
November 2025 scoring sensitivity.

LightGBM's correction improved retrospective MAE by only **0.92%**, falling to
approximately **0.06%** when November 2025 was excluded from scoring. It had
worse RMSE than the weekday baseline and no evaluated prediction intervals.
The weekday average remained the procedure selected by that follow-up's
earlier selection stage.

### Prediction intervals

Native nominal **80% pointwise intervals** covered **75.55%** of daily outcomes
for TimesFM and **70.33%** for Chronos-2. Original SARIMAX residual intervals
covered **44.78%**; the follow-up weekday baseline's separately calibrated
intervals covered **95.88%**, with greater mean width.

These compare complete forecasting procedures with different point models and
interval constructions. Pointwise coverage does not promise containment of all
28 outcomes on a forecast path.

### What the evidence supports

The later experiments are retrospective comparisons on an already inspected
year. Exact Buckethead-series membership and evaluated-target overlap in
foundation-model pretraining remain unverified. No task-specific fine-tuning
does not establish that the series was unseen during pretraining.

The data were retrieved afterward, so the historical two-day cutoff is a
simulation of availability using the preserved snapshot. The project records
**zero independently evaluated new future paths**. The saved results nominate
procedures for further testing rather than establish independent confirmation
of future performance.

## Run the notebook

The recorded environment is **Python 3.10.18 on macOS arm64**. The single root
[requirements.txt](requirements.txt) contains the pinned reader, modeling, and
foundation-model dependencies. Its combined 90-package environment was
previously installed and dependency-checked in an isolated environment on that
host; this does not establish cross-platform numerical equivalence.

From the repository root on macOS or Linux:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/execute.py
```

Normal notebook execution reads the included data and saved predictions,
verifies the reported numbers, and displays the figures. It runs all **30 code
cells** in memory without overwriting the notebook. It requires no network
requests after installation and performs no model fitting, tuning, or
foundation-model inference.

For interactive use, select the environment as the notebook kernel in your
editor. Keep the supporting folders beside the notebook; they are execution
inputs even though the outputs are already saved.

Optional numerical reconstruction and fixed-recipe refitting are documented
in the [reproduction guide](docs/reproducibility.md). The
[foundation-model guide](docs/foundation_models.md) covers checkpoint revisions,
pretraining limitations, and optional inference. Model weights are downloaded
separately and are excluded from Git.

## Repository structure

```text
buckethead-wikipedia-forecasting/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── buckethead_wikipedia_forecasting.ipynb
├── data/
│   ├── README.md
│   ├── raw/                       # Frozen API responses and request metadata
│   ├── processed/                 # Daily datasets used in the analysis
│   └── research/                  # Event ledger
├── src/                           # Analysis, verification, and model implementations
├── scripts/                       # Execution and reproduction commands
├── tests/                         # Integrity and forecast-timing checks
├── results/
│   ├── development/
│   ├── original_evaluation/
│   ├── diagnostics/
│   ├── historical_context/
│   ├── retrospective_followup/
│   └── foundation_models/
├── figures/                       # Figures used by the notebook and research reports
├── docs/                          # Methods, sources, and reproduction instructions
└── provenance/                    # Protocols, configurations, and integrity records
```

Saved forecast caches inside `results/` are scientific records used by the
verification workflow and belong in Git. The data, source code, figures,
protocols, and integrity manifests are also needed for the documented
workflows. The original working archive and the author's other projects are
not required.

[.gitignore](.gitignore) excludes virtual environments, downloaded model
weights, bytecode, notebook checkpoints, and build artifacts. It also excludes
the private `publication/` folder, article-only supporting files, and the
explicitly named notebook backup. Shared research figures remain included.
The public notebook runs without the article drafts or publication exports.

## Methods and reproducibility records

- [Methodology and evaluation chronology](docs/methodology.md)
- [Source guide](docs/sources.md)
- [Data definitions](docs/data_dictionary.md) and [data-quality assessment](docs/research/data_and_sources/DATA_QUALITY.md)
- [Reproduction instructions](docs/reproducibility.md)
- [Foundation-model extension](docs/foundation_models.md)
- [Archive scope and packaging](docs/research_archive.md)
- [Delivery integrity manifest](provenance/manifest.json) and [verification evidence](provenance/checks/README.md)

## Data Source and License

The daily Wikipedia pageview data used in this project were obtained from the
Wikimedia Analytics API and are made available under the **Creative Commons
CC0 1.0 Universal Public Domain Dedication**.

**Source:** Wikimedia Foundation — [Wikimedia Analytics API](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html)

**Article:** [Buckethead (English Wikipedia)](https://en.wikipedia.org/wiki/Buckethead)

**Period:** July 1, 2015–October 31, 2026

**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

The API data may be reused, modified, and redistributed, including for commercial
purposes. This project is an independent research study and is not affiliated
with or endorsed by the Wikimedia Foundation.

**Official Wikimedia licensing documentation:** [Analytics API access policy](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/documentation/access-policy.html).

CC0 applies to the Wikimedia Analytics API data. It does not automatically
license project code, writing, charts, third-party photographs, album covers,
music, or external research materials. No separate code or writing license has
been selected in this delivery; see the [license notice](LICENSE).

## Author

Nahid Ahmadvand

[LinkedIn](https://www.linkedin.com/in/nahid-ahmadvand-00382b365/) | [Website](https://www.nahidahmadvand.website/)

Questions, corrections, and ideas for extending the analysis are welcome.
