# Frozen foundation-model extension

This is a new retrospective comparison of two pretrained forecasting recipes:
`amazon/chronos-2` and `google/timesfm-2.5-200m-pytorch`. The model choices and
settings were recorded before new target inference. The original final year
had already been examined, so the comparison is not independent confirmation.

The [machine-readable protocol](../provenance/foundation_models/protocol.json)
records exact checkpoint revisions, expected weight hashes, input hashes,
context length, prediction extraction and scoring rules. Its
[implementation lock](../provenance/foundation_models/protocol_lock.json)
protects the runner, analysis code and input records. The original experiment
and Phase 4 forecasts, model settings and result tables retain their bytes.

## What the new recipes receive

Each model sees one sequence of **1,024 raw daily counts**, ending at `O - 2`.
It receives no article title, event metadata, related pages, future observations
or locally fitted correction. Each issue predicts 29 dates; `O - 1` is retained
in the audit and discarded from scoring. The remaining 28 dates match the
original task. Context length was fixed, with no scored search or fine-tuning.

Both point forecasts use the native 0.5 quantile. RMSE therefore describes the
same median forecast used for MAE, rather than a separately selected mean.
The native 0.1 and 0.9 quantiles define the nominal 80% pointwise band.
The shared postprocessing sorts those three quantiles within a date and clips
at zero; raw outputs and repair counts remain available. All operations use
CPU float32, one thread, seed 42, and disabled torch compilation.

The 37 scored issues retain the original 12 tuning, 12 selection and 13 final
paths, excluding the two incomplete boundary blocks. Those legacy stage names
describe target populations; they do not mean that these models were tuned or
selected on those scores. Earlier-period model availability is not claimed.

## Pretraining exposure

The [TimesFM model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
documents Wikimedia Pageviews through November 2023, alongside other real and
synthetic sources. The first original tuning path overlaps that cutoff month.
Exact Buckethead membership and the date coverage of every corpus component
have not been verified.

The [Chronos-2 report](https://arxiv.org/html/2510.15821v1) lists Wiki among its
real pretraining datasets. Exact Buckethead membership and complete training
date coverage have not been recovered. The
[official Chronos model card](https://huggingface.co/amazon/chronos-2) describes
the pretrained inference interface. Copies of both cards, configurations and
repository metadata are retained under `provenance/foundation_models/`.

Exposure to older pageviews is different from training on an evaluated target.
This audit does not certify an absence of evaluated-target overlap. Here,
“zero-shot” means no task-specific weight fitting in this project. It does not
establish a previously unseen article, domain or evaluation sample.

## Read, verify, or repeat

Use the root `requirements.txt`, which includes the existing reader/model
packages and the pinned foundation-model dependencies. There is no second
root requirements or lock file. Model weights remain outside Git; downloading
them requires network access. Reading and verifying saved results does not.

```sh
python scripts/foundation_models.py --mode plan
python scripts/foundation_models.py --mode verify
python -m unittest discover -s tests -v
```

To repeat inference, use a new output directory, and invoke one model at a
time to limit memory. Without `--weights-root`, the runner retrieves the exact
checkpoint revisions into the Hugging Face cache and verifies their hashes:

```sh
python scripts/foundation_models.py --mode infer --model chronos2 --output ../buckethead-foundation-replay
python scripts/foundation_models.py --mode infer --model timesfm25 --output ../buckethead-foundation-replay
python scripts/foundation_models.py --mode summarize --output ../buckethead-foundation-replay
```

Cached origins are reused only when their protocol and context hashes match.
Unexpected errors or nonfinite outputs stop the run and produce a failure
record. No fallback, tuning, optimizer training or automatic model substitution
is performed. The final notebook reads the recorded forecasts and figures;
its normal execution does not call either inference library.

## Reading the comparison

The primary table pools 364 final-year targets for each recipe. The November
mask re-scores identical predictions on 335 dates; it does not remove November
observations from later contexts. MASE retains the original common three-year
weekly-difference denominator. Paired path losses use matching issues, with
the partial retained path weighted by its days in pooled masked scores.

Interval comparisons include the original SARIMAX's frozen residual band and
Phase 4's separately calibrated weekday-mean band. Those are different whole
procedures from the native foundation-model quantiles. Correction LightGBM has
no evaluated intervals and is excluded from interval comparisons.

Per-origin caches retain raw 29-by-3 quantile arrays, context hashes, timing,
warnings and processing repairs. Execution records include package versions,
platform, peak process memory and exact checkpoint revision. Runtime is an
observation on the recorded machine, not a hardware-normalized speed claim.
Figures show all path differences and the first three retrospective paths in
chronological order. No new future observations or forecasts are part of this
extension; selecting a winner on this inspected year would require a later
independent confirmation.
