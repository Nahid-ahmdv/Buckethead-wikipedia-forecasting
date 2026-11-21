# Delivery scope and original research archive

This repository is a separate GitHub delivery prepared from the completed
Buckethead research project. The original working directory was backed up
separately before its removal.
No experiment was added, retuned or substituted during packaging.

Included here are frozen data and request metadata, the event ledger, scientific
result tables, every origin cache used for verification, reader-linked figures,
selected research reports, source registers, original model implementations,
contracts, independent verifiers and historical manifests. The notebook is the
single reader entry point.

Private Towards Data Science article files are excluded from Git and the
required public inventory. Shared notebook figures have tracked copies under
`figures/retrospective_followup/`; article text and article-only exports remain
local. Historical publication-only manifests are also excluded. The original
GitHub delivery inventory is preserved as a dated record, not a claim that
private article files are included in a public checkout.

The full working archive also contains superseded presentation builders,
editorial reviews, repeated rendered previews, screenshots, old notebook
backups and execution logs. Those are generally omitted from this delivery.
Virtual environments, bytecode and notebook checkpoints are excluded.

## Integrity boundary

[source_map.json](../provenance/source_map.json) records the original relative
path and SHA-256 of every selected source artifact, its new path and delivery
hash, and any transformation. The mathematical model implementations, input
data, predictions, metrics, caches and original contracts retain their bytes.
Linked report copies update relative Markdown targets where necessary.

[manifest.json](../provenance/manifest.json) protects the current delivery.
[Historical manifests](../provenance/historical/) retain their original bytes
and dates. Their inventories refer to the original layout and can include files
outside this selected package. The notebook reports partial retained coverage
explicitly, with omitted files counted as **not verified**. No historical
manifest has been regenerated to fit this layout.

The original notebook is stored as compressed JSON in
`provenance/original_notebook.json.gz` to make the packaging diff inspectable.
It is an audit record, not a second runnable notebook. The
[packaging record](../provenance/packaging.json) distinguishes notebook path
changes, link updates, reproduction notes and refreshed delivery-check outputs.
The nine scientific figure payloads retain their original bytes.

## Reproduction boundary

The documented workflows work using this repository alone. The adapter rebuilds
the original computational paths in a temporary directory from byte-preserved
selected files, and replaces broad historical presentation guards with the new
delivery guard. Source/configuration locks inside the model runners remain
active. Dated historical checks and new packaging evidence have separate records.

To publish the full archive later, make it a separately versioned research
archive rather than adding every editorial intermediate to the main reader
repository. No public archive URL has been assigned or invented here.
