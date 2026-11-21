# GitHub packaging verification

This directory contains dated delivery checks. Each JSON report states its
scope. The current [single-requirements check](single_requirements.json)
records consolidation of the dependency files with all 64 version pins unchanged.
The preceding [review-correction check](review_corrections.json) binds its
notebook and delivery manifest hashes, reports the failure-policy tests, and
verifies that the scientific records were unchanged in that release. The preceding
[public-only release check](public_only_release.json) remains dated evidence
for the earlier notebook and manifest. The earlier [delivery-v1 release check](release.json) records
packaging before the article was excluded from Git. Its historical archive
comparison remains dated evidence; current execution requires no original archive.

The delivery manifest excludes this directory to avoid a circular hash dependency
between execution evidence and the notebook that displays the manifest check.
Reports are evidence from their recorded run, not inputs for calculating forecasts.

The source archive's original numerical check remains separately dated in
[numerical_verification_v1.json](../historical/numerical_verification_v1.json).
The packaging numerical reconstruction does not fit optimizers, and cache replay
does not retrain models. The full optimizer pipeline was not rerun during this
packaging task.

The original archive's research and current-publication integrity inventories
pass (934 and 141 files). Its older writing-preservation check reports a
pre-existing `.DS_Store` mismatch. The task-start and task-end hashes of that
Finder metadata file agree; it was left untouched. This archived presentation
issue is recorded separately from the passing GitHub-delivery checks.
