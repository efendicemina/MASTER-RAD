# MYLYN transfer, temporal-drift, and threshold protocol v2

This development-only protocol is frozen before real transfer results. Target data are
only the 7,664 approved MYLYN development rows (SHA-256
`8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5`), using outer-fold
fingerprint `5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903`
and seed 42. The held-out test is rejected fail-closed and never used for schema, labels,
selection, calibration, thresholding, or metrics.

## Tasks and sources

S6 is blocker/critical/major/normal/minor/trivial. S3 maps blocker+critical to HIGH,
major+normal to MEDIUM, and minor+trivial to LOW. S2 maps blocker+critical+major to
HIGH_IMPACT and normal+minor+trivial to LOWER_IMPACT. Enhancement, task, feature, blank,
missing, unknown and unmapped labels are excluded. S6/S3/S2 remain separate claims.

The exact transfer sources are BIRT, CDT, Equinox, JDT, Papyrus, PDE, Platform and TPTP.
MYLYN is prohibited as a source. Dry-run records every Parquet SHA-256; real/resume must
match. Schema is read via `ParquetFile.schema_arrow.names`; only project, internal issue
key, creation_time, severity, summary and description are batch-read.

For every inner/outer validation start, source rows must satisfy strict
`creation_time < validation_start`. Exact and normalized text identities are label-free.
Overlapping rows are removed only from target/source training; validation is unchanged.
Cutoff violations and post-purge overlap must both be zero per project/fold.

## Representations and grid

R0 is approved P0 Summary+Description word TF-IDF `(1,2)`, min_df 2, max_df .98,
max_features 50,000, sublinear TF, L2. R1 has separate Summary word max 20,000 and
Description word max 40,000 branches with Summary weights 1, 2, 3. R2 adds cleaned
concatenated `char_wb` 3–5, min_df 3, max_df .995, max 30,000, also at weights 1, 2, 3.

Cleaning performs deterministic Unicode normalization, HTML entity decode/tag removal,
whitespace normalization and neutral URL/email/Bugzilla-ID/long-hex placeholders while
preserving negation, code, punctuation and technical terms. A compound token emits its
normalized original plus camelCase, snake_case and dotted parts; there is no arbitrary
token truncation.

LinearSVC uses C 0.1/0.5/1.0 × weight none/balanced. LogisticRegression uses C
0.25/1/4 × none/balanced, lbfgs, max_iter 2000. ComplementNB uses alpha 0.1/0.5/1 only
on non-negative matrices. S2 additionally permits NB-SVM smoothing 1 with LR C
0.25/1/4 × none/balanced. Every concrete task/representation/Summary-weight/model
combination is a candidate row; stage options are expanded, never stored as inert lists.

## Nested Stages A–E

Each outer target-training partition creates two expanding-window inner splits when class
support permits, otherwise one documented temporal holdout. Outer validation is evaluated
once after all choices.

Stage A uses target-only inner OOF macro-F1 (S2 uses calibrated/thresholded inner OOF) and
keeps two candidates per task. Ties: minimum recall, relevant minority recall, lower fold
SD, R0<R1<R2, fewer features, runtime, candidate ID.

Stage B compares target-only with source weights .10/.25/.50; target weight is 1. Combined
source and recency weights are normalized to mean one. Stage C tests all-history,
five-year, and exponential half-lives 2/4/6 years relative to validation start. Stage D
tests plain pooling and sparse `[shared,target-specific,source-specific]` augmentation.
Stage E tests score ensembles .25/.75, .50/.50, .75/.25 only for compatible top models and
accepts one only for inner OOF gain >=.01. Outer results decide nothing.

S2 sigmoid calibration is fitted only on inner OOF scores. Threshold grid is .05–.95 by
.005, maximizing HIGH_IMPACT F2 subject to precision >=.30, then recall, precision,
macro-F1, distance to .50, higher threshold, stable numeric ordering. The frozen
calibrator/threshold is applied once to outer scores.

## Metrics, controls, and success

Metrics include macro precision/recall/F1, balanced accuracy, weighted F1, accuracy,
fixed-label per-class metrics/support, minimum recall, distributions/dominant share,
raw/normalized matrices, runtime, dimensions, serialized size and RSS. S2 adds high-impact
precision/recall/F1/F2, lower-impact recall, specificity, NPV, positive rate, PR-AUC and
secondary ROC-AUC.

Matched source-free comparators share outer rows and selected representation/model.
Paired bootstrap uses 1,000 identical row resamples, seed 42, and reports delta percentile
95% CI; misalignment fails. Clear transfer evidence requires positive delta, CI lower >0,
gains in two folds, and no integrity warning; otherwise `NO_CLEAR_TRANSFER_EVIDENCE`.

S6 material: macro-F1 >=.2463, gains in two folds, min-recall loss <=.05, nonzero blocker
and critical recall, all labels predicted, dominant <85%. Strong additionally requires
macro-F1/balanced accuracy >=.35 and min/blocker/critical recall >=.15. S3 material:
baseline+.05, two folds, HIGH recall gain >=.10, min recall >=.20, all labels, dominant
<85%; strong requires macro-F1 >=.50, HIGH >=.35, min >=.30, balanced >=.50, SD <=.08.
S2 material: baseline+.05, two folds, balanced >=.63, high-impact precision/recall >=.45,
improved F1, positive rate 2–60%; strong requires macro-F1 >=.70, balanced >=.65, recall
>=.60, precision >=.50, F1 >=.55 and SD <=.08. Accuracy alone never establishes success.

Controls are Dummy, one deterministic shuffled-training-label outer-fold-1 run,
label-token masking, matched source-free comparison, cutoff/duplicate/calibration/threshold
proofs, distributions, and non-selectable coverage 100/95/90/80/70 using confidence fixed
without outer-label adaptation.

## Resources, manifests, and publication

Sparse float32, n_jobs 1, batch reads and one task/fold/stage/representation in memory are
required. Real run requires >=4 GiB currently available RAM, passing dry-run estimate and
disk. Below that, code/tests/synthetic/dry-run and protocol push finish, then status is
`RUN_BLOCKED_LOW_MEMORY`; no metrics/results commit is invented.

Atomic files are temporary siblings, flushed/closed then `os.replace`. Existing output
fails unless `--resume`. Resume requires identical development/fold/inner/source hashes,
protocol content hash, candidate checksum, seed, mappings, preprocessing, grid, success
criteria, CLI, packages, schema version and completed stages. Any mismatch fails closed.
Dry-run fits no model but performs real target/source/hash/fold/cutoff/duplicate/grid/resource
audits. Only sanitized dry-run artifacts are committed. A later real run uses a new output,
the pushed protocol commit, atomic checkpoints, and never overwrites or accesses test data.
