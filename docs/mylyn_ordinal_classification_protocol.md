# Frozen MYLYN ordinal severity-classification protocol

This protocol is frozen before any real ordinal result is observed. The study uses only
the immutable MYLYN `development_split.csv` (SHA-256
`8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5`), Summary,
Description, Severity, and technical identifiers required for the approved folds and
duplicate protection. The held-out test and all other Eclipse projects are prohibited.

## Fixed task, order, and data policy

The final task remains exact six-class classification. The immutable increasing severity
order is `trivial=0`, `minor=1`, `normal=2`, `major=3`, `critical=4`, `blocker=5`.
Bugzilla severities express increasing seriousness; neighboring judgments are closer, so
blocker-to-critical is less severe than blocker-to-trivial. Standard multiclass loss does
not represent that distance. Labels are never merged and regression output never replaces
the six-class prediction.

Only the six fixed labels are accepted; enhancement, missing, blank, unknown, and unmapped
labels are excluded. Text and missing values follow the approved deterministic
Summary-separator-Description preprocessing: lowercase, HTML removal, URL/email
replacement, Unicode and whitespace normalization, while code blocks and stack traces are
retained. No Priority, Status, Resolution, Product, Component, Version, platform, OS,
comments, history, attachments, people/contact fields, post-report, target-derived, or
other structured metadata is predictive.

The approved duplicate policy retains the earliest valid exact-text row, removes later
duplicates, prevents Dupe-of groups crossing folds, preserves conflict handling, and never
silently drops rows. The main immutable split is already policy-filtered, so this study
performs no additional main-data removals. A diagnostic only sensitivity run excludes the
five retained known conflict rows (`108445`, `166615`, `265078`, `373112`, `377134`); it
cannot select a model.

## Frozen representation and folds

Every method uses word TF-IDF `(1,2)`, `min_df=2`, `max_df=0.98`,
`max_features=50000`, `sublinear_tf=True`, fitted only on each training fold. The flat
model is LogisticRegression `C=1.0`, balanced weights, seed 42, `max_iter=2000`, default
`lbfgs`. Its expected macro-F1 is `0.2262765`; absolute reproduction tolerance is
`0.0001`, with failure blocking candidate creation.

The exact three approved expanding-window folds are reused: 1916/1871, 3832/1855, and
5748/1889 train/validation rows. Frozen fold fingerprint is
`5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903`.
Dataset, fold, train/validation row, and preprocessing hashes are persisted. Row membership,
temporal boundaries, mapping, preprocessing, duplicate protection, and ordering cannot
change.

## Fixed ordinal candidates and decoding

Approach A fits one training-only TF-IDF vectorizer and five independent balanced binary
LogisticRegression models (`C=1.0`, seed 42, `max_iter=2000`, `lbfgs`) for `y>0` through
`y>4`. Binary weights and supports come only from training labels; any single-class
threshold fails. Thresholds are never separately tuned.

Raw cumulative probabilities must decrease. Each sample is projected onto the
non-increasing unit interval by deterministic equal-weight Pool Adjacent Violators:
adjacent violating blocks are pooled to their mean until ordered; no labels are used.
Violation frequency and mean/maximum absolute correction are reported.

The fixed cumulative decoders are: (A) count raw binary decisions at probability 0.5;
(B) after monotonic correction reconstruct six non-negative class probabilities
`[1-q0, q0-q1, q1-q2, q2-q3, q3-q4, q4]` and take explicit-order argmax; and (C) compute
expected rank from these probabilities, round to nearest integer using deterministic
half-up rounding, clip to 0..5, and map back. Decoder C is deployable but secondary.
No thresholds or calibration are tuned.

Approach B fits deterministic sparse Ridge rank regression with `alpha=1.0`,
`solver="lsqr"`, and fixed rank targets 0..5 on the same training-only TF-IDF matrix.
Continuous outputs are half-up rounded, clipped to 0..5, and mapped back. Continuous,
rounded, clipping, and distribution diagnostics are retained.

## Metrics and analysis

All five methods—flat, cumulative hard, cumulative probability argmax, cumulative
expected rank, and Ridge—use the fixed six labels for fold and aggregate macro-F1
(primary), standard deviation, macro precision/recall, balanced accuracy, weighted F1,
accuracy, per-class precision/recall/F1/support, minimum recall, blocker/critical means,
distributions, confusion matrices, runtime, feature count, and serialized size.

Ordinal metrics are MAE, median absolute error, RMSE, quadratic weighted kappa, Spearman
correlation, exact rank accuracy, within-one and within-two accuracy, signed error,
over/underestimation rates, and distance counts 0..5. An extreme error is frozen as
absolute rank distance `>=3`. Specified blocker, critical, trivial, major/normal, and
minor/normal errors are reported. Each threshold reports binary macro-F1, balanced
accuracy, positive-class precision/recall/F1, support, predicted-positive rate, confusion
matrix, feature count, size, and fit/inference runtime. Development OOF comparison covers
corrected/worsened rows, exact/ordinal/extreme errors, minority false negatives, normal
overprediction, folds, text length, and early/middle/late periods. Persisted predictions
contain safe identifiers and labels/ranks/errors only.

## Frozen selection and challenger rules

A primary challenger requires mean exact six-class macro-F1 `>=0.2463`. Primary ties use:
higher macro-F1, lower standard deviation, lower MAE, higher quadratic kappa, lower extreme
rate, higher minimum recall, higher blocker/critical recall, lower cost, then simplicity.

A secondary development-only ordinal candidate must satisfy every condition: macro-F1
`>=0.2163`; MAE improves at least 10% relatively; quadratic kappa improves `>=0.0500`;
within-one accuracy improves `>=0.0300`; extreme-error rate falls at least 20% relatively;
blocker/critical recall declines no more than `0.0200`; no class loses over `0.1000` recall
without documented trade-off; and distribution remains plausible. Accuracy, weighted F1,
or sensitivity cannot select. Three folds permit descriptive differences, not statistical
superiority.

Only a primary method may create a disabled ordinal challenger. A secondary-only method
may create only a disabled configuration explicitly labeled development-only. Both require
`training.enabled: false`, `held_out_test.evaluation_allowed: false`, exact frozen settings,
and explicit new researcher approval. Otherwise a no-improvement document is created.
This study never accesses or evaluates the MYLYN held-out test.
