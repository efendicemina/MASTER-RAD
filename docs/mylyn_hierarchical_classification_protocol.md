# Frozen MYLYN hierarchical severity-classification protocol

This protocol is frozen before any real hierarchical result is produced. The study uses
only the immutable MYLYN `development_split.csv` (SHA-256
`8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5`), Summary,
Description, Severity, and technical identifiers needed to preserve folds and duplicate
groups. The fixed final labels are blocker, critical, major, normal, minor, and trivial.
Enhancement, blank, missing, unknown, and unmapped labels are excluded. No metadata is a
predictor. The held-out test and every other Eclipse project are prohibited.

## Frozen data, preprocessing, folds, and duplicates

Text is constructed deterministically from Summary, a separator, and Description using
the approved baseline preprocessing: lowercase, HTML removal, URL and email replacement,
Unicode and whitespace normalization, while code blocks and stack traces are retained.
The leakage-safe sklearn Pipeline contains word TF-IDF `(1, 2)`, `min_df=2`,
`max_df=0.98`, `max_features=50000`, `sublinear_tf=True`, followed by LogisticRegression
`C=1.0`, `class_weight="balanced"`, `max_iter=2000`, default `lbfgs` solver, and seed 42.
Every transformer and classifier is fitted only on a fold's training rows.

The exact three approved duplicate-purged expanding-window folds are reused. Their frozen
fingerprint is `5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903`.
Fold identifiers, train/validation row hashes, preprocessing fingerprint, and dataset
fingerprint are persisted. Temporal boundaries and row membership cannot change.

The approved duplicate policy retains the chronologically earliest valid exact-text
duplicate, removes later exact-text duplicates, prevents explicit Dupe-of groups from
crossing folds, and preserves existing conflict handling. No removal is silent. A separate
diagnostic excludes all rows in the five known conflicting exact-text groups. It is not
the main dataset and cannot select a model.

## Fixed hierarchies and routing

Hierarchy A maps blocker/critical to `SEVERE` and major/normal/minor/trivial to
`NON_SEVERE`. Its children are a blocker-versus-critical classifier and a four-class
major/normal/minor/trivial classifier.

Hierarchy B maps blocker/critical to `HIGH`, major/normal to `MEDIUM`, and minor/trivial
to `LOW`. Its children are the corresponding three binary classifiers. LogisticRegression
with the fixed baseline pipeline is used at every node; nodes are never tuned separately.
Every required node must contain at least two training classes or execution fails clearly.

Hard routing uses the predicted Level-1 group and that child's prediction. Soft routing
computes, in explicit class order, `P(group|text) * P(class|group,text)` for all six final
labels and selects the maximum without calibration or threshold tuning. Balanced Logistic
Regression probabilities need not be calibrated. Oracle routing uses the true validation
group only to choose a child and is stored as `NON-DEPLOYABLE DIAGNOSTIC ONLY`; it cannot
select a challenger.

## Metrics and error decomposition

The reproduced flat model, hard, soft, and oracle outputs use the fixed six-label list for
fold and aggregate macro precision/recall/F1, balanced accuracy, weighted F1, accuracy,
per-class precision/recall/F1/support, minimum class recall, blocker/critical means,
predicted distributions, confusion matrices, runtime, feature count, and serialized model
size. Level 1 additionally reports macro-F1, balanced accuracy, group recalls and confusion
matrices. Each child reports accuracy, balanced accuracy, macro-F1, per-class metrics,
support, and predicted distribution.

Development OOF errors are classified as correct, Level-1 routing error, or within-group
child error. Counts, percentages, source classes, predicted groups, child nodes, specified
neighbor confusions, and hard-versus-soft differences are reported against the reproduced
flat OOF predictions. Oracle values diagnose routing headroom only.

## Baseline gate and selection

The approved flat mean CV macro-F1 is `0.2263`. Reproduction tolerance is absolute
`0.0001`; exceeding it is a blocking warning and prevents challenger creation.

A deployable hard or soft candidate meets the primary criterion only at mean macro-F1
`>=0.2463`. Primary candidates rank by: higher mean macro-F1, lower fold standard
deviation, higher minimum per-class recall, higher blocker/critical recall, lower routing
error, lower cost, simpler hierarchy, then easier reproducibility.

A secondary development-only minority-benefit candidate must simultaneously have macro-F1
`>=0.2163`, improve mean blocker/critical recall by `>=0.0500` over the reproduced flat
model, lose no class by more than `0.1000` recall without a documented practical trade-off,
and retain a plausible distribution not explained only by excessive HIGH/SEVERE output.
This does not authorize held-out evaluation. Accuracy and weighted F1 are never primary;
oracle and conflict sensitivity never select; three folds support no significance claim.

Only a primary candidate may produce the disabled hierarchical challenger configuration.
A secondary candidate produces only a disabled, explicitly development-only configuration.
Otherwise a no-improvement document is produced. Every configuration sets
`training.enabled: false` and `held_out_test.evaluation_allowed: false`; any future test
evaluation requires explicit researcher approval. This study never opens or evaluates the
MYLYN held-out test.
