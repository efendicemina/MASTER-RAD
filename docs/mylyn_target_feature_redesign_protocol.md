# Frozen MYLYN target and feature-redesign protocol

This protocol is committed before real redesign results. Only the immutable 7,664-row
MYLYN `development_split.csv` (SHA-256
`8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5`) is model data.
The held-out test, rows outside this split, and other Eclipse projects are prohibited.
Prior predictions are references only and are never imported as new results.

## Fixed tasks and eligibility

S6 retains blocker, critical, major, normal, minor, trivial. S3 maps
blocker/critical=`HIGH`, major/normal=`MEDIUM`, minor/trivial=`LOW`. S2 maps
blocker/critical/major=`HIGH_IMPACT` and normal/minor/trivial=`LOWER_IMPACT`.
Enhancement, blank, missing, unknown, and unmapped severities are excluded. The tasks are
reported separately; S2/S3 scores never imply S6 improvement.

Metadata is `PRIMARY_ELIGIBLE` only when present at creation, not target/outcome-derived,
directly available or reliably reconstructed, low leakage-risk, and manageable.
`SENSITIVITY_ONLY` means likely early but initial value or stability is unproven.
`INELIGIBLE` means post-report, target/workflow-derived, personal, unstable, or unsafe.
Severity, Priority, Status, Resolution, Dupe-of, dependency fields, assignees, QA, CC,
Creator, Comments, History, attachments, last-change, deadline, milestone, flags,
whiteboard, confirmation/open status, year, and fold are always ineligible predictors.
History may only audit/reconstruct provenance. Missing values are explicit; categorical
rare levels have training-only count `<5` mapped to `[RARE]`, with unseen values handled
as unknown. No target encoding is allowed. If creation-time provenance cannot be proved,
metadata-enhanced F5–F7 are blocked rather than promoted.

## Data quality, duplicates, and folds

The audit covers reconciliation, severity normalization, missing/malformed text/time,
identifiers, exact/normalized/near duplicates, conflicts, lengths, HTML/entities,
code/stack traces, URLs/emails, Unicode, boilerplate, environment sections, direct label
tokens, drift, and metadata quality/association. Unusual rows are never automatically
removed or relabeled. Review samples are redacted and labels remain unchanged.

The approved policy retains the earliest exact-text duplicate, removes later copies,
preserves conflict handling, and prevents explicit duplicate groups crossing folds.
Normalized duplicate identities are computed without labels; training members crossing an
outer or inner validation group are purged deterministically while validation identity is
untouched. Every purge is recorded. Five known retained conflicts are excluded only in a
diagnostic sensitivity run and never select a model.

Outer folds are exactly 1916/1871, 3832/1855, 5748/1889 with fingerprint
`5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903`.
Within each outer training set, the same expanding-window splitter creates two inner
splits when all task classes exist; otherwise one chronological holdout is used. All
dataset/fold/row/preprocessing hashes and temporal boundaries are persisted. Outer
validation is untouched until one configuration is selected by inner macro-F1.

## Frozen preprocessing and feature sets

P0 is the approved Summary+Description baseline. P1 constructs
`[SUMMARY] ... [DESCRIPTION] ...` with Unicode normalization, HTML entity decoding and
visible-text tag removal, normalized lines/space, neutral URL/email/Bugzilla-ID/long-hex
placeholders, while preserving negation, technical terms, code and punctuation. P2 fits
independent Summary and Description word branches. P3 adds a bounded `char_wb` branch over
P1. P4 contains only initial-text structural counts/ratios and indicators listed in the
request, scaled on training only.

F0=P0 word TF-IDF `(1,2)`, min_df 2, max_df .98, max 50,000, sublinear. F1=P1 with the
same vectorizer. F2 uses Summary word `(1,2)`, max 20,000 and Description word `(1,2)`,
max 40,000. F3 adds scaled P4 to F2. F4 uses F2 plus cleaned concatenated `char_wb`
3–5 grams, min_df 3, max_df .995, max 30,000, and scaled P4. All fitting is training-only.
F5=eligible metadata only, F6=F3+metadata, F7=F4+metadata, and are run only with at least
one `PRIMARY_ELIGIBLE` field. Sensitivity-only metadata can never select or support claims.

Direct case-insensitive tokens blocker, critical, major, normal, minor, trivial, severity
are audited. A fixed diagnostic masks them all with `[SEVERITY_TERM]` independently of the
label. Primary natural text may retain creation-time wording, but token-driven gains are
reported and cannot alone select a challenger.

## Frozen candidate matrix and nested selection

For every task and eligible F0–F7 feature set evaluate LogisticRegression with
`C={0.25,1,4}`, class_weight `{None,balanced}`, `max_iter=2000`, `lbfgs`, seed 42; and
LinearSVC with `C={0.1,0.5,1,2}`, class_weight `{None,balanced}`, seed 42. ComplementNB is
omitted because F3/F4 scaled structural values can be negative and a uniform compatible
matrix is required. Dummy most-frequent is a non-selectable sanity bound.

Inner selection maximizes mean macro-F1, then minimum recall, HIGH/HIGH_IMPACT recall for
S3/S2, lower fold variance, simpler feature set (F0..F7 order), lower dimensionality,
runtime, and reproducibility. One winner is refitted on full outer training and evaluated
once. No random split, resampling, synthetic text, threshold tuning, validation fitting,
or adaptive search is allowed. Configuration stability is reported.

## Metrics, controls, and success rules

All outer results use fixed task labels for macro precision/recall/F1, balanced accuracy,
weighted F1, accuracy, per-class metrics/support, minimum recall, raw/normalized matrices,
distributions, dominant prediction rate, deterministic 1,000-sample combined-OOF bootstrap
macro-F1 interval, runtime, dimensions, serialized size, and RSS delta. S3 reports HIGH
misses to MEDIUM/LOW. S2 reports HIGH_IMPACT precision/recall/F1, LOWER_IMPACT recall,
specificity, NPV, balanced accuracy, predicted-positive rate, PR-AUC for LR and secondary
ROC-AUC.

Controls are one deterministic shuffled-training-label run on outer fold 1; strongest
text-only model; metadata-only/combined/field ablation only if primary metadata exists;
duplicate-boundary proofs; preprocessing-fit tests; label-token masking; and five-conflict
sensitivity. Shuffled performance must approach the task's chance/dummy range.

S6 baseline is `0.2262765`. `MATERIAL_IMPROVEMENT` requires macro-F1 `>=0.2463`, gains in
at least two folds, minimum mean-class recall loss no worse than 0.05, nonzero blocker and
critical recall, plausible distribution (all classes predicted; dominant share <85%), and
no leakage-risk metadata. S6 `PRACTICALLY_STRONG` additionally requires macro-F1 >=0.35,
balanced accuracy >=0.35, minimum recall >=0.15, and blocker/critical mean recall >=0.15.

S3 material requires `F0+0.0500`, gains in two folds, HIGH recall gain >=0.10, minimum
recall >=0.20, every class predicted with dominant share <85%, and safe features. Strong
requires macro-F1 >=0.50, HIGH recall >=0.35, minimum recall >=0.30, balanced accuracy
>=0.50, and stability (fold macro-F1 SD <=0.08).

S2 material requires `F0+0.0500`, gains in two folds, balanced accuracy >=0.63,
HIGH_IMPACT precision/recall >=0.45, improved HIGH_IMPACT F1, predicted-positive rate
between 2% and 60%, and safe features. Strong requires macro-F1 >=0.70, balanced accuracy
>=0.65, HIGH_IMPACT recall >=0.60, precision >=0.50, F1 >=0.55, plausible rate, and fold
SD <=0.08. Thresholds never change after results.

Each task is finally labeled NOT_PREDICTABLE_ENOUGH, RESEARCH_ONLY, PROMISING, or
PRACTICALLY_STRONG based on semantic validity, macro-F1, minority recognition, stability,
leakage safety, interpretation, and reproducibility—not largest F1 alone. Challenger YAML
is created only for material/strong results, always disabled with held-out evaluation false
and requiring explicit new researcher approval. This study never accesses the test set.
