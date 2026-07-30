# Frozen Eclipse pooled CPU impact ensemble protocol v1

This is a CPU-only, exploratory development study based on source commit
`631bc39c6621f39dda0c39925a10cb4181dc53d9`. It uses exactly the approved pooled
development fingerprint `cde41dd37386b38f6d3009ff42bff9c063d6d140da36bf781faefe34c388c01b`
and frozen fold fingerprint `727b2af6b8451adee768bc60fb6704e7459b09911b8d01099dc6df96d5c6b3bc`.
The pooled locked test, reserved MYLYN held-out test and raw MYLYN file are prohibited.
No manual, LLM-generated or model-generated replacement labels are permitted.

## Frozen targets

| Repository severity | Old S3 reference | S2-impact | S3-impact |
|---|---|---|---|
| blocker | HIGH | HIGH_IMPACT | HIGH |
| critical | HIGH | HIGH_IMPACT | HIGH |
| major | MEDIUM | HIGH_IMPACT | HIGH |
| normal | MEDIUM | LOWER_IMPACT | MEDIUM |
| minor | LOW | LOWER_IMPACT | LOW |
| trivial | LOW | LOWER_IMPACT | LOW |

S2-impact is directly comparable with rescue S2 R-05 on identical OOF rows. S3-impact is
a different target from old S3; its ensemble is compared only with a newly trained L-S3I
same-target lexical baseline. S6 is diagnostic only and receives no new model.

For both S2-impact and S3-impact, a deterministic most-frequent `DummyClassifier` reports
accuracy, Macro-F1 and balanced accuracy. Dummy is a non-selectable sanity baseline. No
scientific improvement, bootstrap delta or success claim may use Dummy as comparator; all
claims use L-S2 or L-S3I on identical rows.

## Isolation, features and preprocessing

Every outer/inner fit, vocabulary, embedding cache, retrieval index, calibration and stacker
uses training rows only. Duplicate groups cannot cross an inner split and a query cannot
retrieve its own protected group. Outer validation is used once after inner selection.
Locked/reserved rows never enter preprocessing, cache generation or diagnostics.

Allowed inputs are project, product, component, Summary and Description. Status, resolution,
History, comments, reporter, assignee, QA contact, identities and all post-submission fields
are forbidden. History/duplicates may support aggregate audit/split protection only; History
never affects a feature or sample weight.

P-00 is the rescue field-aware representation. P-01 always preserves Summary; constructs a
deterministic Description head and tail; preserves technical/error/camelCase tokens and adds
de-camelized copies; replaces email, URL, long-ID and path patterns; retains stop words; and
uses project/product/component. P-00 versus P-01 is selected only with representative-fold
outer-training inner OOF predictions. No stemming, lemmatization or broad preprocessing grid.

## Frozen representations and candidates

L-S2 and L-S3I use the strongest defensible field-aware lexical architecture with mild
`balanced_weight**0.25`, training-only rare metadata, duplicate inverse-square-root weighting
and mean-one final weights. S2 threshold selection uses inner OOF predictions only.

Frozen embeddings use `sentence-transformers/all-MiniLM-L6-v2` at a recorded immutable
revision, CPU inference only, no gradients/fine-tuning. Summary, Description head and tail
are encoded separately, each L2-normalized, then concatenated and normalized. Cache keys are
hashes of sanitized text block, preprocessing version and model revision; caches contain no
text, IDs or labels. PyTorch CPU is required; ONNX/OpenVINO are benchmarked only if already
available or safely installable. Quantization requires numerical/ranking validation.

BM25 is fitted separately inside every split, with Summary weight 2, global and same-project
neighbors, duplicate-group exclusion, and `k` selected from `{5,15,30}` by inner OOF only.
Features are class vote distribution, agreement, best/mean score, global/local vote delta
and entropy. No retrieved text or identifier is published.

Automatic label quality is strictly cross-fitted within outer training. Repository labels
remain unchanged. `label_weight=clip(0.50+0.50*P(repository_label),0.50,1.00)` composes once
with duplicate weighting and mild class weight, then normalizes to mean one. Unweighted and
weighted ablations are both mandatory.

A small L2 LogisticRegression stacker consumes only cross-fitted lexical, BM25 and frozen
MiniLM probabilities/logits plus disagreement/entropy and training-fitted safe project
indicators. It never receives in-sample base predictions.

The fixed candidate set is C-00 lexical baseline, C-01 lexical+BM25, C-02 lexical+MiniLM,
C-03 lexical+BM25+MiniLM, and C-04 C-03 with soft label-quality weights. S3-impact also
compares flat and hierarchical forms using identical outer rows. Hierarchy is HIGH versus
NOT_HIGH, then MEDIUM versus LOW; combined probabilities must be nonnegative and sum to one.
No candidate expansion, broad seed search, transformer training or GPU use is permitted.

## Stages and fit/resource budget

The representative outer fold is deterministically the median-sized validation fold before
new metrics: fold 2 (40,171 rows; folds 1/3 have 40,327/40,487). Stage A selects P-00/P-01,
two lexical/embedding regularization values, BM25 k, weighting and hierarchy through three
group-aware inner folds of fold-2 training, then evaluates each frozen C-00..C-04 once on
fold-2 outer validation. Stage B advances at most the best unweighted and best weighted
architecture per task and evaluates them on all three outer folds, reusing completed fold 2.

The implementation records every base/inner/outer fit. The hard architecture budget is five
Stage-A candidates per task plus at most two Stage-B architectures per task; inner base
predictions are cached and reused across ablations. Embeddings are reused across tasks, never
fitted classifiers/indexes/calibrators/stackers. Seed is 42.

Before full extraction, a deterministic approximately 2,000-row development benchmark records
CPU/core/RAM/disk/package/backend details, embedding throughput/cache projection, BM25 speed,
classifier runtime and projected peak RSS. Execution proceeds only if projected total runtime
is <=12 hours, peak RSS is below min(8 GiB,70% available RAM), temporary disk <10 GiB, and at
least two logical cores can remain free. Backend/batching/thread optimization precedes any
resource-blocked conclusion; dataset/folds/protocol cannot be reduced.

## Metrics, thresholds and ranking

Primary metric is pooled OOF Macro-F1. Secondary metrics are project-macro Macro-F1,
accuracy, weighted F1, balanced accuracy, per-class precision/recall/F1/support, actual and
predicted prevalence, fold/project results, confusion matrices, paired 2,000-sample row and
project bootstrap deltas, runtime/RSS/disk/throughput. Ranking uses Macro-F1, project-macro,
minimum recall, balanced accuracy, lower complexity/runtime and stable candidate ID.

S2 additionally reports HIGH_IMPACT precision/recall/F1, PR-AUC, default .5 metrics and an
inner-selected threshold. Thresholds first require precision >=.35 and recall >=.50, then
maximize Macro-F1 with deterministic recall/precision/distance ties. If infeasible, inner
Macro-F1 is maximized and the operational gate is marked failed. Outer labels never select it.

S3-impact additionally reports ordinal MAE (LOW<MEDIUM<HIGH), quadratic weighted kappa,
within-one-band accuracy, and flat versus hierarchy on identical rows.

## Scientific gates

S2 compares only with exact-row S2 R-05 and requires Macro-F1 delta >=.015,
non-decreasing project-macro, >=2/3 fold wins, >=6/9 project wins, HIGH_IMPACT precision
>=.35 and recall >=.50. Scores .62-.65 are aspirational only.

S3-impact compares only with L-S3I and requires delta >=.015, non-decreasing project-macro,
>=2/3 fold wins, >=6/9 project wins, improved HIGH and LOW F1, and both recalls >=.25.
Absolute .50 Macro-F1 is reported but never achieved by mapping manipulation.

Passing either development gate does not authorize locked evaluation. Final recommendation is
exactly one of `CPU_ENSEMBLE_QUALIFIES_FOR_PROTOCOL_REVIEW`,
`S3_IMPACT_TARGET_IS_VIABLE_BUT_ENSEMBLE_ADDS_NO_RELIABLE_GAIN`,
`CPU_APPROACH_DID_NOT_IMPROVE_RELIABLY`, or `CPU_EXECUTION_RESOURCE_BLOCKED`.

## Checkpointing, privacy and publication

Atomic, hash-validated checkpoints occur per embedding batch, retrieval index, inner fold,
outer fold, candidate and pre-bootstrap state. Resume skips only complete compatible work.
Manifest locks source/protocol commits and hashes, input/development/fold/candidate/model
revision/preprocessing/package/backend/thread/canonical-command details and always records
`held_out_test_accessed=false`, `locked_evaluation_authorized=false`.

Only source/tests/config/protocol/docs and lightweight aggregate tables are publishable.
Weights, raw/processed text, embeddings, caches, predictions, IDs, labels, row-quality scores,
BM25 indexes, checkpoints and engine state remain local and untracked.
