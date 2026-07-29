# Eclipse pooled severity-classification protocol v1

This protocol is frozen before any pooled real-run or locked-test model evaluation. It asks
how accurately severity can be classified when BIRT, CDT, Equinox, JDT, MYLYN, PDE, Papyrus,
Platform and TPTP jointly define the target population. It is a separate claim from every
MYLYN-only or transfer experiment.

## Immutable inputs and reservations

For BIRT, CDT, Equinox, JDT, PDE, Papyrus, Platform and TPTP, the immutable Parquet inputs and
SHA-256 values are:

| Project | SHA-256 |
|---|---|
| BIRT | `a0e2c21c9095bcf250cc5d389177d2e6058b39cd589199316d235b00b532fa76` |
| CDT | `00cbe579b9a1a4606db80933154cb7a2498fc8931310805f033cb4558d725004` |
| Equinox | `f0ee1c9185c6edc119bae133fb64c81c3790559878f4f5279c12c189f337d5ba` |
| JDT | `0c1b597e9be4fce63a22cdac137e2f1f1657f8d6512fec689dc7be2233340c61` |
| PDE | `034ac8abccce6a5fcae49ad604faafd3f2f794c2d4dfd9a993430ef44e1d1808` |
| Papyrus | `b82272d153d1edefa03a82268099f878a4f3e0f9f5a652dbbf259e5f7cc366f4` |
| Platform | `21a82ca083594b286f226debf1bd98ad29063cef267d3809e6d160919d3bfdf1` |
| TPTP | `7aceea71ca29e14e11ce4953203acc5e924a75154715e100566543070b70eca7` |

MYLYN is never read from its full Parquet or reserved test. Its only permitted input is the approved
7,664-row `development_split.csv`, SHA-256
`8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5`.

For each non-MYLYN project, the latest 20% of eligible rows from the pre-existing per-project
chronological audit remains reserved and is excluded before this experiment. The earliest
80% is the allowed legacy-development universe. MYLYN uses its approved development split as
that universe. No prior reserved test row may enter pooled development, pooled locked test,
timing pilots, vocabulary fitting, calibration, thresholding or diagnostics.

## Inclusion, labels and text

Rows require a valid UTC creation time and a normalized severity in
`blocker, critical, major, minor, normal, trivial`. Enhancement, task, feature, missing,
blank, unknown and all other labels are excluded. The unchanged task mappings are:

- S6: the six normalized labels.
- S3: blocker+critical = HIGH; major+normal = MEDIUM; minor+trivial = LOW.
- S2: blocker+critical+major = HIGH_IMPACT; minor+normal+trivial = LOWER_IMPACT.

Text is `summary + " [SEP] " + description`, with missing fields replaced by empty strings.
Cleaning uses the repository software-aware normalization: Unicode normalization, HTML and
URL handling, issue-ID masking, lowercasing, and software-token preservation. Explicit
severity terms are masked in both fields. Project, product, component, IDs, URLs, reporters,
assignees and duplicate metadata are not predictive features.

## Partitioning and duplicate protection

Within each allowed project universe, rows are stably sorted by `(creation_time,
global_report_key)` and split 80/20: earliest rows are pooled development and latest rows are
the new locked test. MYLYN uses its privacy-safe row key as the secondary order. Exact-text
hash and linked `dupe_of`/duplicate-group relationships form connected components across all
nine projects. Any group represented in development is removed from locked test; no group is
moved from development into test. Empty project partitions or any row/group overlap fail.

The locked-test row identities, labels and text are sealed after partition construction.
Plan/dry-run may report aggregate counts and hashes only. Validation before final selection
may verify file existence and hash but may not load test content. Model evaluation on locked
test is a separate, explicit, one-shot command after all task candidates are frozen.

Development uses three deterministic expanding-window folds over the pooled chronological
order. Validation rows whose duplicate group occurs in training are purged. Every fold must
contain every task label overall; per-project class absence is reported, not silently filled.
Seed is 42 for all stochastic controls, bootstrap and estimators. Sorting uses stable
mergesort; BLAS/thread counts and package versions are recorded.

## Compute-efficient staged search

Stage 0 evaluates two development-only controls per fold: most-frequent DummyClassifier and
the repository pooled linear baseline. Stage 1 uses one frozen screening split capped at the
earliest 4,000 allowed development rows per project (at most 36,000 rows). Its 36 candidates
are the Cartesian product of:

- representations: word TF-IDF `(1,2)`, character `char_wb (3,5)`, and word+character;
- LinearSVC `C in {.25, 1}` or LogisticRegression `C=.5`, lbfgs, max_iter 2000;
- class weight `None` or `balanced`;
- project sample weighting `none` or normalized inverse square-root project frequency.

Word/character branches use `min_df=3`, `max_df=.98`, sublinear TF, float32 and caps of
75,000 features each; the union cap is 100,000 effective features. Stage 1 prunes candidates
whose Macro-F1 is more than .03 below the representation/model-family leader, then ranks the
remainder. Stage 2 evaluates exactly the best six eligible candidates per task on all three
frozen development folds. Stage 3 selects one candidate per task without locked-test access.

Budget per task is 6 Stage-0 fits + 36 Stage-1 fits + 18 Stage-2 fits = 60 development fits;
180 fits total. Future one-shot locked evaluation adds one refit/evaluation per task, for a
maximum experiment budget of 183 fits. Failed candidates are recorded and skipped. A task
stops only when no eligible candidate remains at a required stage.

## Ranking, thresholding and metrics

Pooled Macro-F1 is primary. Stage-1 and Stage-2 ranking uses, in order: higher pooled
Macro-F1, higher macro-average of per-project Macro-F1, higher minimum class recall, higher
balanced accuracy, lower fold Macro-F1 SD, lower feature count, lower runtime, then stable
candidate ID. Outer/locked results never alter selection.

Required outputs are accuracy, weighted F1, balanced accuracy, fixed-label per-class
precision/recall/F1/support, raw and normalized confusion matrices, per-project metrics,
macro-average and minimum of per-project Macro-F1, predicted distributions, dimensions,
runtime, RAM and model size. Large projects and majority classes are therefore not sufficient
for success.

S2 calibration is fit only from development OOF scores. Thresholds `.05–.95` by `.005`
maximize HIGH_IMPACT F2 subject to precision `>=.30`, then recall, precision, Macro-F1,
distance to `.50`, higher threshold and stable numeric order. Candidate-level infeasibility
is recorded with best precision/threshold, class counts and predicted positives; the candidate
is excluded. The task fails only if every eligible candidate is infeasible.

Paired bootstrap uses 2,000 identical row resamples, seed 42, and percentile 95% intervals
for Macro-F1 delta versus the strongest frozen Stage-0 comparator. A second project-block
bootstrap resamples nine projects with replacement and reports the same delta. Misaligned
predictions fail closed.

## Preregistered success criteria

All criteria require positive row-bootstrap and project-bootstrap lower CI bounds and gains
in at least two of three development folds before locked evaluation is scientifically useful.

- S6: Macro-F1 >= .35, per-project Macro-F1 mean >= .28, minimum class recall >= .10,
  nonzero blocker/critical recall, all labels predicted and dominant prediction <85%.
- S3: Macro-F1 >= .50, per-project Macro-F1 mean >= .42, minimum recall >= .25, HIGH recall
  >= .30, all labels predicted and dominant prediction <85%.
- S2: Macro-F1 >= .70, balanced accuracy >= .65, per-project Macro-F1 mean >= .60,
  HIGH_IMPACT precision >= .50, recall >= .55 and F1 >= .52, positive rate 2–60%.

Accuracy alone never establishes success. Failure of an absolute criterion is reported even
when transfer over a baseline is statistically positive.

## Checkpoints, provenance and publication

Plan, dry-run, real-run and validation use separate output directories. Real-run requires
this exact protocol content hash and its Git commit. Manifests lock input hashes, split and
fold fingerprints, candidate checksum, package versions, seed and canonical CLI. Atomic
checkpoints are written per task/stage/fold/candidate and `--resume` validates all locks.

Privacy-safe publication includes aggregate audit, candidate status, selected configuration,
fold/task/project/class metrics, confusion matrices, bootstrap intervals, resources,
deviations and a validation bundle. Raw data, locked rows, individual predictions or IDs,
models, large grids, engine state, caches and temporary files are never committed.

After a one-shot locked evaluation, metrics and the decision are frozen before any optional
final model training. Training on 100% of allowable pooled data is never automatic and
requires a separate protocol and authorization.
