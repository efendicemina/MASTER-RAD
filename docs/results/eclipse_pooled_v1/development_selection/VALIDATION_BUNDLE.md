# Eclipse pooled v1 development-selection validation

## Decision

**DO_NOT_EVALUATE_AND_START_RESCUE_PLAN.** The locked evaluation remains unauthorized and
was not accessed. S6 and S3 have stable, project-wide gains over the strongest Stage-0
baseline, but fail their preregistered absolute quality thresholds. S2 fails both the
approximately +.02 development-gain expectation and nearly every absolute success threshold.

## Provenance and artifact validation

- Branch: `experiment/eclipse-pooled-v1`.
- Frozen protocol commit: `428af8aa61ef490d50c7a5e1eb1e47f38999cf4c`.
- Protocol SHA-256: `159f9c7d54a49a9cb354214708e67adbbbd4f4b987a7050c4c263e2fb0ad2400`.
- Manifest mode is `real-run`, `models_fitted=true`, `held_out_test_accessed=false`.
- Development/test fingerprints and all nine input hashes are present and unchanged.
- Engine state contains exactly 183 unique completed keys: 61 per task.
- Every task has 6 Stage-0, 36 Stage-1, 18 Stage-2 and one Stage-3 checkpoint.
- `candidate_results.csv` has 180 unique task/stage/candidate/fold rows.
- Selection, bootstrap and per-project tables have exactly 3, 3 and 27 rows.
- No duplicate output rows, missing required task/project pairs or all-null columns were found.
- Built-in artifact validation returned `VALIDATION_PASS`; privacy-safe filename checks passed.
- Raw predictions, engine state, checkpoints, identifiers, locked rows and models are not
  included in this bundle.

## Development results

| Task | Best baseline Macro-F1 | Selected | Macro-F1 | Project-macro | Accuracy | Weighted F1 | Balanced acc. | Delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| S6 | .185108 | P-002 | .246622 | .227608 | .681561 | .668051 | .247220 | +.061514 |
| S3 | .350880 | P-011 | .421434 | .406762 | .665601 | .721527 | .522954 | +.070554 |
| S2 | .583523 | P-002 | .591893 | .584903 | .710079 | .741849 | .642519 | +.008370 |

The strongest baseline is the frozen `POOLED_LINEAR_BASELINE` for every task. All selected
models win all three folds and all nine projects. Paired row-bootstrap 95% delta intervals
are S6 `[.058679,.067017]`, S3 `[.066705,.075737]`, S2 `[.006661,.010669]`;
project-bootstrap intervals are S6 `[.050959,.068931]`, S3 `[.060023,.088713]`, and
S2 `[.005911,.013572]`. Thus gains are consistently positive, but statistical positivity
does not replace the frozen absolute criteria.

Selected configurations:

- S6 P-002: word TF-IDF, LinearSVC, C=.25, balanced class weights, no project weights.
- S3 P-011: word TF-IDF, LogisticRegression, C=.5, balanced class weights,
  inverse-square-root project weights.
- S2 P-002: word TF-IDF, LinearSVC, C=.25, balanced class weights, no project weights.

## Minority classes and S2 feasibility

S6 recall rises from `.0168` to `.1547` for blocker, `.0265` to `.1343` for critical,
`.0860` to `.1568` for major, `.0158` to `.0994` for minor, and `.0108` to `.0933` for
trivial. S3 HIGH recall rises from `.0415` to `.4442`, and LOW from `.0277` to `.4266`.
These recall gains come with materially lower minority precision and remain insufficient for
the absolute task thresholds.

All 36 S2 Stage-1 candidates and all 18 S2 Stage-2 fold evaluations were feasible; no
candidate was rejected as infeasible. The frozen calibration constraint remained HIGH_IMPACT
precision >=.30. Selected fold thresholds are `.185`, `.195`, and `.180`. On the subsequent
development validation folds, mean HIGH_IMPACT precision/recall/F1 are `.2873/.5421/.3724`.
Validation precision below .30 does not alter the calibration constraint; it is an
out-of-sample result after threshold selection.

## Frozen success-criteria assessment

- S6 fails Macro-F1 `.35`, project-macro `.28`, and minimum recall `.10` (observed `.0863`).
- S3 fails Macro-F1 `.50` and project-macro `.42`; it passes minimum recall and HIGH recall.
- S2 fails Macro-F1 `.70`, balanced accuracy `.65`, project-macro `.60`, HIGH_IMPACT
  precision `.50`, recall `.55`, and F1 `.52`. Its Macro-F1 gain is only `.0084`.
- Positive row/project CIs, 3/3 fold wins and 9/9 project wins pass for every task.

Consequently, locked evaluation is not scientifically justified under the frozen protocol.
A rescue plan must be separately designed and preregistered using development data only.

## Software validation

- Targeted pooled tests: `13 passed`.
- Complete suite: `154 passed, 9 warnings`.
- Ruff: `All checks passed!`.
- The warnings are pre-existing third-party NumPy/joblib/scikit-learn warnings.

Machine-readable companion tables contain the task summary, fold results, minority metrics
and per-project baseline comparisons. Values are aggregate development-only results.
