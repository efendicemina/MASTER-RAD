# VALIDATION BUNDLE — MYLYN transfer-study v2

## Completion and provenance

- Run: `20260727T203800_real_run`
- Frozen protocol commit: `42da410e3e8d693f9ca6e1bb90b53d8e7ae1e3b3`
- Development SHA-256: `8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5`
- Outer-fold fingerprint: `5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903`
- Manifest: `dry_run=false`, `models_fitted=true`, `held_out_test_accessed=false`
- Engine state: exactly 9 completed task/folds and 16,845 aligned OOF rows
- Checkpoints: S6, S3 and S2 folds 1–3 each record stages A–E
- Baseline reproduction: S6, S3 and S2 reproduced with absolute difference `0.0`

S6 and S3 completed before the S2 control-flow correction. S2 completed after
`b851359cf0a688ca53a69c67b628aca2d2152195`. The correction did not change data, source
projects, folds, preprocessing, model space, threshold grid (`0.05–0.95` by `0.005`), frozen
precision constraint (`0.30`), ranking criteria or tie-break rules. It records and excludes
candidate-level infeasibility and stops only when every eligible candidate is infeasible.

## Task-level metrics

Matched target-only is the fold-matched MYLYN-only comparator used by paired bootstrap.
Frozen baseline is the preregistered historical MYLYN-only Macro-F1 reference.

| Task | Frozen baseline Macro-F1 | Matched target-only Macro-F1 | Transfer Macro-F1 | Transfer accuracy | Transfer weighted F1 | Transfer balanced accuracy | Delta vs frozen | Delta vs matched | Paired delta 95% CI | Fold wins | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S6 | 0.226277 | 0.221029 | 0.240741 | 0.463392 | 0.487562 | 0.269026 | +0.014465 | +0.019712 | [-0.008909, 0.026739] | 3/3 | RESEARCH_ONLY |
| S3 | 0.397187 | 0.415912 | 0.416616 | 0.627376 | 0.640448 | 0.433507 | +0.019429 | +0.000704 | [-0.011723, 0.014823] | 1/3 | RESEARCH_ONLY |
| S2 | 0.591776 | 0.520098 | 0.578578 | 0.849638 | 0.837443 | 0.580495 | -0.013197 | +0.058480 | [0.021010, 0.052222] | 2/3 | NOT_PREDICTABLE_ENOUGH; clear matched transfer evidence |

Bootstrap mean deltas are S6 `0.008720`, S3 `0.001845`, and S2 `0.036466`. They are pooled,
paired row-resampling estimates and therefore need not equal the arithmetic mean of the three
fold deltas.

## Outer-fold results and selected configurations

| Task/fold | Matched Macro-F1 | Transfer Macro-F1 | Delta | Accuracy | Weighted F1 | Balanced accuracy | Selected configuration |
|---|---:|---:|---:|---:|---:|---:|---|
| S6/1 | 0.208105 | 0.257616 | +0.049512 | 0.552646 | 0.556654 | 0.268183 | LR C=.25 balanced, R2/SW3, source=.10, five-year, plain |
| S6/2 | 0.211694 | 0.213100 | +0.001406 | 0.414555 | 0.432099 | 0.228795 | LR C=1 balanced, R1/SW1, source=.25, all-history, plain |
| S6/3 | 0.243287 | 0.251508 | +0.008221 | 0.422975 | 0.473934 | 0.310101 | LR C=.25 balanced, R2/SW1, source=.25, all-history, plain |
| S3/1 | 0.423908 | 0.426019 | +0.002111 | 0.664351 | 0.660659 | 0.434603 | LR C=.25 balanced, R2/SW2, source=.10, five-year, plain |
| S3/2 | 0.413573 | 0.413573 | 0.000000 | 0.599461 | 0.611180 | 0.420919 | LR C=.25 balanced, R2/SW1, source=0, half-life 2, plain |
| S3/3 | 0.410255 | 0.410255 | 0.000000 | 0.618317 | 0.649505 | 0.444999 | LR C=.25 balanced, R2/SW1, source=0, half-life 4, plain |
| S2/1 | 0.483534 | 0.581972 | +0.098439 | 0.883485 | 0.864130 | 0.566061 | LR C=.25, R2/SW2, source=.50, five-year, plain; threshold=.24 |
| S2/2 | 0.464337 | 0.542220 | +0.077883 | 0.852830 | 0.820509 | 0.537876 | LinearSVC C=1, R0/SW1, source=.25, half-life 4, plain; threshold=.19 |
| S2/3 | 0.612423 | 0.611543 | -0.000880 | 0.812599 | 0.827690 | 0.637548 | LR C=.25 balanced, R2/SW1, source=.50, all-history, domain-augmented; threshold=.19 |

## S2 infeasibility audit

All 159 infeasible Stage-A candidate/fold evaluations have reason
`no_s2_threshold_meets_precision_constraint`, constraint `0.30`, and best precision below
`0.30`. Fold 1 retained 26/147 feasible candidates, fold 2 retained 127/147, and fold 3
retained 129/147. The selected candidate in every fold was feasible in that fold; none of
the fold-specific rejected candidates entered ranking or selection.

## Per-class and confusion-matrix findings

- S6 aggregate recalls: blocker `0.180`, critical `0.130`, major `0.214`, minor `0.362`,
  normal `0.562`, trivial `0.206`. Blocker/critical precision is only `0.066/0.058`.
  The 61 blocker examples produced 11 correct predictions; the 115 critical examples
  produced 15. Normal remains dominant (2,089 true positives), with substantial minority
  confusion into normal and minor. This prevents a material/production claim.
- S3 aggregate HIGH recall is `0.170` with precision `0.127` (30/176 correct). MEDIUM recall
  is `0.707` and dominates the confusion matrix; 120 HIGH cases were predicted MEDIUM.
- S2 HIGH_IMPACT precision is exactly `0.300`, recall `0.227`, and F1 `0.258` (147/649 true
  positives). LOWER_IMPACT recall is `0.931`. Although paired transfer evidence is positive,
  S2 fails frozen deployability requirements (balanced accuracy `0.580 < 0.63`, and
  high-impact precision/recall below `0.45`).

Aggregate confusion matrices, using the documented label order:

- S6 `[blocker, critical, major, minor, normal, trivial]`:
  `[[11,4,6,11,29,0],[7,15,23,28,40,2],[24,48,101,108,179,13],`
  `[14,19,48,300,389,58],[108,167,338,785,2089,233],[2,4,10,117,199,86]]`
- S3 `[HIGH, LOW, MEDIUM]`: `[[30,26,120],[21,527,698],[185,1042,2966]]`
- S2 `[HIGH_IMPACT, LOWER_IMPACT]`: `[[147,502],[343,4623]]`

## Integrity, privacy and artifact validation

- Protocol content SHA-256 still matches the frozen manifest; `protocol_deviations.md`
  records none. The implementation correction is a control-flow/auditability fix, not a
  post-hoc scientific rule change.
- All 10 pre-run integrity checks pass; cutoff violations and target/source overlap are zero.
- Every outer validation set is evaluated exactly once; `(task, fold)` keys are unique.
- All required CSV outputs are non-empty. Task-specific `NaN` fields in the wide fold table
  and inapplicable `alpha`/ensemble fields are structural, not missing results.
- OOF keys are unique by `(task, row_key)` and contain no summary, description, issue ID,
  reporter, assignee or email columns. Privacy scan found no email-like values.
- Held-out test access is explicitly false. No held-out labels, predictions or metrics were
  inspected or used in this validation.
- Targeted artifact validation: PASS (19 required CSVs, 9 folds, 16,845 OOF rows).
- Targeted S2/resume tests: PASS (`6 passed`).
- Complete pytest suite: PASS (`141 passed`, 9 third-party warnings).
- Ruff: PASS.

## Final scientific conclusion

S6 and S3 show small exploratory improvements but no CI-supported transfer effect and remain
research-only. S2 has clear paired evidence that source transfer improves the matched
target-only configuration, but it remains below the frozen absolute quality and minority
detection requirements and is not predictable enough for deployment. These results do not
authorize a pooled Eclipse experiment or held-out-test evaluation.
