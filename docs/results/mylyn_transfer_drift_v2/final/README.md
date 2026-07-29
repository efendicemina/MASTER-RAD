# MYLYN transfer-study v2 final results

This directory is the lightweight, privacy-safe publication package for the completed
development-only run `20260727T203800_real_run`. Large engine state, candidate grids, raw
source data and OOF output remain under ignored `reports/` paths and are not committed.

The frozen protocol commit is `42da410e3e8d693f9ca6e1bb90b53d8e7ae1e3b3`. S6 and S3
completed before the S2 candidate-infeasibility control-flow correction. S2 completed after
commit `b851359cf0a688ca53a69c67b628aca2d2152195`. That correction retained the same approved
development data, source data, folds, model space, threshold grid, precision constraint and
selection/tie-break rules; it only rejected an infeasible candidate instead of terminating
the entire S2 task.

`task_metrics.csv` distinguishes the frozen historical MYLYN-only baseline from the matched
target-only comparator used for paired transfer inference. `fold_metrics.csv` contains every
outer-fold result for both the selected transfer model and its matched comparator.
`per_class_aggregate.csv` and the original fold-level `per_class_results.csv` document class
behavior. `s2_infeasibility_summary.csv` is aggregated from structured Stage-A diagnostics;
the three fold identities are recovered from their unique inner-OOF class-count signatures.

See `VALIDATION_BUNDLE.md` for the final audit and scientific interpretation.
