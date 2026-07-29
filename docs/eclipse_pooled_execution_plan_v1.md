# Eclipse pooled v1 execution plan

## Scope and gates

The experiment branch is `experiment/eclipse-pooled-v1`, based on finalized MYLYN results
commit `e92eff2747586baee7480a2524e425bca25e0071`. The frozen pooled protocol is commit
`428af8aa61ef490d50c7a5e1eb1e47f38999cf4c`. MYLYN transfer v2 is immutable.

Preparation has three gates:

1. `plan` verifies inputs, reservations, partitions, duplicate components, folds, grid and
   protocol provenance without fitting.
2. `dry-run` performs only three small development pilot fits and estimates resources.
3. `real-run` performs the preregistered 180-fit development search and freezes one candidate
   per task. It deliberately leaves `held_out_test_accessed=false` and does not train on 100%
   of data. Locked evaluation requires a later explicit implementation/authorization after
   development outputs validate; it cannot be triggered by this command.

## Deterministic search budget

Per task: 6 Stage-0 control fits, 36 Stage-1 screening fits, and top 6 candidates × 3 Stage-2
folds = 18 fits. Stage 3 is deterministic aggregation without a fit. Total: 60 per task and
180 across S6/S3/S2. Failed or S2-infeasible candidates are recorded and excluded; a stage
fails only if no eligible candidate remains.

The 36 candidates combine three text representations, LinearSVC C `.25/1` or LR C `.5`,
class weighting none/balanced, and project weighting none/inverse-square-root. Stage 1 uses
at most 4,000 early development rows per project. Stage 2 uses all development rows and all
three frozen folds. Ranking follows the frozen protocol.

## Resource envelope

The final pilot used 2,700 development rows and three fits in 27.53 seconds, or 9.18 seconds
per fit. Conservative nonlinear scaling to 165,998 development rows estimates 24,026 seconds
(6.67 hours) for 180 fits. Reserve at least 8 GB RAM and 250 MB output disk; 12 GB available
RAM is preferred because native sparse allocations were released before the endpoint RSS
measurement and the measured delta was therefore zero. CPU-only execution is required.

## Future commands

Development-selection real-run (do not run until separately authorized):

```powershell
.\.venv\Scripts\python.exe -m defect_classifier.pooled_study_v1 real-run --processed-root "data\processed\eclipse_core" --mylyn-development "reports\experiments\eclipse_training_mylyn_pilot_20260719T174210.314891+0000\tables\development_split.csv" --audit-output "reports\data_audit\eclipse_pooled_v1\20260729T_pooled_v1_real_run" --model-output "reports\model_development\eclipse_pooled_v1\20260729T_pooled_v1_real_run" --protocol "docs\eclipse_pooled_protocol_v1.md" --protocol-commit "428af8aa61ef490d50c7a5e1eb1e47f38999cf4c"
```

Idempotent resume using the same directories:

```powershell
.\.venv\Scripts\python.exe -m defect_classifier.pooled_study_v1 real-run --processed-root "data\processed\eclipse_core" --mylyn-development "reports\experiments\eclipse_training_mylyn_pilot_20260719T174210.314891+0000\tables\development_split.csv" --audit-output "reports\data_audit\eclipse_pooled_v1\20260729T_pooled_v1_real_run" --model-output "reports\model_development\eclipse_pooled_v1\20260729T_pooled_v1_real_run" --protocol "docs\eclipse_pooled_protocol_v1.md" --protocol-commit "428af8aa61ef490d50c7a5e1eb1e47f38999cf4c" --resume
```

Post-run validation, still without locked-test access:

```powershell
.\.venv\Scripts\python.exe -m defect_classifier.pooled_study_v1 validate --audit-output "reports\data_audit\eclipse_pooled_v1\20260729T_pooled_v1_real_run" --model-output "reports\model_development\eclipse_pooled_v1\20260729T_pooled_v1_real_run" --protocol "docs\eclipse_pooled_protocol_v1.md" --protocol-commit "428af8aa61ef490d50c7a5e1eb1e47f38999cf4c"
```

There is intentionally no command in this preparation package that evaluates locked rows or
fits a final model on 100% of allowable data.
