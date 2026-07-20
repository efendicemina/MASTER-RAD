# MYLYN Transfer + Drift + Threshold Study Protocol (development-only)

This document summarizes the development-only protocol for the MYLYN transfer/drift/threshold study.

Key immutable inputs and hashes

- Approved development_split.csv SHA-256: 8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5
- Expected development outer-fold fingerprint: 5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903
- Approved development row count: 7664

Study invariants

- This study is development-only. The held-out test set MUST NOT be accessed.
- Only the approved development_split.csv (by SHA-256) is accepted.
- The study writes all protocol outputs under an experiment-specific output directory.
- --dry-run must not fit models and must emit a minimal candidate_matrix.csv and audit manifests.
- --resume may be used only when an existing manifest matches dataset and fold fingerprints exactly.

CLI overview

python -m defect_classifier.transfer_study \
  --target-development <development_split.csv> \
  --processed-root data/processed/eclipse_core \
  --audit-output <new-path> \
  --model-output <new-path> \
  [--dry-run] [--resume]

Minimum dry-run outputs

- candidate_matrix.csv
- fold_identity.csv
- source_dataset_manifest.csv
- data_cutoff_proof.csv
- study_manifest.json
- VALIDATION_BUNDLE.md (marked DRY_RUN)

Important safety checks implemented in the code

- Reject any input path not named "development_split.csv".
- Reject any development split whose SHA-256 differs from the approved value.
- Verify the development row count equals the approved value.
- Compute and verify the outer-fold fingerprint matches the approved fingerprint.
- Abort if the processed-root appears to include the MYLYN project as a source for transfer.

Resume semantics

- Resume only permitted when existing study_manifest.json in the output agrees on:
  - dataset_sha256
  - folds_fingerprint
  - candidate_matrix checksum
  - random seed / manifest stable identifiers
- Resume will only attempt to finish incomplete outer folds; it will not change selection or pick new candidates.

Audit and reproducibility

All outputs include deterministic identifiers and enough metadata to allow external verification. A VALIDATION_BUNDLE.md is produced in dry-run mode and in full run mode; the dry-run bundle is explicitly labeled DRY_RUN.

Protocol freeze v1.1 (complete specifications)

The following items are frozen for protocol v1.1 and must not be changed without creating a new protocol version and repeating dry-run/protocol freeze commits.

- Tasks and label groupings:
  - S6: [blocker, critical, major, normal, minor, trivial]
  - S3: HIGH = blocker + critical; MEDIUM = major + normal; LOW = minor + trivial
  - S2: HIGH_IMPACT = blocker + critical + major; LOWER_IMPACT = normal + minor + trivial
  - Exclude: enhancement, task, feature, blank, missing, unknown, unmapped

- Immutable dataset and fold identifiers:
  - development_split.csv SHA-256: 8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5
  - outer-fold fingerprint: 5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903
  - development rows: 7664

- Representations (R0/R1/R2):
  - R0 (baseline): Summary + Description; P0 preprocessing; word TF-IDF; ngram_range=(1,2); min_df=2; max_df=0.98; max_features=50000; sublinear_tf=True; L2 normalization.
  - R1 (software-aware branches): separate Summary TF-IDF (max_features=20000) and Description TF-IDF (max_features=40000); word ngram_range=(1,2); min_df=2; max_df=0.98; sublinear_tf=True; Summary branch weights: 1.0, 2.0, 3.0.
  - R2: R1 plus char_wb branch on cleaned concatenated Summary+Description; ngram_range=(3,5); min_df=3; max_df=0.995; max_features=30000.

- Tokenizer requirements:
  - Preserve original technical token (exact original string)
  - Emit camelCase splits, snake_case splits, dotted.identifier splits as separate tokens
  - Neutralize URLs, emails, Bugzilla IDs, long hex as placeholders
  - Preserve negation, code, exception names, and technical terms
  - Tokenizer must be deterministic and not depend on labels

- Fixed model grid:
  - LinearSVC: C = [0.1, 0.5, 1.0]; class_weight = [None, 'balanced']; random_state=42
  - LogisticRegression: C = [0.25, 1.0, 4.0]; class_weight = [None, 'balanced']; max_iter=2000; solver=lbfgs; random_state=42
  - ComplementNB: alpha = [0.1, 0.5, 1.0]
  - NB-SVM (only S2): binary log-count ratio transform + LogisticRegression with C = [0.25, 1.0, 4.0]; class_weight=[None, 'balanced']; smoothing=1.0; random_state=42

- Selection stages (A-E):
  - Stage A: target-only selection evaluating R0,R1,R2 across the fixed model grid. Keep at most top two candidates per task.
  - Stage B: for top-two candidates evaluate source pooling with source weights [0.10, 0.25, 0.50]. Normalize sample weights so mean sample weight on training = 1.0.
  - Stage C: recency selection for winner of Stage B across modes: all-history, five-year sliding window, exponential half-life 2y, 4y, 6y (relative to validation start).
  - Stage D: domain representation: plain pooled features vs shared+MYLYN-specific+non-MYLYN-specific sparse augmentation.
  - Stage E: small score-level ensemble only if inner OOF improves by >=0.01; weights tested: [0.25/0.75, 0.50/0.50, 0.75/0.25]. All ensembles chosen only on inner OOF.

- Thresholding and calibration for S2:
  - Sigmoid calibrator fit only on inner OOF decision scores
  - Threshold grid: 0.05 to 0.95 step 0.005
  - Primary objective: maximize HIGH_IMPACT F2 with HIGH_IMPACT precision >= 0.30

- Selection metric and tie-break rules as specified in the original prompt (macro-F1 primary for S6/S3; S2 ranking rules described)

- Success criteria (exact thresholds for MATERIAL_IMPROVEMENT, PRACTICAL_STRENGTH, etc.) and paired bootstrap procedure (n=1000, seed=42) are frozen and must not be changed.

- Forbidden methods:
  - No SMOTE, no random oversampling, no synthetic text augmentation, no LLM-based augmentation, no resampling of validation/test folds.

- Outputs (complete schema): All files listed in the original prompt (study_manifest.json, protocol_snapshot.md, candidate_matrix.csv, candidate_status.csv, fold_identity.csv, inner_fold_identity.csv, source_dataset_manifest.csv, source_project_counts.csv, class_distributions.csv, label_quality_audit.csv, data_cutoff_proof.csv, duplicate_purge_results.csv, integrity_checks.csv, baseline_reproduction.csv, inner_selection_results.csv, selected_configurations.csv, task_summary.csv, fold_results.csv, per_class_results.csv, transfer_comparison.csv, recency_comparison.csv, domain_adaptation_comparison.csv, ensemble_comparison.csv, threshold_results.csv, calibration_results.csv, predicted_class_distributions.csv, control_results.csv, coverage_results.csv, bootstrap_delta_results.csv, runtime_summary.csv, model_size_summary.csv, protocol_deviations.md, final_recommendation.md, VALIDATION_BUNDLE.md, oof_predictions.csv (no raw text or original IDs)

- Candidate matrix: must contain the entire enumerated grid for models, parameters, representations, task eligibility, stage, source weight options, recency modes, and default selectable flag True/False.

Versioning

- This document is protocol v1.1. Any change to frozen elements above requires bumping the protocol version and repeating the protocol-freeze commit and dry-run.
