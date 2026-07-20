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
