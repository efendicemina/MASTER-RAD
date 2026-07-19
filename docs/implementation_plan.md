# Implementation Plan

## Phases

1. Create the reproducible Python package skeleton and configuration system.
2. Implement data loading, schema validation, cleaning, splitting, and inspection.
3. Implement leakage-aware TF-IDF pipelines and model factories.
4. Implement training, evaluation, error analysis, and reporting outputs.
5. Add tests, CI, documentation, and a smoke-experiment path.

## Assumptions

- The user will place a CSV file in `data/raw/`.
- The dataset resembles Eclipse Bugzilla reports but may vary in naming and formatting.
- The thesis uses the initial bug report text only.

## Risks

- Missing or malformed creation timestamps can block chronological splitting.
- Rare labels may be too sparse for both holdout partitions.
- Duplicate narratives may remain even after exact deduplication.

## Leakage Risks

- Post-report metadata such as status, resolution, comments, and history logs.
- Preprocessing fit on all rows instead of only training rows.
- Duplicate reports appearing in both train and test.

## Dataset-Dependent Decisions

- The exact target labels to keep or exclude.
- Whether grouped severity is meaningful for the provided dataset.
- Whether chronological splitting is feasible or a documented fallback must be enabled.

## Acceptance Criteria

- The package installs in editable mode.
- Ruff passes.
- Pytest passes with synthetic fixtures.
- The CLI supports inspect, prepare, train, evaluate, run-all, and predict.
- A tiny end-to-end smoke experiment completes on synthetic data.
- Real dataset outputs are only produced when the user supplies the dataset.
