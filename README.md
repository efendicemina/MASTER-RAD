# Automatic Classification of Software Defects Using Machine Learning

Research-grade Python project for a master's thesis on multiclass software defect severity classification from the initial bug report text only.

This repository is designed to be reproducible, leakage-aware, and maintainable. It provides:

- a Python package with CLI workflows
- YAML configuration files
- data validation and inspection
- pipeline-based preprocessing and modeling
- experiment tracking and reproducible outputs
- automated tests and CI

## Research objective

Predict the `Severity` class from the normalized concatenation of `Summary` and `Description` only.

Primary experiment: original multiclass severity classification.

Secondary experiment: grouped severity classification.

## Repository structure

- `.github/` CI and Copilot instructions
- `configs/` YAML experiment configs
- `data/` raw, interim, and processed data locations
- `docs/` protocol, leakage notes, and implementation plan
- `notebooks/` notebook workspace notes only
- `reports/` experiment outputs
- `src/defect_classifier/` Python package
- `tests/` automated tests

## Environment setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If execution policy blocks activation, run PowerShell as needed for your local policy or use the Python executable directly.

## Dataset placement

Place the manually provided CSV file in `data/raw/`, for example:

- `data/raw/sample_data.csv`

The project does not download data automatically and does not commit raw data.

## Expected columns

The code normalizes column names and resolves common variants, but the expected semantic fields are:

- report identifier
- summary
- description
- severity
- creation time
- product
- component
- duplicate link column such as `Dupe of`

See `docs/dataset.md` for details.

## Commands

Inspect data:

```powershell
python -m defect_classifier inspect-data --config configs/baseline.yaml
```

Audit large Eclipse CSV schemas without loading full files:

```powershell
python -m defect_classifier audit-schema --config configs/eclipse_full.yaml
```

Build reduced, atomic project Parquet files in bounded chunks:

```powershell
python -m defect_classifier build-dataset --config configs/eclipse_full.yaml --project MYLYN
```

Validate processed outputs and pre-training feasibility without fitting a model:

```powershell
python -m defect_classifier validate-project-output --config configs/eclipse_full.yaml --project MYLYN
python -m defect_classifier validate-training-readiness --config configs/eclipse_full.yaml
python -m defect_classifier finalize-pretraining --config configs/eclipse_full.yaml
```

The Eclipse ingestion and candidate training configurations keep `training.enabled: false`.
Researcher approval and human label-quality review are required before any full-data training.

The approved controlled MYLYN pilot is isolated in
`configs/eclipse_training_mylyn_pilot.yaml`. It uses only the processed MYLYN Parquet,
the six original severity labels, chronological holdout, duplicate-group-purged
expanding-window CV, and single-threaded model search. Do not change this configuration
into a pooled or cross-project run without a separate research decision.

Prepare data:

```powershell
python -m defect_classifier prepare-data --config configs/baseline.yaml
```

Train:

```powershell
python -m defect_classifier train --config configs/baseline.yaml
```

Evaluate:

```powershell
python -m defect_classifier evaluate --config configs/baseline.yaml
```

Held-out test evaluation is one-shot. If reporting is interrupted after predictions are
persisted, the evaluator resumes only the missing reports and does not repeat inference.

Run the controlled MYLYN development-only study (never the held-out test):

```powershell
python -m defect_classifier.development_study `
  --development reports\experiments\<approved_pilot_id>\tables\development_split.csv `
  --parquet data\processed\eclipse_core\MYLYN.parquet `
  --output reports\model_development\mylyn
```

The predeclared rule is in `docs/mylyn_model_selection_protocol.md`. The generated
challenger configuration remains disabled until a researcher explicitly authorizes any
new held-out evaluation.

Run the full pipeline:

```powershell
python -m defect_classifier run-all --config configs/baseline.yaml
```

Predict on a single report:

```powershell
python -m defect_classifier predict --model reports\experiments\<experiment_id>\artifacts\model.joblib --summary "Crash on startup" --description "Application fails when opening the main window."
```

## Outputs

Each experiment writes to its own timestamped directory under `reports/experiments/` and saves:

- configuration snapshot
- metadata
- inspection summaries
- fitted pipeline
- cross-validation results
- held-out test metrics
- predictions
- confusion matrices
- error analysis
- selected model features when supported

The checked-in sample dataset is intended only to validate that the workflow executes. Do not
use its scores as thesis findings; run the finalized protocol on the full dataset.

## Reproducibility

The experiment metadata records timestamp, commit hash when available, Python version, platform information, package versions, random seed, input hash, and class distributions.

## Testing

```powershell
python -m pip install -e .[dev]
ruff check .
pytest
```

CI runs the same checks without requiring the real dataset.

## Known limitations

- Real research findings are not included until the user runs the pipeline on the actual dataset.
- Duplicate handling is conservative and documented in `docs/data_leakage.md`.
- Chronological splitting depends on a valid creation-time column and well-formed timestamps.
