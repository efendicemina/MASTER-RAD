from __future__ import annotations

import json

import pandas as pd
import pytest

from defect_classifier.cli import main


def test_cli_missing_dataset_errors(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
dataset:
  path: data/raw/missing.csv
output:
  root_dir: reports
  experiments_dir: reports/experiments
source_columns:
  summary: [summary]
  description: [description]
  severity: [severity]
target_column: severity
label_exclusions: []
label_grouping:
  enabled: false
  groups: {high: [blocker], medium: [major], low: [minor]}
text_columns:
  summary: summary
  description: description
cleaning:
  lowercase: true
  remove_html: true
  replace_urls: true
  replace_emails: true
  remove_code_blocks: false
  remove_stack_traces: false
  normalize_unicode: true
  normalize_whitespace: true
split:
  strategy: chronological
  train_fraction: 0.8
  random_state: 42
  allow_missing_dates_fallback: false
features:
  representation: summary_description
  analyzer: word
  ngram_range: [1, 1]
  min_df: 1
  max_df: 1.0
  max_features: 20
  sublinear_tf: true
  lowercase: false
models:
  enabled: [DummyClassifier]
  search_method: grid
  search_iterations: 1
  class_weight: balanced
  random_state: 42
  param_grids:
    DummyClassifier: {}
cv:
  strategy: auto
  n_splits: 2
  shuffle: true
  random_state: 42
primary_metric: macro_f1
bootstrap:
  n_resamples: 10
  confidence_level: 0.95
  random_state: 42
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_info:
        main(["inspect-data", "--config", str(config)])
    assert exc_info.value.code == 2


def test_inspect_data_writes_summary_and_tables(sample_config_path, capsys):
    exit_code = main(["inspect-data", "--config", str(sample_config_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Inspection summary" in captured.out
    assert "Resolved dataset path" in captured.out
    assert "Severity counts" in captured.out

    output_root = sample_config_path.parent / "reports"
    metrics_path = output_root / "metrics" / "data_inspection.json"
    severity_path = output_root / "tables" / "severity_distribution.csv"
    missing_path = output_root / "tables" / "missing_values.csv"
    filtering_path = output_root / "tables" / "filtering_summary.csv"

    assert metrics_path.exists()
    assert severity_path.exists()
    assert missing_path.exists()
    assert filtering_path.exists()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["total_rows"] > 0
    assert payload["severity_class_counts"]
    assert payload["missing_summary_count"] >= 0
    assert payload["missing_description_count"] >= 0
    assert payload["missing_severity_count"] >= 0
    assert payload["duplicate_count"] >= 0
    assert payload["proposed_retained_rows"] >= 0
    assert payload["proposed_excluded_rows"] >= 0

    severity_frame = pd.read_csv(severity_path)
    missing_frame = pd.read_csv(missing_path)
    filtering_frame = pd.read_csv(filtering_path)

    assert not severity_frame.empty
    assert not missing_frame.empty
    assert not filtering_frame.empty
    assert "count" in severity_frame.columns
    assert "missing_count" in missing_frame.columns


def test_end_to_end_smoke_training(sample_config_path):
    from defect_classifier.training import evaluate_experiment, train_experiment

    outcome = train_experiment(sample_config_path)
    assert outcome.model_path.exists()
    result = evaluate_experiment(outcome.experiment_dir, sample_config_path)
    assert "macro_f1" in result["metrics"]
    assert (outcome.experiment_dir / "tables" / "classification_report.csv").exists()
    assert (outcome.experiment_dir / "metrics" / "error_analysis.json").exists()
