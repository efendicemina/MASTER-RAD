from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    severities = ["blocker", "minor", "major", "normal"] * 5
    summary_templates = [
        "Crash on startup",
        "UI glitch when resizing",
        "Memory leak in parser",
        "Incorrect tooltip",
    ]
    description_templates = [
        "Application crashes immediately after launch.",
        "The toolbar overlaps the title bar.",
        "Parser keeps allocating memory for each request.",
        "Tooltip text is clipped in the settings page.",
    ]
    summaries = [f"{summary_templates[index % 4]} #{index + 1}" for index in range(20)]
    descriptions = [
        f"{description_templates[index % 4]} Example {index + 1}." for index in range(20)
    ]
    return pd.DataFrame(
        {
            "ID": list(range(1, 21)),
            "Summary": summaries,
            "Description": descriptions,
            "Severity": severities,
            "Creation time": pd.date_range("2024-01-01", periods=20, freq="D"),
            "Product": ["Core"] * 20,
            "Component": ["UI", "UI", "Parser", "Search"] * 5,
        }
    )


@pytest.fixture()
def sample_config_path(tmp_path: Path, sample_frame: pd.DataFrame) -> Path:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / "sample_data.csv"
    sample_frame.to_csv(csv_path, index=False)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        """
dataset:
  path: data/raw/sample_data.csv
output:
  root_dir: reports
  experiments_dir: reports/experiments
source_columns:
  id: [id]
  summary: [summary]
  description: [description]
  severity: [severity]
  creation_time: [creation time]
  product: [product]
  component: [component]
target_column: severity
label_exclusions: []
label_grouping:
  enabled: false
  groups:
    high: [blocker, critical]
    medium: [major, normal]
    low: [minor, trivial]
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
  max_features: 200
  sublinear_tf: true
  lowercase: false
models:
  enabled: [DummyClassifier, MultinomialNB, LogisticRegression, LinearSVC]
  search_method: grid
  search_iterations: 2
  class_weight: balanced
  random_state: 42
  param_grids:
    DummyClassifier: {}
    MultinomialNB:
      clf__alpha: [1.0]
      tfidf__ngram_range: [[1, 1]]
    LogisticRegression:
      clf__C: [1.0]
      tfidf__ngram_range: [[1, 1]]
    LinearSVC:
      clf__C: [1.0]
      tfidf__ngram_range: [[1, 1]]
cv:
  strategy: auto
  n_splits: 2
  shuffle: true
  random_state: 42
primary_metric: macro_f1
bootstrap:
  n_resamples: 20
  confidence_level: 0.95
  random_state: 42
""",
        encoding="utf-8",
    )
    return config_path
