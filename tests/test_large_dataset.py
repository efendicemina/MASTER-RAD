from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from defect_classifier.config import load_config
from defect_classifier.dataset_audits import generate_dataset_audits
from defect_classifier.dataset_builder import (
    FORBIDDEN_COLUMNS,
    build_dataset,
    validate_parquet,
    validate_project_output,
)
from defect_classifier.dataset_schema import (
    audit_schema_file,
    derive_source_project,
    resolve_core_columns,
)
from defect_classifier.final_reports import create_final_reports
from defect_classifier.readiness import validate_training_readiness


def _write_source(path: Path, rows: int = 24, bom: bool = False) -> None:
    severities = ["normal", "major", "minor", "critical"] * (rows // 4)
    frame = pd.DataFrame(
        {
            "Issue URL": [f"https://bugs/{index}" for index in range(rows)],
            "ID": [f"{index:03d}" for index in range(rows)],
            "Product": [path.stem] * rows,
            "Component": ["Core"] * rows,
            "Summary": ["same" if index < 2 else f"Summary {index}" for index in range(rows)],
            "Description": [
                "duplicate" if index < 2 else f"Description {index}" for index in range(rows)
            ],
            "Severity": severities,
            "Creation time": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "Dupe of": ["", "000", *([""] * (rows - 2))],
            "Comments": ["private@example.com"] * rows,
            "Attachments": ["large"] * rows,
        }
    )
    frame.to_csv(path, index=False, encoding="utf-8-sig" if bom else "utf-8")


def _write_config(path: Path, dataset_yaml: str) -> Path:
    path.write_text(
        f"""
dataset:
{dataset_yaml}
  chunksize: 5
output:
  root_dir: reports
  experiments_dir: reports/experiments
processed:
  core_dir: data/processed/eclipse_core
label_exclusions: [enhancement]
split:
  strategy: chronological
  train_fraction: 0.8
  random_state: 42
  allow_missing_dates_fallback: false
cv:
  strategy: auto
  n_splits: 2
  shuffle: false
  random_state: 42
readiness:
  minimum_total_rows: 10
  minimum_class_rows: 2
  rare_class_warning: 5
""",
        encoding="utf-8",
    )
    return path


def test_single_list_and_glob_configurations(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    first = data / "A_dataset_issues.csv"
    second = data / "B_dataset_issues.csv"
    _write_source(first)
    _write_source(second)
    single = load_config(
        _write_config(tmp_path / "single.yaml", "  path: data/A_dataset_issues.csv\n")
    )
    listed = load_config(
        _write_config(
            tmp_path / "list.yaml",
            "  paths:\n    - data/A_dataset_issues.csv\n    - data/B_dataset_issues.csv\n",
        )
    )
    globbed = load_config(_write_config(tmp_path / "glob.yaml", "  glob: data/*_issues.csv\n"))
    assert len(single["dataset"]["resolved_paths"]) == 1
    assert len(listed["dataset"]["resolved_paths"]) == 2
    assert len(globbed["dataset"]["resolved_paths"]) == 2


def test_header_audit_and_bom(tmp_path):
    source = tmp_path / "BOM_dataset_issues.csv"
    _write_source(source, bom=True)
    audit = audit_schema_file(source)
    assert audit.audit_status == "PASS"
    assert audit.detected_encoding == "utf-8-sig"
    assert audit.canonical_column_mappings["issue_id"] == "ID"


def test_missing_and_ambiguous_columns_fail(tmp_path):
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"ID": [1], "Summary": ["x"]}).to_csv(missing, index=False)
    assert audit_schema_file(missing).audit_status == "FAIL"
    _, ambiguous = resolve_core_columns(["ID", "Issue ID", "Summary"])
    assert "issue_id" in ambiguous


def test_chunked_parquet_manifest_resume_force_and_privacy(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "Demo_dataset_issues.csv"
    _write_source(source)
    config = load_config(
        _write_config(tmp_path / "config.yaml", "  path: data/Demo_dataset_issues.csv\n")
    )
    first = build_dataset(config, chunksize=5)[0]
    output = Path(first["output_parquet_path"])
    validation = validate_parquet(output)
    assert validation["rows"] == 24
    assert not set(FORBIDDEN_COLUMNS) & set(validation["columns"])
    frame = pd.read_parquet(output)
    assert frame.loc[0, "issue_id"] == "000"
    assert str(frame["creation_time"].dtype).endswith("UTC]")
    assert frame.loc[1, "duplicate_group_id"] == "Demo:000"
    assert frame["exact_text_hash"].duplicated().any()
    validated_project = validate_project_output(
        output, "Demo", "Demo_dataset_issues.csv", expected_rows=24
    )
    assert validated_project["rows"] == 24
    manifest = output.parent / "manifest.json"
    assert manifest.exists()
    second = build_dataset(config, chunksize=5)[0]
    assert second["resume_status"] == "skipped_unchanged"
    partial = output.parent / ".Demo.parquet.tmp"
    partial.write_bytes(b"partial")
    rebuilt = build_dataset(config, force=True, chunksize=5)[0]
    assert rebuilt["resume_status"] == "rebuilt"
    assert not partial.exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))[0]["source_sha256"]


def test_audits_and_readiness_are_generated(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "Audit_dataset_issues.csv"
    _write_source(source)
    config = load_config(
        _write_config(tmp_path / "config.yaml", "  path: data/Audit_dataset_issues.csv\n")
    )
    build_dataset(config, chunksize=6)
    outputs = generate_dataset_audits(config)
    assert "severity_by_project" in outputs
    assert (tmp_path / "reports" / "dataset_audit" / "duplicate_groups.csv").exists()
    readiness = validate_training_readiness(config)
    assert readiness["projects"]
    assert (tmp_path / "reports" / "dataset_audit" / "training_readiness.md").exists()
    final_outputs = create_final_reports(config)
    assert final_outputs["project_summary"].exists()
    sample = pd.read_csv(final_outputs["label_quality_review_sample"])
    assert set(sample["review_status"]) == {"not_reviewed"}
    assert not sample["description"].str.contains("@", regex=False).any()


def test_source_project_derivation():
    assert derive_source_project("MYLYN_dataset_issues.csv") == "MYLYN"


def test_ambiguous_schema_stops_build(tmp_path):
    source = tmp_path / "Bad_dataset_issues.csv"
    pd.DataFrame(
        {
            "ID": ["1"],
            "Issue ID": ["1"],
            "Summary": ["x"],
            "Description": ["y"],
            "Severity": ["normal"],
            "Creation time": ["2020-01-01"],
        }
    ).to_csv(source, index=False)
    config = load_config(
        _write_config(tmp_path / "config.yaml", "  path: Bad_dataset_issues.csv\n")
    )
    with pytest.raises(ValueError, match="ambiguous"):
        build_dataset(config)
