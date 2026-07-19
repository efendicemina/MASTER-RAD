from __future__ import annotations

from pathlib import Path

import pandas as pd

from defect_classifier.pilot_preparation import prepare_pilot_dataframe


def test_pilot_keeps_earliest_exact_duplicate_and_audits_conflict(tmp_path: Path):
    source = tmp_path / "MYLYN.parquet"
    pd.DataFrame(
        {
            "issue_id": ["2", "1", "3", "4"],
            "global_report_key": ["MYLYN:2", "MYLYN:1", "MYLYN:3", "MYLYN:4"],
            "summary": ["same", "same", "other", "excluded"],
            "description": ["text", "text", "body", "body"],
            "severity": ["critical", "major", "minor", "enhancement"],
            "creation_time": ["2020-02-01", "2020-01-01", "2020-03-01", "2020-04-01"],
            "exact_text_hash": ["hash", "hash", "other-hash", "excluded-hash"],
            "duplicate_group_id": [None, None, None, None],
        }
    ).to_parquet(source, index=False)
    config = {
        "dataset": {"resolved_paths": [str(source)]},
        "source_columns": {
            "id": ["issue_id"],
            "summary": ["summary"],
            "description": ["description"],
            "severity": ["severity"],
            "creation_time": ["creation_time"],
        },
        "text_columns": {"summary": "summary", "description": "description"},
        "target_column": "severity",
        "target_labels": ["blocker", "critical", "major", "normal", "minor", "trivial"],
        "label_exclusions": ["enhancement", ""],
    }

    result = prepare_pilot_dataframe(config)

    assert result.frame["global_report_key"].tolist() == ["MYLYN:1", "MYLYN:3"]
    assert result.duplicate_removals.loc[0, "removed_report_key"] == "MYLYN:2"
    assert result.duplicate_removals.loc[0, "retained_report_key"] == "MYLYN:1"
    assert result.conflicting_duplicate_labels.loc[0, "labels"] == "critical,major"
    assert result.filtering_summary["configured_excluded_label_rows"] == 1
