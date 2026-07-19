from __future__ import annotations

import pandas as pd
import pytest

from defect_classifier.data_loading import resolve_columns
from defect_classifier.data_validation import (
    apply_grouped_labels,
    apply_label_filters,
    drop_exact_duplicate_reports,
    inspect_dataset,
)


def test_required_column_normalization(sample_frame):
    frame, resolved = resolve_columns(
        sample_frame,
        {
            "summary": ["summary"],
            "description": ["description"],
            "severity": ["severity"],
            "creation_time": ["creation time"],
        },
    )
    assert "summary" in frame.columns
    assert resolved["summary"] == "summary"


def test_missing_target_filtering(sample_frame):
    frame = sample_frame.copy()
    frame.loc[0, "Severity"] = ""
    filtered = apply_label_filters(frame.rename(columns=str.lower), "severity", ["enhancement"])
    assert "" not in filtered["severity"].tolist()


def test_duplicate_removal(sample_frame):
    frame = sample_frame.rename(columns=str.lower).copy()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    deduplicated, removed = drop_exact_duplicate_reports(frame, ["summary", "description"])
    assert removed > 0
    assert len(deduplicated) < len(frame)


def test_dataset_inspection(sample_frame):
    frame = sample_frame.rename(columns=lambda value: value.lower().replace(" ", "_"))
    inspection = inspect_dataset(
        frame,
        {
            "text_columns": {"summary": "summary", "description": "description"},
            "target_column": "severity",
            "label_exclusions": [],
            "label_grouping": {"enabled": False, "groups": {}},
        },
    )
    assert inspection.summary["total_rows"] == 20
    assert inspection.summary["empty_summary_count"] == 0


def test_grouped_labels_reject_unmapped_classes():
    frame = pd.DataFrame({"severity": ["major", "custom"]})
    with pytest.raises(ValueError, match="does not map"):
        apply_grouped_labels(frame, "severity", {"medium": ["major"]})
