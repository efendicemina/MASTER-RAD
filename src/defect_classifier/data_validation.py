"""Validation, filtering, and inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_loading import coerce_datetime_column, ensure_text_columns


@dataclass(slots=True)
class InspectionArtifacts:
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise an error when a required column is missing."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def normalize_label(value: Any) -> str:
    """Normalize a label value for filtering and comparison."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def apply_label_filters(
    frame: pd.DataFrame, target_column: str, exclusions: list[str]
) -> pd.DataFrame:
    """Remove rows with missing or excluded labels."""

    result = frame.copy()
    result[target_column] = result[target_column].map(normalize_label)
    exclusion_set = {normalize_label(value) for value in exclusions}
    mask = result[target_column].ne("") & ~result[target_column].isin(exclusion_set)
    return result.loc[mask].copy()


def apply_grouped_labels(
    frame: pd.DataFrame, target_column: str, groups: dict[str, list[str]]
) -> pd.DataFrame:
    """Map labels into grouped severity buckets when enabled."""

    result = frame.copy()
    reverse_lookup = {
        normalize_label(label): group for group, labels in groups.items() for label in labels
    }
    normalized = result[target_column].map(normalize_label)
    unmapped = sorted(set(normalized) - set(reverse_lookup))
    if unmapped:
        raise ValueError(f"Grouped severity configuration does not map labels: {unmapped}")
    result[target_column] = normalized.map(reverse_lookup)
    return result


def drop_exact_duplicate_reports(
    frame: pd.DataFrame, text_columns: list[str]
) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicates based on text columns and return the count removed."""

    before = len(frame)
    deduplicated = frame.drop_duplicates(subset=text_columns, keep="first").copy()
    return deduplicated, before - len(deduplicated)


def inspect_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> InspectionArtifacts:
    """Build a structured inspection summary and tables."""

    summary_column = config["text_columns"]["summary"]
    description_column = config["text_columns"]["description"]
    target_column = config["target_column"]
    required_columns = [
        column
        for column in [summary_column, description_column, target_column, "creation_time"]
        if column in frame.columns
    ]
    require_columns(frame, required_columns)

    frame = ensure_text_columns(frame, [summary_column, description_column])
    missing_counts = frame.isna().sum().to_dict()
    empty_summary = int(frame[summary_column].str.strip().eq("").sum())
    empty_description = int(frame[description_column].str.strip().eq("").sum())
    duplicate_reports = int(frame.duplicated(subset=[summary_column, description_column]).sum())

    missing_summary_count = int(
        frame[summary_column].isna().sum()
        + frame[summary_column].astype(str).str.strip().eq("").sum()
    )
    missing_description_count = int(
        frame[description_column].isna().sum()
        + frame[description_column].astype(str).str.strip().eq("").sum()
    )
    missing_severity_count = int(
        frame[target_column].isna().sum()
        + frame[target_column].astype(str).str.strip().eq("").sum()
    )

    severity_counts = (
        frame[target_column].map(normalize_label).value_counts(dropna=False).sort_index()
    )
    severity_frame = severity_counts.rename_axis(target_column).reset_index(name="count")
    severity_frame["percentage"] = severity_frame["count"] / max(len(frame), 1)

    proposed_frame = filter_dataset(frame, config)
    proposed_retained_rows = int(len(proposed_frame))
    proposed_excluded_rows = int(len(frame) - proposed_retained_rows)

    product_frame = pd.DataFrame(columns=["value", "count"])
    if "product" in frame.columns:
        product_frame = (
            frame["product"]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
            .rename_axis("value")
            .reset_index(name="count")
        )

    component_frame = pd.DataFrame(columns=["value", "count"])
    if "component" in frame.columns:
        component_frame = (
            frame["component"]
            .fillna("<missing>")
            .astype(str)
            .value_counts()
            .rename_axis("value")
            .reset_index(name="count")
        )

    creation_range = None
    if "creation_time" in frame.columns:
        parsed = pd.to_datetime(frame["creation_time"], errors="coerce", utc=True)
        creation_range = {
            "earliest": None if parsed.dropna().empty else parsed.min().isoformat(),
            "latest": None if parsed.dropna().empty else parsed.max().isoformat(),
        }

    summary = {
        "resolved_dataset_path": str(config.get("dataset", {}).get("path", "")),
        "total_rows": int(len(frame)),
        "detected_columns": list(frame.columns),
        "source_columns": list(frame.columns),
        "missing_value_counts": {key: int(value) for key, value in missing_counts.items()},
        "severity_class_counts": severity_frame.to_dict(orient="records"),
        "severity_class_percentages": severity_frame[[target_column, "percentage"]].to_dict(
            orient="records"
        ),
        "missing_summary_count": missing_summary_count,
        "missing_description_count": missing_description_count,
        "missing_severity_count": missing_severity_count,
        "empty_summary_count": empty_summary,
        "empty_description_count": empty_description,
        "duplicate_count": duplicate_reports,
        "creation_date_range": creation_range,
        "proposed_retained_rows": proposed_retained_rows,
        "proposed_excluded_rows": proposed_excluded_rows,
        "proposed_filtering_summary": {
            "label_exclusions": config.get("label_exclusions", []),
            "grouped_labels_enabled": bool(config.get("label_grouping", {}).get("enabled", False)),
            "drop_exact_duplicates": True,
            "proposed_retained_rows": proposed_retained_rows,
            "proposed_excluded_rows": proposed_excluded_rows,
        },
    }

    tables: dict[str, pd.DataFrame] = {
        "severity_counts": severity_frame,
        "missing_values": pd.DataFrame(
            {
                "column": list(missing_counts.keys()),
                "missing_count": list(map(int, missing_counts.values())),
            }
        ),
        "filtering_summary": pd.DataFrame(
            [
                {"metric": "total_rows", "value": int(len(frame))},
                {"metric": "proposed_retained_rows", "value": proposed_retained_rows},
                {"metric": "proposed_excluded_rows", "value": proposed_excluded_rows},
                {
                    "metric": "label_exclusions",
                    "value": ", ".join(str(value) for value in config.get("label_exclusions", [])),
                },
                {
                    "metric": "grouped_labels_enabled",
                    "value": int(bool(config.get("label_grouping", {}).get("enabled", False))),
                },
            ]
        ),
    }
    if not product_frame.empty:
        tables["product_counts"] = product_frame
    if not component_frame.empty:
        tables["component_counts"] = component_frame
    return InspectionArtifacts(summary=summary, tables=tables)


def filter_dataset(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply documented filtering decisions to the dataset."""

    result = ensure_text_columns(
        frame.copy(), [config["text_columns"]["summary"], config["text_columns"]["description"]]
    )
    result = apply_label_filters(
        result, config["target_column"], config.get("label_exclusions", [])
    )
    if config.get("label_grouping", {}).get("enabled", False):
        result = apply_grouped_labels(
            result, config["target_column"], config["label_grouping"]["groups"]
        )
    result = result.dropna(
        subset=[config["text_columns"]["summary"], config["text_columns"]["description"]]
    ).copy()
    return result


def add_duplicate_group_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a conservative duplicate group column when an ID or duplicate link exists."""

    result = frame.copy()
    if "id" in result.columns:
        fallback = pd.Series(result.index.astype(str), index=result.index)
        result["duplicate_group"] = (
            result["id"].astype("object").where(result["id"].notna(), fallback)
        )
        result["duplicate_group"] = result["duplicate_group"].astype(str)
    else:
        result["duplicate_group"] = result.index.astype(str)
    if "dupe_of" in result.columns:
        linked = result["dupe_of"].fillna("").astype(str).str.strip()
        result.loc[linked.ne(""), "duplicate_group"] = linked.loc[linked.ne("")]
    return result


def coerce_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse creation time when available."""

    if "creation_time" not in frame.columns:
        return frame.copy()
    return coerce_datetime_column(frame, "creation_time")
