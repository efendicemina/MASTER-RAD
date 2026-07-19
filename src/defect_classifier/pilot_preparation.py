"""Approved modeling-view construction for controlled Eclipse pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data_loading import load_raw_csv, resolve_columns
from .data_validation import normalize_label


@dataclass(slots=True)
class PilotPreparation:
    frame: pd.DataFrame
    filtering_summary: dict[str, int]
    duplicate_removals: pd.DataFrame
    conflicting_duplicate_labels: pd.DataFrame


def prepare_pilot_dataframe(config: dict[str, Any]) -> PilotPreparation:
    """Create the approved six-class view while retaining a full decision audit."""

    paths = config["dataset"]["resolved_paths"]
    if len(paths) != 1 or not paths[0].lower().endswith(".parquet"):
        raise ValueError("Controlled pilot requires exactly one processed Parquet input")
    raw = load_raw_csv(paths[0])
    frame, _ = resolve_columns(raw, config["source_columns"])
    raw_rows = len(frame)
    summary = frame[config["text_columns"]["summary"]].fillna("").astype(str)
    description = frame[config["text_columns"]["description"]].fillna("").astype(str)
    normalized_labels = frame[config["target_column"]].map(normalize_label)
    expected_labels = list(config["target_labels"])
    expected = set(expected_labels)
    exclusions = {normalize_label(value) for value in config.get("label_exclusions", [])}
    dates = pd.to_datetime(frame["creation_time"], errors="coerce", utc=True)
    valid_text = summary.str.strip().ne("") | description.str.strip().ne("")
    blank_labels = normalized_labels.eq("")
    excluded_labels = normalized_labels.isin(exclusions)
    unexpected_labels = ~blank_labels & ~excluded_labels & ~normalized_labels.isin(expected)
    invalid_dates = dates.isna()
    eligible = valid_text & ~blank_labels & ~excluded_labels & ~unexpected_labels & ~invalid_dates
    modeling = frame.loc[eligible].copy()
    modeling[config["target_column"]] = normalized_labels.loc[eligible]
    modeling["creation_time"] = dates.loc[eligible]
    modeling["_original_order"] = modeling.index
    if "exact_text_hash" not in modeling:
        raise ValueError("Processed pilot input must contain exact_text_hash")
    if "global_report_key" not in modeling:
        raise ValueError("Processed pilot input must contain global_report_key")
    conflicts = _conflicting_labels(modeling, config["target_column"])
    modeling = modeling.sort_values(
        ["creation_time", "global_report_key"], kind="mergesort"
    ).copy()
    duplicate_mask = modeling.duplicated("exact_text_hash", keep="first")
    kept_by_hash = modeling.loc[~duplicate_mask].set_index("exact_text_hash")
    removed = modeling.loc[duplicate_mask].copy()
    duplicate_removals = pd.DataFrame(
        {
            "removed_report_key": removed["global_report_key"],
            "retained_report_key": removed["exact_text_hash"].map(
                kept_by_hash["global_report_key"]
            ),
            "removed_creation_time": removed["creation_time"],
            "retained_creation_time": removed["exact_text_hash"].map(
                kept_by_hash["creation_time"]
            ),
            "removed_label": removed[config["target_column"]],
            "retained_label": removed["exact_text_hash"].map(
                kept_by_hash[config["target_column"]]
            ),
            "exact_text_hash": removed["exact_text_hash"],
            "decision": "removed_later_exact_duplicate",
        }
    )
    modeling = modeling.loc[~duplicate_mask].copy()
    modeling["duplicate_group"] = _connected_duplicate_groups(modeling)
    modeling = modeling.sort_values("creation_time", kind="mergesort").reset_index(drop=True)
    modeling = modeling.drop(columns=["_original_order"])
    filtering_summary = {
        "raw_core_rows": raw_rows,
        "missing_or_blank_label_rows": int(blank_labels.sum()),
        "configured_excluded_label_rows": int(excluded_labels.sum()),
        "unexpected_or_unmapped_label_rows": int(unexpected_labels.sum()),
        "invalid_creation_time_rows": int(invalid_dates.sum()),
        "invalid_text_rows": int((~valid_text).sum()),
        "eligible_before_exact_deduplication": int(eligible.sum()),
        "later_exact_duplicates_removed": int(duplicate_mask.sum()),
        "conflicting_exact_text_groups": int(len(conflicts)),
        "modeling_rows_after_filtering_and_deduplication": int(len(modeling)),
    }
    return PilotPreparation(
        frame=modeling,
        filtering_summary=filtering_summary,
        duplicate_removals=duplicate_removals.reset_index(drop=True),
        conflicting_duplicate_labels=conflicts,
    )


def _conflicting_labels(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    grouped = frame.groupby("exact_text_hash", sort=False)
    rows = []
    for text_hash, group in grouped:
        labels = sorted(group[target].unique())
        if len(labels) > 1:
            rows.append(
                {
                    "exact_text_hash": text_hash,
                    "labels": ",".join(labels),
                    "row_count": len(group),
                    "report_keys": ",".join(sorted(group["global_report_key"].astype(str))),
                    "earliest_creation_time": group["creation_time"].min(),
                    "latest_creation_time": group["creation_time"].max(),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "exact_text_hash",
            "labels",
            "row_count",
            "report_keys",
            "earliest_creation_time",
            "latest_creation_time",
        ],
    )


def _connected_duplicate_groups(frame: pd.DataFrame) -> pd.Series:
    keys = frame["global_report_key"].astype(str)
    parent = {key: key for key in keys}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    if "duplicate_group_id" in frame:
        for key, raw_target in zip(keys, frame["duplicate_group_id"], strict=True):
            if pd.notna(raw_target) and str(raw_target).strip():
                union(key, str(raw_target))
    return keys.map(find)
