"""Streaming audit reports for reduced Eclipse project datasets."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_validation import normalize_label
from .dataset_builder import CORE_COLUMNS, FORBIDDEN_COLUMNS, validate_parquet
from .dataset_schema import read_csv_header, resolve_core_columns
from .utils import ensure_directory, write_json

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
HTML_RE = re.compile(r"<[^>]+>")
CODE_RE = re.compile(r"```|Traceback \(|Exception in thread|^\s*at\s+\S+", re.MULTILINE)


def _processed_paths(config: dict[str, Any], project: str | None = None) -> list[Path]:
    core = Path(config.get("processed", {}).get("core_dir", "data/processed/eclipse_core"))
    if not core.is_absolute():
        core = Path(config["output"]["root_dir"]).parent / core
    paths = sorted(core.glob("*.parquet"))
    if project:
        paths = [path for path in paths if path.stem.lower() == project.lower()]
    if not paths:
        raise FileNotFoundError(f"No processed Parquet files found in {core}")
    return paths


def _length_stats(values: pd.Series) -> dict[str, float | int]:
    numeric = values.astype("int64")
    return {
        "minimum": int(numeric.min()) if len(numeric) else 0,
        "maximum": int(numeric.max()) if len(numeric) else 0,
        "mean": float(numeric.mean()) if len(numeric) else 0.0,
    }


def generate_dataset_audits(config: dict[str, Any], project: str | None = None) -> dict[str, Path]:
    import pyarrow.parquet as pq

    report_dir = ensure_directory(Path(config["output"]["root_dir"]) / "dataset_audit")
    exclusions = {normalize_label(value) for value in config.get("label_exclusions", [])}
    groups = config.get("label_grouping", {}).get("groups", {})
    reverse_groups = {
        normalize_label(label): group for group, labels in groups.items() for label in labels
    }
    severity_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    year_counter: Counter[tuple[str, int, str]] = Counter()
    hash_projects: dict[str, Counter[str]] = defaultdict(Counter)
    preferred_exact_report: dict[str, tuple[pd.Timestamp | None, str]] = {}
    global_keys: Counter[str] = Counter()
    issue_ids: Counter[tuple[str, str]] = Counter()
    duplicate_links: Counter[tuple[str, str]] = Counter()
    holdout_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    combined_severity: Counter[str] = Counter()
    grouped_counter: Counter[tuple[str, str]] = Counter()
    combined_eligible_frames: list[pd.DataFrame] = []

    for path in _processed_paths(config, project):
        validate_parquet(path)
        source_project = path.stem
        project_severity: Counter[str] = Counter()
        total = 0
        missing_summary = missing_description = both_missing = 0
        url_count = email_count = html_count = code_count = 0
        summary_lengths: list[int] = []
        description_lengths: list[int] = []
        combined_lengths: list[int] = []
        summary_tokens: list[int] = []
        description_tokens: list[int] = []
        dates: list[pd.Series] = []
        labels: list[pd.Series] = []
        invalid_dates = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=50000):
            frame = batch.to_pandas()
            total += len(frame)
            normalized = frame["severity"].map(normalize_label)
            project_severity.update(normalized)
            combined_severity.update(normalized)
            for value in normalized:
                grouped_counter[(source_project, reverse_groups.get(value, "<unmapped>"))] += 1
            summary = frame["summary"].fillna("").astype(str)
            description = frame["description"].fillna("").astype(str)
            summary_empty = summary.str.strip().eq("")
            description_empty = description.str.strip().eq("")
            missing_summary += int(summary_empty.sum())
            missing_description += int(description_empty.sum())
            both_missing += int((summary_empty & description_empty).sum())
            summary_lengths.extend(summary.str.len().tolist())
            description_lengths.extend(description.str.len().tolist())
            combined_lengths.extend((summary.str.len() + description.str.len()).tolist())
            summary_tokens.extend(summary.str.split().str.len().tolist())
            description_tokens.extend(description.str.split().str.len().tolist())
            combined_text = summary.str.cat(description, sep="\n")
            url_count += int(combined_text.str.contains(URL_RE, na=False).sum())
            email_count += int(combined_text.str.contains(EMAIL_RE, na=False).sum())
            html_count += int(combined_text.str.contains(HTML_RE, na=False).sum())
            code_count += int(combined_text.str.contains(CODE_RE, na=False).sum())
            parsed = pd.to_datetime(frame["creation_time"], errors="coerce", utc=True)
            invalid_dates += int(parsed.isna().sum())
            valid = parsed.notna()
            dates.append(parsed[valid])
            labels.append(normalized[valid])
            for year, label in zip(parsed[valid].dt.year, normalized[valid], strict=True):
                year_counter[(source_project, int(year), label)] += 1
            for key in frame["global_report_key"].astype(str):
                global_keys[key] += 1
            for issue_id in frame["issue_id"].astype(str):
                issue_ids[(source_project, issue_id)] += 1
            for text_hash in frame["exact_text_hash"].astype(str):
                hash_projects[text_hash][source_project] += 1
            for text_hash, key, timestamp in zip(
                frame["exact_text_hash"].astype(str),
                frame["global_report_key"].astype(str),
                parsed,
                strict=True,
            ):
                valid_timestamp = None if pd.isna(timestamp) else timestamp
                current = preferred_exact_report.get(text_hash)
                if current is None or (
                    valid_timestamp is not None
                    and (current[0] is None or valid_timestamp < current[0])
                ):
                    preferred_exact_report[text_hash] = (valid_timestamp, key)
            linked = frame["dupe_of"].fillna("").astype(str).str.strip()
            for target in linked[linked.ne("")]:
                duplicate_links[(source_project, target)] += 1

        for label, count in sorted(project_severity.items()):
            severity_rows.append(
                {
                    "scope": source_project,
                    "raw_severity": label,
                    "normalized_severity": label,
                    "count": count,
                    "percentage": count / max(total, 1),
                    "is_missing_or_blank": label == "",
                    "is_excluded": label in exclusions,
                    "is_expected": label in reverse_groups,
                    "proposed_retained_count": count if label and label not in exclusions else 0,
                }
            )
        summary_stats = _length_stats(pd.Series(summary_lengths, dtype="int64"))
        description_stats = _length_stats(pd.Series(description_lengths, dtype="int64"))
        combined_stats = _length_stats(pd.Series(combined_lengths, dtype="int64"))
        text_rows.append(
            {
                "scope": source_project,
                "rows": total,
                "missing_or_empty_summary": missing_summary,
                "missing_or_empty_description": missing_description,
                "both_missing": both_missing,
                "summary_min_chars": summary_stats["minimum"],
                "summary_mean_chars": summary_stats["mean"],
                "summary_max_chars": summary_stats["maximum"],
                "description_min_chars": description_stats["minimum"],
                "description_mean_chars": description_stats["mean"],
                "description_max_chars": description_stats["maximum"],
                "combined_min_chars": combined_stats["minimum"],
                "combined_mean_chars": combined_stats["mean"],
                "combined_max_chars": combined_stats["maximum"],
                "summary_mean_tokens": float(np.mean(summary_tokens)),
                "summary_max_tokens": max(summary_tokens, default=0),
                "description_mean_tokens": float(np.mean(description_tokens)),
                "description_max_tokens": max(description_tokens, default=0),
                "extremely_short_under_20_chars": sum(value < 20 for value in combined_lengths),
                "extremely_long_over_100000_chars": sum(
                    value > 100000 for value in combined_lengths
                ),
                "url_occurrence_rows": url_count,
                "email_occurrence_rows": email_count,
                "html_occurrence_rows": html_count,
                "code_or_stack_trace_rows": code_count,
            }
        )
        all_dates = (
            pd.concat(dates, ignore_index=True) if dates else pd.Series(dtype="datetime64[ns, UTC]")
        )
        all_labels = pd.concat(labels, ignore_index=True) if labels else pd.Series(dtype="string")
        temporal_rows.append(
            {
                "scope": source_project,
                "minimum_creation_time": all_dates.min(),
                "maximum_creation_time": all_dates.max(),
                "invalid_dates": invalid_dates,
                "valid_dates": len(all_dates),
            }
        )
        if len(all_dates):
            ordered = pd.DataFrame({"date": all_dates, "severity": all_labels}).sort_values("date")
            ordered = ordered.loc[
                ordered["severity"].ne("") & ~ordered["severity"].isin(exclusions)
            ].copy()
            combined_eligible_frames.append(ordered)
            cut = int(np.floor(len(ordered) * float(config["split"]["train_fraction"])))
            for partition, subset in (
                ("development", ordered.iloc[:cut]),
                ("test", ordered.iloc[cut:]),
            ):
                for label, count in subset["severity"].value_counts().items():
                    holdout_rows.append(
                        {
                            "scope": source_project,
                            "partition": partition,
                            "severity": label,
                            "count": count,
                        }
                    )
            boundaries = np.linspace(0, cut, int(config["cv"]["n_splits"]) + 2, dtype=int)
            for fold in range(int(config["cv"]["n_splits"])):
                train = ordered.iloc[: boundaries[fold + 1]]
                validation_fold = ordered.iloc[boundaries[fold + 1] : boundaries[fold + 2]]
                for partition, subset in (("train", train), ("validation", validation_fold)):
                    for label, count in subset["severity"].value_counts().items():
                        cv_rows.append(
                            {
                                "scope": source_project,
                                "fold": fold + 1,
                                "partition": partition,
                                "severity": label,
                                "count": count,
                            }
                        )

    if text_rows:
        total_rows = sum(row["rows"] for row in text_rows)
        combined_text_row: dict[str, Any] = {"scope": "combined", "rows": total_rows}
        sum_columns = [
            "missing_or_empty_summary",
            "missing_or_empty_description",
            "both_missing",
            "extremely_short_under_20_chars",
            "extremely_long_over_100000_chars",
            "url_occurrence_rows",
            "email_occurrence_rows",
            "html_occurrence_rows",
            "code_or_stack_trace_rows",
        ]
        for column in sum_columns:
            combined_text_row[column] = sum(row[column] for row in text_rows)
        for prefix in ["summary", "description", "combined"]:
            combined_text_row[f"{prefix}_min_chars"] = min(
                row[f"{prefix}_min_chars"] for row in text_rows
            )
            combined_text_row[f"{prefix}_max_chars"] = max(
                row[f"{prefix}_max_chars"] for row in text_rows
            )
            combined_text_row[f"{prefix}_mean_chars"] = sum(
                row[f"{prefix}_mean_chars"] * row["rows"] for row in text_rows
            ) / max(total_rows, 1)
        for prefix in ["summary", "description"]:
            combined_text_row[f"{prefix}_max_tokens"] = max(
                row[f"{prefix}_max_tokens"] for row in text_rows
            )
            combined_text_row[f"{prefix}_mean_tokens"] = sum(
                row[f"{prefix}_mean_tokens"] * row["rows"] for row in text_rows
            ) / max(total_rows, 1)
        text_rows.append(combined_text_row)

    if temporal_rows:
        temporal_rows.append(
            {
                "scope": "combined",
                "minimum_creation_time": min(row["minimum_creation_time"] for row in temporal_rows),
                "maximum_creation_time": max(row["maximum_creation_time"] for row in temporal_rows),
                "invalid_dates": sum(row["invalid_dates"] for row in temporal_rows),
                "valid_dates": sum(row["valid_dates"] for row in temporal_rows),
            }
        )
    for (scope, year, severity), count in list(year_counter.items()):
        if scope != "combined":
            year_counter[("combined", year, severity)] += count

    if combined_eligible_frames:
        combined_ordered = pd.concat(combined_eligible_frames, ignore_index=True).sort_values(
            "date"
        )
        combined_cut = int(
            np.floor(len(combined_ordered) * float(config["split"]["train_fraction"]))
        )
        for partition, subset in (
            ("development", combined_ordered.iloc[:combined_cut]),
            ("test", combined_ordered.iloc[combined_cut:]),
        ):
            for label, count in subset["severity"].value_counts().items():
                holdout_rows.append(
                    {
                        "scope": "combined",
                        "partition": partition,
                        "severity": label,
                        "count": count,
                    }
                )
        boundaries = np.linspace(0, combined_cut, int(config["cv"]["n_splits"]) + 2, dtype=int)
        for fold in range(int(config["cv"]["n_splits"])):
            train = combined_ordered.iloc[: boundaries[fold + 1]]
            validation_fold = combined_ordered.iloc[boundaries[fold + 1] : boundaries[fold + 2]]
            for partition, subset in (("train", train), ("validation", validation_fold)):
                for label, count in subset["severity"].value_counts().items():
                    cv_rows.append(
                        {
                            "scope": "combined",
                            "fold": fold + 1,
                            "partition": partition,
                            "severity": label,
                            "count": count,
                        }
                    )

    combined_total = sum(combined_severity.values())
    combined_rows = []
    for label, count in sorted(combined_severity.items()):
        grouped_counter[("combined", reverse_groups.get(label, "<unmapped>"))] += count
        combined_rows.append(
            {
                "scope": "combined",
                "normalized_severity": label,
                "count": count,
                "percentage": count / max(combined_total, 1),
                "is_missing_or_blank": label == "",
                "is_excluded": label in exclusions,
                "is_expected": label in reverse_groups,
                "proposed_retained_count": count if label and label not in exclusions else 0,
            }
        )
    severity_frame = pd.DataFrame(severity_rows)
    combined_frame = pd.DataFrame(combined_rows)
    severity_frame.to_csv(report_dir / "severity_by_project.csv", index=False)
    combined_frame.to_csv(report_dir / "severity_combined.csv", index=False)
    pd.DataFrame(
        [
            {"scope": scope, "grouped_severity": group, "count": count}
            for (scope, group), count in sorted(grouped_counter.items())
        ]
    ).to_csv(report_dir / "grouped_severity_preview.csv", index=False)
    pd.concat(
        [
            severity_frame.loc[~severity_frame["is_expected"]],
            combined_frame.loc[~combined_frame["is_expected"]],
        ],
        ignore_index=True,
    ).to_csv(report_dir / "unexpected_labels.csv", index=False)
    pd.DataFrame(text_rows).to_csv(report_dir / "text_missing_audit.csv", index=False)
    for row in temporal_rows:
        scope = row["scope"]
        dev_classes = set(
            entry["severity"]
            for entry in holdout_rows
            if entry["scope"] == scope and entry["partition"] == "development"
        )
        test_classes = set(
            entry["severity"]
            for entry in holdout_rows
            if entry["scope"] == scope and entry["partition"] == "test"
        )
        fold_numbers = set(
            entry["fold"]
            for entry in cv_rows
            if entry["scope"] == scope and entry["partition"] == "validation"
        )
        row["holdout_viable"] = bool(
            dev_classes and test_classes and not (test_classes - dev_classes)
        )
        row["cv_viable"] = len(fold_numbers) == int(config["cv"]["n_splits"])
        row["missing_test_classes"] = ",".join(sorted(dev_classes - test_classes))
        row["future_only_classes"] = ",".join(sorted(test_classes - dev_classes))
    pd.DataFrame(temporal_rows).to_csv(report_dir / "temporal_coverage.csv", index=False)
    yearly_rows: list[dict[str, Any]] = []
    observed_labels = sorted(combined_severity)
    scope_years: dict[str, set[int]] = defaultdict(set)
    for scope, year, _ in year_counter:
        scope_years[scope].add(year)
    for scope, years in scope_years.items():
        scope_total = sum(
            count for (counter_scope, _, _), count in year_counter.items() if counter_scope == scope
        )
        scope_label_totals = {
            label: sum(
                count
                for (counter_scope, _, counter_label), count in year_counter.items()
                if counter_scope == scope and counter_label == label
            )
            for label in observed_labels
        }
        for year in sorted(years):
            year_total = sum(year_counter.get((scope, year, label), 0) for label in observed_labels)
            for severity in observed_labels:
                count = year_counter.get((scope, year, severity), 0)
                year_share = count / max(year_total, 1)
                scope_share = scope_label_totals[severity] / max(scope_total, 1)
                yearly_rows.append(
                    {
                        "scope": scope,
                        "year": year,
                        "severity": severity,
                        "count": count,
                        "percentage_within_year": year_share,
                        "class_missing_in_period": count == 0,
                        "label_drift_delta_from_scope_share": year_share - scope_share,
                    }
                )
    pd.DataFrame(yearly_rows).to_csv(report_dir / "severity_by_year.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(report_dir / "proposed_holdout_distribution.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(report_dir / "proposed_cv_fold_distribution.csv", index=False)
    duplicate_groups = [
        {
            "group_type": "exact_text_hash",
            "group_id": text_hash,
            "exact_text_hash": text_hash,
            "row_count": sum(projects.values()),
            "project_count": len(projects),
            "projects": ",".join(sorted(projects)),
            "preferred_report_key": preferred_exact_report[text_hash][1],
            "retention_rule": "earliest_valid_creation_time",
        }
        for text_hash, projects in hash_projects.items()
        if sum(projects.values()) > 1
    ]
    duplicate_groups.extend(
        {
            "group_type": "dupe_of",
            "group_id": f"{source_project}:{target}",
            "exact_text_hash": "",
            "row_count": count + 1,
            "project_count": 1,
            "projects": source_project,
            "preferred_report_key": f"{source_project}:{target}",
            "retention_rule": "canonical_dupe_of_target",
        }
        for (source_project, target), count in sorted(duplicate_links.items())
    )
    pd.DataFrame(duplicate_groups).to_csv(report_dir / "duplicate_groups.csv", index=False)
    pd.DataFrame(
        [
            {
                "metric": "duplicate_project_issue_ids",
                "count": sum(count - 1 for count in issue_ids.values() if count > 1),
            },
            {
                "metric": "duplicate_global_report_keys",
                "count": sum(count - 1 for count in global_keys.values() if count > 1),
            },
            {
                "metric": "exact_duplicate_text_rows",
                "count": sum(
                    row["row_count"] - 1
                    for row in duplicate_groups
                    if row["group_type"] == "exact_text_hash"
                ),
            },
            {"metric": "linked_duplicate_rows", "count": sum(duplicate_links.values())},
            {
                "metric": "cross_project_exact_duplicate_groups",
                "count": sum(
                    row["project_count"] > 1
                    for row in duplicate_groups
                    if row["group_type"] == "exact_text_hash"
                ),
            },
        ]
    ).to_csv(report_dir / "duplicate_summary.csv", index=False)
    excluded_source_columns: set[str] = set()
    for source_path in config["dataset"]["resolved_paths"]:
        raw_columns, _ = read_csv_header(source_path)
        mappings, _ = resolve_core_columns(raw_columns)
        excluded_source_columns.update(set(raw_columns) - set(mappings.values()))
    write_json(
        report_dir / "retained_columns.json",
        {"processed_columns": CORE_COLUMNS, "model_feature_columns": ["summary", "description"]},
    )
    write_json(
        report_dir / "excluded_columns.json",
        {
            "forbidden_canonical_columns": FORBIDDEN_COLUMNS,
            "excluded_raw_source_columns": sorted(excluded_source_columns),
        },
    )
    outputs = {path.stem: path for path in report_dir.glob("*.*")}
    return outputs


def write_manifest_csv(config: dict[str, Any]) -> Path:
    core = Path(config.get("processed", {}).get("core_dir", "data/processed/eclipse_core"))
    if not core.is_absolute():
        core = Path(config["output"]["root_dir"]).parent / core
    manifest = json.loads((core / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(manifest)
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, dict)).any():
            frame[column] = frame[column].map(json.dumps)
    output = ensure_directory(Path(config["output"]["root_dir"]) / "dataset_audit")
    path = output / "dataset_manifest.csv"
    frame.to_csv(path, index=False)
    return path
