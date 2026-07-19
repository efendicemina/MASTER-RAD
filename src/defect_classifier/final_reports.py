"""Final cross-project pre-training summaries and human-review sample."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .data_validation import normalize_label
from .utils import ensure_directory

EMAIL_RE = re.compile(r"\S*@\S*")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _core_dir(config: dict[str, Any]) -> Path:
    core = Path(config.get("processed", {}).get("core_dir", "data/processed/eclipse_core"))
    return core if core.is_absolute() else Path(config["output"]["root_dir"]).parent / core


def _audit_dir(config: dict[str, Any]) -> Path:
    return ensure_directory(Path(config["output"]["root_dir"]) / "dataset_audit")


def create_final_reports(config: dict[str, Any]) -> dict[str, Path]:
    audit = _audit_dir(config)
    core = _core_dir(config)
    manifest = json.loads((core / "manifest.json").read_text(encoding="utf-8"))
    severity = pd.read_csv(audit / "severity_by_project.csv")
    text = pd.read_csv(audit / "text_missing_audit.csv")
    temporal = pd.read_csv(audit / "temporal_coverage.csv")
    holdout = pd.read_csv(audit / "proposed_holdout_distribution.csv")
    folds = pd.read_csv(audit / "proposed_cv_fold_distribution.csv")
    readiness = json.loads((audit / "training_readiness.json").read_text(encoding="utf-8"))
    readiness_by_scope = {row["scope"]: row for row in readiness["projects"]}
    manifest_by_project = {row["source_project"]: row for row in manifest}
    projects = sorted(manifest_by_project)
    severity_labels = sorted(severity["normalized_severity"].dropna().unique())
    summary_rows: list[dict[str, Any]] = []
    for project in projects:
        manifest_row = manifest_by_project[project]
        project_severity = severity.loc[severity["scope"] == project]
        project_text = text.loc[text["scope"] == project].iloc[0]
        project_temporal = temporal.loc[temporal["scope"] == project].iloc[0]
        project_holdout = holdout.loc[holdout["scope"] == project]
        project_folds = folds.loc[folds["scope"] == project]
        row: dict[str, Any] = {
            "project": project,
            "raw_rows": manifest_row["raw_row_count"],
            "modeling_eligible_rows": readiness_by_scope[project]["usable_rows"],
            "enhancement_rows": int(
                project_severity.loc[
                    project_severity["normalized_severity"] == "enhancement", "count"
                ].sum()
            ),
            "missing_text_rows": manifest_row["missing_text_count"],
            "invalid_date_rows": manifest_row["invalid_date_count"],
            "exact_duplicate_rows": manifest_row["exact_duplicate_row_count"],
            "dupe_of_relationships": _count_dupe_links(core / f"{project}.parquet"),
            "minimum_creation_time": project_temporal["minimum_creation_time"],
            "maximum_creation_time": project_temporal["maximum_creation_time"],
            "development_size": int(
                project_holdout.loc[project_holdout["partition"] == "development", "count"].sum()
            ),
            "test_size": int(
                project_holdout.loc[project_holdout["partition"] == "test", "count"].sum()
            ),
            "within_project_readiness": readiness_by_scope[project]["status"],
            "feature_building_risk": _feature_risk(
                int(readiness_by_scope[project]["usable_rows"]),
                float(project_text["combined_mean_chars"]),
            ),
        }
        for label in severity_labels:
            row[f"severity_{label}"] = int(
                project_severity.loc[
                    project_severity["normalized_severity"] == label, "count"
                ].sum()
            )
            row[f"test_{label}"] = int(
                project_holdout.loc[
                    (project_holdout["partition"] == "test")
                    & (project_holdout["severity"] == label),
                    "count",
                ].sum()
            )
        for fold in sorted(project_folds["fold"].unique()):
            validation = project_folds.loc[
                (project_folds["fold"] == fold) & (project_folds["partition"] == "validation")
            ]
            row[f"cv_fold_{int(fold)}_viable"] = set(validation["severity"]) >= set(
                readiness_by_scope[project]["class_counts"]
            )
        summary_rows.append(row)
    project_summary = pd.DataFrame(summary_rows)
    project_summary_path = audit / "project_summary.csv"
    project_summary.to_csv(project_summary_path, index=False)
    matrix = severity.pivot_table(
        index="scope", columns="normalized_severity", values="count", aggfunc="sum", fill_value=0
    ).reset_index()
    matrix = matrix.rename(columns={"scope": "project"})
    matrix_path = audit / "class_distribution_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    overlap_path = audit / "cross_project_overlap.csv"
    _cross_project_overlap(core, projects, audit).to_csv(overlap_path, index=False)
    review_path = audit / "label_quality_review_sample.csv"
    _label_review_sample(core, projects, config).to_csv(review_path, index=False)
    final_path = audit / "final_pretraining_summary.md"
    final_path.write_text(
        _final_markdown(project_summary, readiness, overlap_path), encoding="utf-8"
    )
    return {
        "project_summary": project_summary_path,
        "class_distribution_matrix": matrix_path,
        "cross_project_overlap": overlap_path,
        "label_quality_review_sample": review_path,
        "final_pretraining_summary": final_path,
    }


def _count_dupe_links(path: Path) -> int:
    import pyarrow.parquet as pq

    count = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["dupe_of"], batch_size=50000):
        values = batch.to_pandas()["dupe_of"].fillna("").astype(str).str.strip()
        count += int(values.ne("").sum())
    return count


def _feature_risk(rows: int, mean_chars: float) -> str:
    characters = rows * mean_chars
    return (
        "high" if characters > 1_000_000_000 else "moderate" if characters > 100_000_000 else "low"
    )


def _cross_project_overlap(core: Path, projects: list[str], audit: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    issue_projects: dict[str, set[str]] = defaultdict(set)
    hash_projects: dict[str, set[str]] = defaultdict(set)
    global_keys: set[str] = set()
    duplicate_global_keys = 0
    qualified_dupe_links = 0
    resolved_qualified_links = 0
    for project in projects:
        parquet = pq.ParquetFile(core / f"{project}.parquet")
        for batch in parquet.iter_batches(
            columns=["issue_id", "global_report_key", "exact_text_hash", "dupe_of"],
            batch_size=50000,
        ):
            frame = batch.to_pandas()
            for issue_id in frame["issue_id"].astype(str):
                issue_projects[issue_id].add(project)
            for text_hash in frame["exact_text_hash"].astype(str):
                hash_projects[text_hash].add(project)
            for key in frame["global_report_key"].astype(str):
                if key in global_keys:
                    duplicate_global_keys += 1
                global_keys.add(key)
            for value in frame["dupe_of"].fillna("").astype(str).str.strip():
                if ":" in value:
                    qualified_dupe_links += 1
                    resolved_qualified_links += int(value in global_keys)
    rows = [
        {
            "metric": "numeric_issue_ids_present_in_multiple_projects",
            "count": sum(len(values) > 1 for values in issue_projects.values()),
            "interpretation": (
                "expected numeric namespace collisions; identities remain project-qualified"
            ),
        },
        {
            "metric": "duplicate_global_report_keys",
            "count": duplicate_global_keys,
            "interpretation": "must remain zero",
        },
        {
            "metric": "exact_text_hashes_across_projects",
            "count": sum(len(values) > 1 for values in hash_projects.values()),
            "interpretation": "cross-project exact narrative overlap requiring grouped splitting",
        },
        {
            "metric": "explicit_cross_project_dupe_links",
            "count": qualified_dupe_links,
            "interpretation": f"{resolved_qualified_links} resolved against known global keys",
        },
    ]
    duplicate_summary = pd.read_csv(audit / "duplicate_summary.csv")
    for metric, count in zip(duplicate_summary["metric"], duplicate_summary["count"], strict=True):
        rows.append({"metric": f"audit_{metric}", "count": int(count), "interpretation": ""})
    return pd.DataFrame(rows)


def _label_review_sample(
    core: Path, projects: list[str], config: dict[str, Any], per_stratum: int = 2
) -> pd.DataFrame:
    frames = []
    exclusions = {normalize_label(value) for value in config.get("label_exclusions", [])}
    for project in projects:
        frame = pd.read_parquet(
            core / f"{project}.parquet",
            columns=[
                "global_report_key",
                "source_project",
                "issue_id",
                "creation_time",
                "severity",
                "summary",
                "description",
                "has_valid_text",
            ],
        )
        frame["severity"] = frame["severity"].map(normalize_label)
        frame = frame.loc[
            frame["has_valid_text"]
            & frame["creation_time"].notna()
            & frame["severity"].ne("")
            & ~frame["severity"].isin(exclusions)
        ].copy()
        frame["time_period"] = pd.qcut(
            frame["creation_time"].rank(method="first"),
            3,
            labels=["early", "middle", "late"],
        )
        sample = (
            frame.sample(frac=1, random_state=42)
            .groupby(["source_project", "severity", "time_period"], observed=True)
            .head(per_stratum)
            .reset_index(drop=True)
        )
        frames.append(sample)
    result = pd.concat(frames, ignore_index=True)
    result["summary"] = result["summary"].fillna("").astype(str).map(_redact)
    result["description"] = (
        result["description"].fillna("").astype(str).str.slice(0, 1000).map(_redact)
    )
    result["review_status"] = "not_reviewed"
    result["reviewer_note"] = ""
    return result[
        [
            "global_report_key",
            "source_project",
            "issue_id",
            "creation_time",
            "severity",
            "summary",
            "description",
            "review_status",
            "reviewer_note",
        ]
    ]


def _redact(value: str) -> str:
    return URL_RE.sub("[URL]", EMAIL_RE.sub("[EMAIL]", value))


def _final_markdown(
    project_summary: pd.DataFrame, readiness: dict[str, Any], overlap_path: Path
) -> str:
    warnings = project_summary.loc[
        project_summary["within_project_readiness"] != "PASS", "project"
    ].tolist()
    return "\n".join(
        [
            "# Final Pre-training Summary",
            "",
            f"Projects processed: {len(project_summary)}",
            f"Raw/core rows: {int(project_summary['raw_rows'].sum()):,}",
            f"Modeling-eligible rows: {int(project_summary['modeling_eligible_rows'].sum()):,}",
            f"Overall automated readiness: **{readiness['overall_status']}**",
            f"Projects with warnings: {warnings or 'none'}",
            "",
            (
                "All results are pre-training feasibility evidence, not model performance "
                "or scientific proof."
            ),
            f"Cross-project overlap details: `{overlap_path.name}`.",
            "Human label review remains required before enabling any training configuration.",
            "",
        ]
    )
