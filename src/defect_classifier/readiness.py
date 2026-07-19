"""Pre-training feasibility checks that never fit a model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .dataset_audits import generate_dataset_audits
from .utils import ensure_directory, write_json


def validate_training_readiness(
    config: dict[str, Any], project: str | None = None
) -> dict[str, Any]:
    report_dir = ensure_directory(Path(config["output"]["root_dir"]) / "dataset_audit")
    generate_dataset_audits(config, project)
    severity = pd.read_csv(report_dir / "severity_by_project.csv")
    combined_severity = pd.read_csv(report_dir / "severity_combined.csv")
    severity = pd.concat([severity, combined_severity], ignore_index=True, sort=False)
    holdout = pd.read_csv(report_dir / "proposed_holdout_distribution.csv")
    folds = pd.read_csv(report_dir / "proposed_cv_fold_distribution.csv")
    temporal = pd.read_csv(report_dir / "temporal_coverage.csv")
    duplicates = pd.read_csv(report_dir / "duplicate_summary.csv")
    text_audit = pd.read_csv(report_dir / "text_missing_audit.csv")
    thresholds = config.get("readiness", {})
    rare_warning = int(thresholds.get("rare_class_warning", 100))
    minimum_class = int(thresholds.get("minimum_class_rows", 20))
    minimum_total = int(thresholds.get("minimum_total_rows", 500))
    excluded = set(config.get("label_exclusions", []))
    rows: list[dict[str, Any]] = []
    scopes = sorted(severity["scope"].unique())
    for scope in scopes:
        scope_severity = severity.loc[severity["scope"] == scope]
        usable = scope_severity.loc[
            (~scope_severity["is_excluded"]) & (~scope_severity["is_missing_or_blank"])
        ]
        counts = dict(zip(usable["normalized_severity"], usable["count"], strict=True))
        total = int(sum(counts.values()))
        rare = sorted(label for label, count in counts.items() if count < rare_warning)
        too_small = sorted(label for label, count in counts.items() if count < minimum_class)
        test_labels = set(
            holdout.loc[(holdout["scope"] == scope) & (holdout["partition"] == "test"), "severity"]
        )
        dev_labels = set(
            holdout.loc[
                (holdout["scope"] == scope) & (holdout["partition"] == "development"),
                "severity",
            ]
        )
        future_only = sorted(test_labels - dev_labels)
        invalid_dates = int(temporal.loc[temporal["scope"] == scope, "invalid_dates"].sum())
        validation_folds = folds.loc[
            (folds["scope"] == scope) & (folds["partition"] == "validation")
        ]
        fold_count = validation_folds["fold"].nunique()
        status = "PASS"
        reasons: list[str] = []
        if total < minimum_total or too_small or future_only or fold_count == 0:
            status = "FAIL"
        elif rare or invalid_dates:
            status = "WARNING"
        if total < minimum_total:
            reasons.append(f"usable rows {total} < {minimum_total}")
        if too_small:
            reasons.append(f"classes below minimum: {too_small}")
        if rare:
            reasons.append(f"rare classes: {rare}")
        if future_only:
            reasons.append(f"classes only in test period: {future_only}")
        if invalid_dates:
            reasons.append(f"invalid dates: {invalid_dates}")
        if not reasons:
            reasons.append("configured thresholds satisfied")
        grouped_counts = {
            group: int(sum(counts.get(label, 0) for label in labels if label not in excluded))
            for group, labels in config.get("label_grouping", {}).get("groups", {}).items()
        }
        grouped_feasible = bool(grouped_counts) and min(grouped_counts.values()) >= minimum_class
        rows.append(
            {
                "scope": scope,
                "status": status,
                "usable_rows": total,
                "class_counts": counts,
                "rare_classes": rare,
                "development_classes": sorted(dev_labels),
                "test_classes": sorted(test_labels),
                "future_only_classes": future_only,
                "cv_fold_count": int(fold_count),
                "invalid_dates": invalid_dates,
                "baseline_multiclass_feasible": status != "FAIL",
                "grouped_classification_feasible": grouped_feasible,
                "within_project_feasible": status != "FAIL",
                "reasons": reasons,
            }
        )
    project_rows = [row for row in rows if row["scope"] != "combined"]
    passing_projects = sum(row["within_project_feasible"] for row in project_rows)
    combined_row = next((row for row in rows if row["scope"] == "combined"), None)
    pooled_feasible = bool(combined_row and combined_row["status"] != "FAIL")
    cross_project_feasible = passing_projects >= 2
    duplicate_risk = int(duplicates["count"].sum()) if not duplicates.empty else 0
    duplicate_metrics = (
        dict(zip(duplicates["metric"], duplicates["count"], strict=True))
        if not duplicates.empty
        else {}
    )
    planned_experiments = {
        "within_project": "PASS" if passing_projects else "FAIL",
        "pooled": "PASS" if pooled_feasible else "FAIL",
        "cross_project": "PASS" if cross_project_feasible else "FAIL",
    }
    combined_text = text_audit.loc[text_audit["scope"] == "combined"]
    estimated_text_characters = (
        int(combined_text.iloc[0]["rows"] * combined_text.iloc[0]["combined_mean_chars"])
        if not combined_text.empty
        else 0
    )
    feature_risk = (
        "high"
        if estimated_text_characters > 1_000_000_000
        else "moderate"
        if estimated_text_characters > 100_000_000
        else "low"
    )
    payload = {
        "overall_status": (
            "PASS" if all(value == "PASS" for value in planned_experiments.values()) else "WARNING"
        ),
        "projects": rows,
        "pooled_evaluation_feasible": pooled_feasible,
        "cross_project_evaluation_feasible": cross_project_feasible,
        "planned_experiments": planned_experiments,
        "duplicate_risk_count": duplicate_risk,
        "duplicate_risk_metrics": duplicate_metrics,
        "thresholds": {
            "rare_class_warning": rare_warning,
            "minimum_class_rows": minimum_class,
            "minimum_total_rows": minimum_total,
        },
        "estimated_text_characters": estimated_text_characters,
        "feature_building_risk": feature_risk,
    }
    write_json(report_dir / "training_readiness.json", payload)
    flat = pd.DataFrame(rows)
    for column in flat.columns:
        if flat[column].map(lambda value: isinstance(value, (dict, list))).any():
            flat[column] = flat[column].map(str)
    flat.to_csv(report_dir / "training_readiness.csv", index=False)
    lines = ["# Training Readiness", "", f"Overall: **{payload['overall_status']}**", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['scope']}: {row['status']}",
                "",
                f"- Usable rows: {row['usable_rows']:,}",
                f"- Class counts: {row['class_counts']}",
                f"- Baseline multiclass feasible: {row['baseline_multiclass_feasible']}",
                f"- Grouped classification feasible: {row['grouped_classification_feasible']}",
                f"- Reasons: {'; '.join(row['reasons'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Combined plans",
            "",
            f"- Pooled evaluation feasible: {pooled_feasible}",
            f"- Cross-project evaluation feasible: {cross_project_feasible}",
            f"- Planned experiment statuses: {planned_experiments}",
            "",
        ]
    )
    (report_dir / "training_readiness.md").write_text("\n".join(lines), encoding="utf-8")
    return payload
