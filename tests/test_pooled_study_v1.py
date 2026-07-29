from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defect_classifier import pooled_study_v1 as study


def _pooled_frame(rows_per_project: int = 30) -> pd.DataFrame:
    rows = []
    severities = study.TASK_LABELS["s6"]
    for project_index, project in enumerate(study.PROJECTS):
        for index in range(rows_per_project):
            duplicate = (
                "cross-boundary" if index in {0, rows_per_project - 1} else f"{project}-{index}"
            )
            rows.append(
                {
                    "source_project": project,
                    "issue_id": str(index),
                    "global_report_key": f"{project}:{index}",
                    "summary": f"summary {duplicate}",
                    "description": f"description {duplicate}",
                    "severity": severities[index % len(severities)],
                    "creation_time": pd.Timestamp("2020-01-01", tz="UTC")
                    + pd.Timedelta(days=index + project_index),
                    "dupe_of": "",
                    "duplicate_group_id": "",
                    "exact_text_hash": duplicate,
                    "row_key": study._row_key(project, index),
                }
            )
    return pd.DataFrame(rows)


def test_label_mappings_are_deterministic_and_frozen() -> None:
    labels = ["blocker", "critical", "major", "minor", "normal", "trivial"]
    assert study.map_labels(labels, "s6").tolist() == labels
    assert study.map_labels(labels, "s3").tolist() == [
        "HIGH",
        "HIGH",
        "MEDIUM",
        "LOW",
        "MEDIUM",
        "LOW",
    ]
    assert study.map_labels(labels, "s2").tolist() == [
        "HIGH_IMPACT",
        "HIGH_IMPACT",
        "HIGH_IMPACT",
        "LOWER_IMPACT",
        "LOWER_IMPACT",
        "LOWER_IMPACT",
    ]


def test_per_project_partition_is_chronological() -> None:
    split = study.partition_allowed_universe(_pooled_frame())
    for project in study.PROJECTS:
        development = split.development.loc[split.development.source_project.eq(project)]
        locked = split.locked_test.loc[split.locked_test.source_project.eq(project)]
        assert not development.empty and not locked.empty
        assert development.creation_time.max() <= locked.creation_time.min()


def test_duplicate_groups_never_cross_partitions() -> None:
    split = study.partition_allowed_universe(_pooled_frame())
    assert not set(split.development.duplicate_group) & set(split.locked_test.duplicate_group)
    assert split.purged_test_rows >= len(study.PROJECTS)


def test_development_and_locked_rows_have_zero_overlap() -> None:
    split = study.partition_allowed_universe(_pooled_frame())
    assert not set(split.development.row_key) & set(split.locked_test.row_key)


def test_project_weights_are_normalized_and_help_smaller_projects() -> None:
    frame = pd.DataFrame({"source_project": ["large"] * 9 + ["small"]})
    weights = study.project_class_weights(frame, "inverse_sqrt")
    assert weights.mean() == pytest.approx(1.0)
    assert weights[-1] > weights[0]
    assert study.project_class_weights(frame, "none").tolist() == [1.0] * 10


def test_staged_pruning_and_ranking_are_deterministic() -> None:
    base = {
        "status": "trained",
        "representation": "word",
        "model": "LinearSVC",
        "project_macro_f1": 0.4,
        "minimum_class_recall": 0.2,
        "balanced_accuracy": 0.4,
        "feature_count": 100,
        "runtime_seconds": 1.0,
    }
    rows = [
        {**base, "candidate_id": "b", "macro_f1": 0.50},
        {**base, "candidate_id": "a", "macro_f1": 0.50},
        {**base, "candidate_id": "pruned", "macro_f1": 0.46},
        {**base, "candidate_id": "failed", "macro_f1": 1.0, "status": "failed"},
    ]
    first = study.rank_candidates(rows)
    assert first == study.rank_candidates(rows)
    assert [row["candidate_id"] for row in first] == ["a", "b"]


def test_s2_threshold_feasibility_and_infeasibility_are_structured() -> None:
    feasible = study.select_s2_threshold(
        np.array([0.1, 0.2, 0.8, 0.9]), np.array([False, False, True, True])
    )
    assert feasible["precision"] >= 0.30
    with pytest.raises(study.S2ThresholdInfeasible) as error:
        study.select_s2_threshold(
            np.array([0.01, 0.02, 0.03, 0.04]), np.array([False, False, False, True])
        )
    assert error.value.diagnostics["precision_constraint"] == 0.30


def test_reserved_mylyn_test_path_is_rejected_before_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development_split.csv"):
        study.load_allowed_universe(tmp_path, tmp_path / "test_split.csv")


def test_resume_manifest_locks_scientific_provenance() -> None:
    expected = {field: "same" for field in study.RESUME_LOCK_FIELDS}
    study.validate_resume_manifest(expected.copy(), expected)
    for field in study.RESUME_LOCK_FIELDS:
        changed = expected.copy()
        changed[field] = "changed"
        with pytest.raises(RuntimeError, match=field):
            study.validate_resume_manifest(changed, expected)


def test_protocol_commit_requires_full_lowercase_sha(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        study._prepare(
            tmp_path,
            tmp_path / "development_split.csv",
            tmp_path / "audit",
            tmp_path / "protocol.md",
            "short",
            "plan",
            "command",
        )


def test_privacy_safe_output_rejects_raw_columns() -> None:
    study.assert_privacy_safe_columns(pd.DataFrame({"task": ["s6"], "macro_f1": [0.2]}))
    with pytest.raises(ValueError, match="description"):
        study.assert_privacy_safe_columns(pd.DataFrame({"description": ["private"]}))


def test_candidate_grid_and_checksum_are_deterministic() -> None:
    assert len(study.candidate_grid()) == 36
    assert study.candidate_checksum() == study.candidate_checksum()
    assert len({candidate.candidate_id for candidate in study.candidate_grid()}) == 36


def test_row_and_project_bootstrap_are_paired_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "truth": ["a", "b"] * len(study.PROJECTS),
            "baseline": ["a", "a"] * len(study.PROJECTS),
            "challenger": ["a", "b"] * len(study.PROJECTS),
            "project": np.repeat(study.PROJECTS, 2),
        }
    )
    first = study.paired_bootstrap(frame, ["a", "b"], n_resamples=30)
    assert first == study.paired_bootstrap(frame, ["a", "b"], n_resamples=30)
    assert first["row_mean_delta"] > 0
