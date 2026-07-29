from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from defect_classifier import transfer_study_v2 as study


def test_tokenizer_preserves_compound_and_all_derived_parts() -> None:
    tokens = study.software_tokenize("NullPointerException.method_name")
    assert "nullpointerexception.method_name" in tokens
    assert {"null", "pointer", "exception", "method", "name"} <= set(tokens)


def test_tokenizer_has_no_arbitrary_truncation() -> None:
    assert len(study.software_tokenize(" ".join(f"token{i}" for i in range(700)))) >= 700


def test_cleaning_masks_sensitive_label_tokens() -> None:
    cleaned = study.software_clean("Critical severity in bug 123", mask_labels=True)
    assert "critical" not in cleaned.lower()
    assert cleaned.count("[SEVERITY_TERM]") == 2
    assert "[BUG_ID]" in cleaned


def test_duplicate_purge_is_training_only_and_catches_normalized() -> None:
    validation = pd.DataFrame({"summary": ["Same title"], "description": ["<b>body</b>"]})
    original = validation.copy(deep=True)
    training = pd.DataFrame(
        {
            "summary": ["Same title", "different"],
            "description": ["body", "safe"],
        }
    )
    kept, audit = study.purge_training_overlap(training, validation)
    pd.testing.assert_frame_equal(validation, original)
    assert kept.summary.tolist() == ["different"]
    assert audit["normalized_removed"] == 1
    assert audit["overlap_after"] == 0


def test_candidate_matrix_is_fully_expanded() -> None:
    matrix = study.candidate_matrix()
    assert len(matrix) > 15_000
    assert set(matrix.task) == {"s6", "s3", "s2"}
    assert set(matrix.representation) == {"R0", "R1", "R2"}
    assert {"LinearSVC", "LogisticRegression", "ComplementNB", "NBSVM"} <= set(matrix.model)
    assert set(matrix.loc[matrix.representation != "R0", "summary_weight"]) == {1.0, 2.0, 3.0}
    assert set(matrix.loc[matrix.source_weight.notna(), "source_weight"]) == {0.0, 0.1, 0.25, 0.5}
    assert set(matrix.loc[matrix.ensemble_weight.notna(), "ensemble_weight"]) == {0.25, 0.5, 0.75}
    assert not matrix.candidate_id.duplicated().any()


def test_all_frozen_model_parameters_are_accepted_by_estimators() -> None:
    for task in study.TASKS:
        for candidate in study.base_model_grid(task):
            model = study._make_model(candidate)
            assert getattr(model, "class_weight", None) in {None, "balanced"}


def test_nbsvm_ratio_uses_only_supplied_training_rows() -> None:
    matrix = sparse.csr_matrix([[1, 0], [0, 1]], dtype=float)
    ratio = study.nbsvm_ratio(matrix, np.array(["yes", "no"]), "yes")
    assert ratio[0] > 0 and ratio[1] < 0


def test_r0_r1_r2_dimensions_summary_weight_and_nonnegativity() -> None:
    frame = pd.DataFrame(
        {
            "summary": [f"NullPointer token{i % 3}" for i in range(18)],
            "description": [f"component method_name failure{i % 4}" for i in range(18)],
        }
    )
    r0 = study.build_representation("R0").fit_transform(frame)
    r1_one = study.build_representation("R1", 1.0).fit_transform(frame)
    r1_three = study.build_representation("R1", 3.0).fit_transform(frame)
    r2 = study.build_representation("R2", 1.0).fit_transform(frame)
    assert r0.shape[1] > 0
    assert r1_one.shape[1] > 0
    assert r2.shape[1] > r1_one.shape[1]
    assert r1_one.shape == r1_three.shape
    assert not np.allclose(r1_one.toarray(), r1_three.toarray())
    assert r2.min() >= 0


def test_combined_sample_weights_are_normalized_and_recency_decays() -> None:
    times = pd.Series(pd.to_datetime(["2023-01-01", "2019-01-01"], utc=True))
    weights = study.combine_sample_weights(
        2, times, pd.Timestamp("2024-01-01", tz="UTC"), 0.5, "half_life_2"
    )
    assert weights.mean() == pytest.approx(1.0)
    assert weights[2] > weights[3]
    assert weights[:2].tolist() == pytest.approx([weights[0], weights[0]])


def test_threshold_selection_is_deterministic_and_constrained() -> None:
    probabilities = np.array([0.1, 0.2, 0.6, 0.9])
    truth = np.array([False, False, True, True])
    first = study.select_s2_threshold(probabilities, truth)
    second = study.select_s2_threshold(probabilities, truth)
    assert first == second
    assert first["threshold"] == pytest.approx(0.5)
    assert first["precision"] >= 0.30
    assert 0.05 <= first["threshold"] <= 0.95


def test_infeasible_s2_candidate_is_recorded_while_feasible_candidate_continues(
    monkeypatch,
) -> None:
    severities = ["blocker", "critical", "major", "normal", "minor", "trivial"] * 24
    outer_train = pd.DataFrame(
        {
            "creation_time": pd.date_range(
                "2020-01-01", periods=len(severities), freq="D", tz="UTC"
            ),
            "severity": severities,
            "summary": [f"summary token{i % 11}" for i in range(len(severities))],
            "description": [f"component failure{i % 7}" for i in range(len(severities))],
        }
    )
    candidates = study._stage_a_candidates("s2", fixture=True)[:2]
    original = study.select_s2_threshold
    calls = 0

    def first_infeasible(probabilities, truth):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise study.S2ThresholdInfeasible(
                {
                    "reason_code": "no_s2_threshold_meets_precision_constraint",
                    "precision_constraint": 0.30,
                    "best_precision": 0.25,
                    "best_precision_threshold": 0.95,
                    "best_precision_recall": 0.10,
                    "predicted_positive_count": 4,
                    "positive_class_count": 40,
                    "negative_class_count": 80,
                    "evaluated_threshold_count": 181,
                }
            )
        return original(probabilities, truth)

    monkeypatch.setattr(study, "select_s2_threshold", first_infeasible)
    rows, oof = study._evaluate_stage_a_grid(outer_train, "s2", candidates)

    assert [row["status"] for row in rows] == ["infeasible", "trained"]
    assert rows[0]["infeasibility_reason"] == (
        "no_s2_threshold_meets_precision_constraint"
    )
    assert list(oof) == [candidates[1]["candidate_id"]]
    assert study._rank_stage(rows) == [rows[1]]


def test_all_s2_candidates_infeasible_has_clear_terminal_error() -> None:
    rows = [
        {"candidate_id": "s2-A-0000", "status": "infeasible"},
        {"candidate_id": "s2-A-0001", "status": "infeasible"},
    ]
    with pytest.raises(
        RuntimeError,
        match="All eligible S2 candidates are infeasible for outer fold 1 stage A",
    ):
        study._require_feasible_stage(rows, "s2", 1, "A")


def test_infeasible_threshold_reports_deterministic_diagnostics() -> None:
    probabilities = np.array([0.01, 0.02, 0.03, 0.04])
    truth = np.array([False, False, False, True])
    with pytest.raises(study.S2ThresholdInfeasible) as first:
        study.select_s2_threshold(probabilities, truth)
    with pytest.raises(study.S2ThresholdInfeasible) as second:
        study.select_s2_threshold(probabilities, truth)
    assert first.value.diagnostics == second.value.diagnostics
    assert first.value.diagnostics == {
        "reason_code": "no_s2_threshold_meets_precision_constraint",
        "precision_constraint": 0.30,
        "best_precision": 0.0,
        "best_precision_threshold": 0.95,
        "best_precision_recall": 0.0,
        "predicted_positive_count": 0,
        "positive_class_count": 1,
        "negative_class_count": 3,
        "evaluated_threshold_count": 181,
    }


def test_calibrator_exposes_probabilities() -> None:
    calibrator = study.fit_calibrator(np.array([-2, -1, 1, 2]), np.array([0, 0, 1, 1]))
    probabilities = calibrator.predict_proba(np.array([[-3], [3]]))[:, 1]
    assert probabilities[0] < probabilities[1]


def test_paired_bootstrap_alignment_and_determinism() -> None:
    truth = np.array(["a", "b", "a", "b"])
    baseline = np.array(["a", "a", "a", "b"])
    challenger = truth.copy()
    first = study.paired_bootstrap(truth, baseline, challenger, ["a", "b"], n=50)
    assert first == study.paired_bootstrap(truth, baseline, challenger, ["a", "b"], n=50)
    with pytest.raises(ValueError, match="not aligned"):
        study.paired_bootstrap(truth, baseline[:-1], challenger, ["a", "b"])


def test_resume_locks_all_scientific_inputs() -> None:
    expected = {
        key: "same"
        for key in (
            "development_sha256",
            "development_rows",
            "outer_fold_fingerprint",
            "source_hashes",
            "source_projects",
            "protocol_content_sha256",
            "candidate_matrix_checksum",
            "seed",
            "task_mappings",
            "preprocessing",
            "model_grid",
            "success_criteria",
            "cli_command",
            "packages",
            "completed_stages",
            "schema_version",
        )
    }
    study.validate_resume(expected.copy(), expected)
    for key in expected:
        existing = expected.copy()
        existing[key] = "changed"
        with pytest.raises(RuntimeError, match=key):
            study.validate_resume(existing, expected)


def test_resume_preserves_completed_checkpoints_after_s2_infeasibility(
    tmp_path: Path, monkeypatch
) -> None:
    audit = tmp_path / "audit"
    model = tmp_path / "model"
    audit.mkdir()
    model.mkdir()
    state = {
        "inner_rows": [],
        "selected_rows": [],
        "fold_rows": [],
        "oof_rows": [],
        "stage_tables": {stage: [] for stage in "ABCDE"},
        "completed": ["s3:1", "s6:1"],
    }
    state_path = audit / "engine_state.json"
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    before = state_path.read_bytes()
    target = pd.DataFrame(
        {
            "creation_time": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
            "severity": ["normal", "major"],
            "summary": ["a", "b"],
            "description": ["a", "b"],
        }
    )
    fold = study.FrozenFold(1, (0,), (1,))
    attempted = []
    monkeypatch.setattr(study, "freeze_temporal_folds", lambda _: (fold,))
    monkeypatch.setattr(study, "_baseline_reproduction", lambda *_args, **_kwargs: pd.DataFrame())

    def fail_s2(_outer_train, task, _candidates):
        attempted.append(task)
        raise RuntimeError("All eligible S2 candidates are infeasible for outer fold 1 stage A")

    monkeypatch.setattr(study, "_evaluate_stage_a_grid", fail_s2)
    with pytest.raises(RuntimeError, match="All eligible S2 candidates"):
        study.execute_production_study(
            target,
            pd.DataFrame(),
            audit,
            model,
            study.EngineOptions(fixture=True, minimum_available_memory_gb=0.0, resume=True),
        )

    assert attempted == ["s2"]
    assert state_path.read_bytes() == before
    assert not (audit / "checkpoint_s2_fold_1.json").exists()


def test_atomic_write_never_presents_partial_final(tmp_path: Path, monkeypatch) -> None:
    final = tmp_path / "result.json"
    monkeypatch.setattr(study.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError, match="boom"):
        study.atomic_json(final, {"complete": True})
    assert not final.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_source_set_allows_target_file_but_rejects_other_projects(tmp_path: Path) -> None:
    for project in (*study.SOURCE_PROJECTS, "MYLYN"):
        (tmp_path / f"{project}.parquet").touch()
    assert set(study.validate_source_set(tmp_path)) == set(study.SOURCE_PROJECTS)
    (tmp_path / "UNKNOWN.parquet").touch()
    with pytest.raises(ValueError, match="Unexpected"):
        study.validate_source_set(tmp_path)


def test_source_set_missing_project_fails_closed(tmp_path: Path) -> None:
    for project in study.SOURCE_PROJECTS[:-1]:
        (tmp_path / f"{project}.parquet").touch()
    with pytest.raises(ValueError, match="Missing"):
        study.validate_source_set(tmp_path)


def test_target_test_path_is_rejected_before_read(tmp_path: Path) -> None:
    path = tmp_path / "test_split.csv"
    path.write_text("not relevant", encoding="utf-8")
    with pytest.raises(ValueError, match="development_split"):
        study.load_target(path)


def test_target_hash_is_checked_before_csv_parse(tmp_path: Path) -> None:
    path = tmp_path / "development_split.csv"
    path.write_text("not approved", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        study.load_target(path)


def test_synthetic_stages_train_a_through_e_without_simulated_status(tmp_path: Path) -> None:
    output = tmp_path / "synthetic"
    result = study.run_synthetic_stages(output)
    assert [row["stage"] for row in result["stages"]] == list("ABCDE")
    assert all(row["status"] == "trained" for row in result["stages"])
    assert "simulated" not in json.dumps(result).lower()


def test_resource_gate_precedes_outputs(tmp_path: Path, monkeypatch) -> None:
    memory = type("Memory", (), {"available": 1})()
    monkeypatch.setattr(study.psutil, "virtual_memory", lambda: memory)
    with pytest.raises(RuntimeError, match="RUN_BLOCKED_LOW_MEMORY"):
        study.real_run(
            tmp_path / "development_split.csv",
            tmp_path,
            tmp_path / "audit",
            tmp_path / "models",
            tmp_path / "protocol.md",
            "test",
        )
    assert not (tmp_path / "audit").exists()


def test_baseline_references_are_exact() -> None:
    assert study.BASELINE_REFERENCES == {
        "s6": 0.22627654455227184,
        "s3": 0.3971869758657194,
        "s2": 0.5917757621918929,
    }


def test_production_fixture_runs_all_stages_without_synthetic_runner(
    tmp_path: Path, monkeypatch
) -> None:
    severities = ["blocker", "critical", "major", "normal", "minor", "trivial"] * 24
    target = pd.DataFrame(
        {
            "id": range(len(severities)),
            "creation_time": pd.date_range(
                "2020-01-01", periods=len(severities), freq="D", tz="UTC"
            ),
            "severity": severities,
            "summary": [f"summary {label} token{i % 9}" for i, label in enumerate(severities)],
            "description": [
                f"description component{i % 7} failure" for i in range(len(severities))
            ],
        }
    )
    source = target.iloc[:72].rename(columns={"id": "issue_id"}).copy()
    source["source_project"] = "BIRT"
    source["creation_time"] -= pd.Timedelta(days=1000)
    source["summary"] = "source " + source.summary
    fit_calls = []
    source_preparations = []
    original_factory = study._make_model
    original_source_prepare = study._prepare_source

    def counted_factory(candidate):
        model = original_factory(candidate)
        original_fit = model.fit

        def counted_fit(*args, **kwargs):
            fit_calls.append(candidate["model"])
            return original_fit(*args, **kwargs)

        model.fit = counted_fit
        return model

    monkeypatch.setattr(study, "_make_model", counted_factory)

    def counted_source_prepare(*args, **kwargs):
        prepared, audit = original_source_prepare(*args, **kwargs)
        source_preparations.append((prepared, audit))
        return prepared, audit

    monkeypatch.setattr(study, "_prepare_source", counted_source_prepare)
    monkeypatch.setattr(
        study,
        "run_synthetic_stages",
        lambda *_: (_ for _ in ()).throw(AssertionError("synthetic runner used")),
    )
    result = study.execute_production_study(
        target,
        source,
        tmp_path / "audit",
        tmp_path / "model",
        study.EngineOptions(fixture=True, minimum_available_memory_gb=0.0),
    )
    assert result["status"] == "PRODUCTION_ENGINE_EXECUTED"
    assert fit_calls
    assert source_preparations
    assert all(audit["overlap_after"] == 0 for _, audit in source_preparations)
    assert all(
        frame.empty or frame.creation_time.max() < pd.Timestamp("2020-01-01", tz="UTC")
        for frame, _ in source_preparations
    )
    inner = pd.read_csv(tmp_path / "model" / "inner_selection_results.csv")
    assert set(inner.stage) == set("ABCD")
    assert (inner.purge_overlap_after == 0).all()
    ensemble = pd.read_csv(tmp_path / "model" / "ensemble_comparison.csv")
    assert not ensemble.empty and set(ensemble.stage) == {"E"}
    folds = pd.read_csv(tmp_path / "model" / "fold_results.csv")
    assert set(folds.task) == {"s6", "s3", "s2"}
    assert (folds.outer_evaluations == 1).all()
    oof = pd.read_csv(tmp_path / "model" / "oof_predictions.csv")
    assert not oof.empty
    assert not {"summary", "description", "issue_id", "id"} & set(oof.columns)
    assert "simulated" not in " ".join(
        path.read_text() for path in (tmp_path / "model").glob("*.csv")
    )
    assert not pd.read_csv(tmp_path / "model" / "calibration_results.csv").empty
    assert not pd.read_csv(tmp_path / "model" / "threshold_results.csv").empty
    assert not pd.read_csv(tmp_path / "model" / "bootstrap_delta_results.csv").empty


def test_frozen_baseline_reproduces_to_one_e_minus_six() -> None:
    path = Path(
        "reports/experiments/"
        "eclipse_training_mylyn_pilot_20260719T174210.314891+0000/"
        "tables/development_split.csv"
    )
    target = study.load_target(path)
    result = study._baseline_reproduction(target, study.freeze_temporal_folds(target))
    assert (result.absolute_difference <= 1e-6).all()
    assert set(result.status) == {"REPRODUCED"}
