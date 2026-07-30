from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from defect_classifier.pooled_rescue_v2 import (
    FieldFeatureBuilder,
    HierarchicalClassifier,
    RareMetadataEncoder,
    RescueCandidate,
    _atomic_checkpoint,
    _load_checkpoint,
    _split_inner,
    candidate_checksum,
    composed_sample_weights,
    fast_paired_bootstrap,
    rank_stage_a,
    rescue_candidates,
    smoothed_class_weights,
)
from defect_classifier.pooled_study_v1 import paired_bootstrap


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "summary": ["crash now", "minor display", "normal issue", "crash again"] * 2,
            "description": ["stack failure", "color wrong", "ordinary bug", "fatal stack"] * 2,
            "source_project": ["A", "A", "B", "B"] * 2,
            "product": ["p1", "p1", "p2", "p2"] * 2,
            "component": ["c1", "c1", "c2", "c2"] * 2,
            "duplicate_group": ["g1", "g1", "g2", "g3"] * 2,
        }
    )


def test_candidate_set_is_bounded_and_deterministic() -> None:
    assert len(rescue_candidates()) == 8
    assert candidate_checksum() == candidate_checksum()
    assert len({row.candidate_id for row in rescue_candidates()}) == 8


def test_field_specific_feature_construction_and_normalization() -> None:
    candidate = RescueCandidate("T", "field_word", 0.5)
    builder = FieldFeatureBuilder(candidate, max_features=100)
    matrix = builder.fit_transform(frame())
    assert matrix.shape[0] == 8
    assert matrix.shape[1] > 2
    np.testing.assert_allclose(sparse.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)


def test_word_char_blocks_are_separate_and_normalized() -> None:
    candidate = RescueCandidate("T", "field_word_char", 0.5)
    matrix = FieldFeatureBuilder(candidate, max_features=200).fit_transform(frame())
    assert matrix.shape[1] > 4
    np.testing.assert_allclose(sparse.linalg.norm(matrix, axis=1), 1.0, atol=1e-6)


def test_safe_metadata_has_deterministic_unknown_categories() -> None:
    encoder = RareMetadataEncoder(minimum_count=2).fit(frame())
    unseen = frame().iloc[:1].copy()
    unseen["component"] = "never-seen"
    first = encoder.transform(unseen)
    second = encoder.transform(unseen)
    assert first.shape[1] == encoder.n_features_
    assert (first != second).nnz == 0
    assert first.nnz == 3


def test_smoothed_class_weights_follow_power_balance() -> None:
    labels = np.array(["majority"] * 6 + ["minority"] * 2)
    half = smoothed_class_weights(labels, 0.5)
    full = smoothed_class_weights(labels, 1.0)
    assert half[6] > half[0]
    assert full[6] / full[0] > half[6] / half[0]
    assert half.mean() == pytest.approx(1.0)


def test_weight_composition_is_normalized_without_double_application() -> None:
    data = frame()
    labels = np.array(["a"] * 6 + ["b"] * 2)
    candidate = RescueCandidate(
        "T", "field_word", 0.5, project_weighting=True, duplicate_weighting=True
    )
    weights = composed_sample_weights(data, labels, candidate)
    assert weights.mean() == pytest.approx(1.0)
    assert np.isfinite(weights).all()
    assert np.all(weights > 0)


def test_duplicate_weighting_reduces_large_component_influence() -> None:
    data = frame()
    labels = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    plain = RescueCandidate("P", "field_word", 0.0)
    duplicate = replace(plain, candidate_id="D", duplicate_weighting=True)
    plain_weights = composed_sample_weights(data, labels, plain)
    duplicate_weights = composed_sample_weights(data, labels, duplicate)
    assert duplicate_weights[data.duplicate_group == "g1"].mean() < plain_weights.mean()


def test_hierarchical_logic_predicts_only_known_classes() -> None:
    matrix = sparse.csr_matrix(np.eye(6, dtype=np.float32))
    labels = np.array(["MEDIUM", "MEDIUM", "HIGH", "HIGH", "LOW", "LOW"])
    model = HierarchicalClassifier("s3")
    model.fit(matrix, labels, np.ones(6))
    assert set(model.predict(matrix)) <= set(labels)


def test_deterministic_ranking_uses_candidate_id_as_final_tie_break() -> None:
    rows = [
        {
            "candidate_id": candidate,
            "status": "trained",
            "macro_f1": 0.5,
            "project_macro_f1": 0.4,
            "minimum_class_recall": 0.3,
            "balanced_accuracy": 0.5,
        }
        for candidate in ("R-02", "R-01", "R-03")
    ]
    assert [row["candidate_id"] for row in rank_stage_a(rows)] == ["R-01", "R-02"]


def test_no_post_submission_fields_are_encoded() -> None:
    forbidden = {"history", "status", "resolution", "comments", "assignee", "reporter"}
    assert forbidden.isdisjoint(RareMetadataEncoder.fields)


def test_locked_test_has_no_candidate_api() -> None:
    candidate_fields = set(RescueCandidate.__dataclass_fields__)
    assert "locked_test" not in candidate_fields
    assert "test" not in candidate_fields


def test_inner_calibration_is_chronological_and_training_only() -> None:
    pieces = []
    for project in ("BIRT", "CDT", "Equinox", "JDT", "MYLYN", "PDE", "Papyrus", "Platform", "TPTP"):
        part = frame().iloc[:5].copy()
        part["source_project"] = project
        part["creation_time"] = pd.date_range("2020-01-01", periods=5, tz="UTC")
        part["row_key"] = [f"{project}-{index}" for index in range(5)]
        pieces.append(part)
    training, calibration = _split_inner(pd.concat(pieces))
    for project in training.source_project.unique():
        assert (
            training.loc[training.source_project == project, "creation_time"].max()
            < calibration.loc[calibration.source_project == project, "creation_time"].min()
        )


def test_checkpoint_roundtrip_supports_idempotent_resume(tmp_path) -> None:
    path = tmp_path / "checkpoint.json"
    result = {"candidate_id": "R-01", "status": "trained", "macro_f1": 0.4}
    predictions = pd.DataFrame(
        {"row_key": ["safe-hash"], "project": ["BIRT"], "truth": ["HIGH"], "predicted": ["LOW"]}
    )
    _atomic_checkpoint(path, result, predictions)
    loaded_result, loaded_predictions = _load_checkpoint(path)
    assert loaded_result == result
    pd.testing.assert_frame_equal(loaded_predictions[predictions.columns], predictions)


def test_fast_bootstrap_is_equivalent_to_reference_implementation() -> None:
    predictions = pd.DataFrame(
        {
            "truth": ["HIGH", "LOW", "MEDIUM"] * 9,
            "baseline": ["MEDIUM", "LOW", "MEDIUM"] * 9,
            "challenger": ["HIGH", "LOW", "MEDIUM"] * 9,
            "project": list(
                project
                for project in (
                    "BIRT",
                    "CDT",
                    "Equinox",
                    "JDT",
                    "MYLYN",
                    "PDE",
                    "Papyrus",
                    "Platform",
                    "TPTP",
                )
                for _ in range(3)
            ),
        }
    )
    labels = ["HIGH", "LOW", "MEDIUM"]
    reference = paired_bootstrap(predictions, labels, n_resamples=25)
    fast = fast_paired_bootstrap(predictions, labels, n_resamples=25)
    assert fast == pytest.approx(reference)
