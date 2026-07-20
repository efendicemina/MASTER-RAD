from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scipy import sparse

from defect_classifier.config import load_config
from defect_classifier.development_study import (
    INPUT_VARIANTS,
    build_study_pipeline,
    chronological_learning_subsets,
    evaluate_folds,
    folds_fingerprint,
    freeze_temporal_folds,
    load_development_only,
    validate_custom_class_weight,
)


def _development_frame() -> pd.DataFrame:
    labels = ["blocker", "critical", "major", "normal", "minor", "trivial"] * 4
    return pd.DataFrame(
        {
            "id": range(24),
            "summary": [
                f"summary category {['alpha', 'beta', 'gamma', 'delta'][index % 4]} item {index}"
                for index in range(24)
            ],
            "description": [
                f"description details class {label} {index}" for index, label in enumerate(labels)
            ],
            "severity": labels,
            "creation_time": pd.date_range("2020-01-01", periods=24, tz="UTC"),
            "duplicate_group": [f"group-{index}" for index in range(24)],
            "product": ["Mylyn"] * 24,
            "component": ["Core"] * 24,
        }
    )


@pytest.mark.parametrize("variant", INPUT_VARIANTS)
def test_text_input_variants_select_expected_fields(variant):
    frame = _development_frame()
    pipeline = build_study_pipeline(variant, "word_1_1")
    text = pipeline.named_steps["text"].fit_transform(frame)
    if variant == "summary":
        assert "description details" not in text[0]
    elif variant == "description":
        assert "summary token" not in text[0]
    else:
        assert "summary category" in text[0] and "description details" in text[0]


@pytest.mark.parametrize("feature_variant", ["char_wb", "word_char"])
def test_character_and_union_features_remain_sparse(feature_variant):
    frame = _development_frame()
    pipeline = build_study_pipeline("summary_description", feature_variant)
    text = pipeline.named_steps["text"].fit_transform(frame)
    matrix = pipeline.named_steps["features"].fit_transform(text)
    assert sparse.issparse(matrix)
    assert matrix.shape[1] > 0


def test_frozen_folds_are_deterministic_and_identical():
    frame = _development_frame()
    first = freeze_temporal_folds(frame)
    second = freeze_temporal_folds(frame.copy())
    assert first == second
    assert folds_fingerprint(first) == folds_fingerprint(second)


def test_learning_subsets_are_chronological_and_increasing():
    subsets = chronological_learning_subsets(_development_frame(), (0.25, 0.5, 1.0))
    assert [len(frame) for _, frame in subsets] == [6, 12, 24]
    assert all(frame.creation_time.is_monotonic_increasing for _, frame in subsets)


def test_oof_predictions_only_cover_validation_indices():
    frame = _development_frame()
    folds = freeze_temporal_folds(frame, n_splits=2)
    result, predictions, _ = evaluate_folds(
        frame,
        folds,
        build_study_pipeline("summary", "word_1_1", "DummyClassifier", None),
        {"model": "DummyClassifier"},
    )
    expected_ids = {frame.iloc[index].id for fold in folds for index in fold.validation}
    assert set(predictions.id) == expected_ids
    assert result["folds_fingerprint"] == folds_fingerprint(folds)


def test_custom_class_weights_are_strictly_validated():
    valid = {label: 1.0 for label in ["blocker", "critical", "major", "normal", "minor", "trivial"]}
    assert validate_custom_class_weight(valid) == valid
    with pytest.raises(ValueError):
        validate_custom_class_weight({"normal": 1.0})
    valid["blocker"] = 0
    with pytest.raises(ValueError):
        validate_custom_class_weight(valid)


def test_loader_rejects_held_out_test_path(tmp_path: Path):
    test_path = tmp_path / "test_split.csv"
    _development_frame().to_csv(test_path, index=False)
    with pytest.raises(ValueError, match="only development_split"):
        load_development_only(test_path)


def test_challenger_configuration_remains_disabled():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "eclipse_training_mylyn_challenger.yaml")
    assert config["training"]["enabled"] is False
    assert config["held_out_test"]["evaluation_allowed"] is False
    assert config["dataset"]["resolved_paths"][0].endswith("MYLYN.parquet")
