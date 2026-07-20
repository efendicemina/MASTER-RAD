from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from defect_classifier.development_study import LABELS, freeze_temporal_folds, redact
from defect_classifier.hierarchical_study import EXPECTED_FOLD_HASH
from defect_classifier.ordinal_study import (
    CONFLICT_IDS,
    LABEL_TO_RANK,
    ORDERED_LABELS,
    RANK_TO_LABEL,
    _fit_representation,
    binary_class_weights,
    cumulative_targets,
    decode_argmax,
    decode_expected_rank,
    decode_hard,
    label_to_rank,
    load_hierarchical_development,
    ordinal_metrics,
    project_monotonic,
    rank_to_label,
    reconstruct_class_probabilities,
    round_clip_ranks,
    run_ordinal_study,
    validate_monotonic,
)
from defect_classifier.pretrained_study import EXPECTED_FOLD_HASH as PRETRAINED_FOLD_HASH


def tiny_frame() -> pd.DataFrame:
    rows = []
    for index, label in enumerate(ORDERED_LABELS * 3):
        rows.append(
            {
                "id": str(index),
                "summary": f"summary group{index % 4} {label} unique{index}",
                "description": f"description group{index % 3} {label}",
                "severity": label,
                "creation_time": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=index),
                "product": "forbidden",
                "component": "forbidden",
                "duplicate_group": f"MYLYN:{index}",
            }
        )
    return pd.DataFrame(rows)


def test_fixed_order_and_bidirectional_mapping():
    assert ORDERED_LABELS == ["trivial", "minor", "normal", "major", "critical", "blocker"]
    expected_labels = {label: index for index, label in enumerate(ORDERED_LABELS)}
    expected_ranks = {index: label for index, label in enumerate(ORDERED_LABELS)}
    assert expected_labels == LABEL_TO_RANK
    assert expected_ranks == RANK_TO_LABEL
    assert [rank_to_label(label_to_rank(label)) for label in ORDERED_LABELS] == ORDERED_LABELS
    assert set(ORDERED_LABELS) == set(LABELS)


def test_invalid_labels_and_ranks_are_rejected():
    with pytest.raises(ValueError):
        label_to_rank("enhancement")
    with pytest.raises(ValueError):
        rank_to_label(6)
    with pytest.raises(ValueError):
        cumulative_targets(np.array([-1, 0]))


def test_cumulative_threshold_targets():
    result = cumulative_targets(np.array([0, 2, 5]))
    assert result.tolist() == [
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ]


def test_training_only_binary_weights_and_single_class_failure():
    assert binary_class_weights(np.array([0, 0, 0, 1])) == {0: 2 / 3, 1: 2.0}
    with pytest.raises(ValueError, match="both binary classes"):
        binary_class_weights(np.ones(4))


def test_training_only_tfidf_and_five_threshold_models():
    frame = tiny_frame()
    train, validation = frame.iloc[:12], frame.iloc[12:]
    vectorizer, x_train, x_validation = _fit_representation(train, validation)
    assert "unique17" not in vectorizer.vocabulary_
    targets = cumulative_targets(train.severity.map(LABEL_TO_RANK).to_numpy())
    for threshold in range(5):
        model = LogisticRegression(class_weight="balanced", random_state=42).fit(
            x_train, targets[:, threshold]
        )
        assert model.predict_proba(x_validation).shape == (6, 2)


def test_hard_decoder_uses_fixed_half_threshold():
    probabilities = np.array([[0.9, 0.7, 0.5, 0.49, 0.1], [0.1, 0.2, 0.3, 0.4, 0.6]])
    assert decode_hard(probabilities).tolist() == [3, 1]


def test_pava_preserves_monotonic_and_corrects_violation():
    monotonic = np.array([[0.9, 0.8, 0.5, 0.2, 0.1]])
    assert np.allclose(project_monotonic(monotonic), monotonic)
    raw = np.array([[0.9, 0.4, 0.8, 0.2, 0.3]])
    corrected = project_monotonic(raw)
    assert validate_monotonic(corrected)
    assert np.all((corrected >= 0) & (corrected <= 1))


def test_reconstructed_probabilities_and_decoders():
    cumulative = np.array([[0.9, 0.7, 0.5, 0.3, 0.1]])
    classes = reconstruct_class_probabilities(cumulative)
    assert np.all(classes >= 0)
    assert classes.sum() == pytest.approx(1.0)
    assert decode_argmax(classes).tolist() == [1]
    assert decode_expected_rank(classes).tolist() == [3]


def test_non_monotonic_reconstruction_is_rejected():
    with pytest.raises(ValueError, match="not monotonic"):
        reconstruct_class_probabilities(np.array([[0.7, 0.8, 0.4, 0.2, 0.1]]))


def test_ridge_half_up_rounding_and_clipping():
    ranks, clipping = round_clip_ranks(np.array([-1.0, 0.49, 0.5, 4.5, 6.0]))
    assert ranks.tolist() == [0, 0, 1, 5, 5]
    assert clipping == 2


def test_ordinal_metrics_mae_kappa_within_one_and_extremes():
    truth = np.array([0, 1, 2, 3, 4, 5])
    perfect = ordinal_metrics(truth, truth)
    assert perfect["mean_absolute_error"] == 0
    assert perfect["quadratic_weighted_kappa"] == pytest.approx(1)
    assert perfect["within_one_accuracy"] == 1
    predicted = np.array([5, 1, 2, 3, 4, 0])
    result = ordinal_metrics(truth, predicted)
    assert result["extreme_error_count"] == 2
    assert result["distance_5_count"] == 2
    assert result["overestimation_rate"] == pytest.approx(1 / 6)
    assert result["underestimation_rate"] == pytest.approx(1 / 6)


def test_frozen_fold_identity_and_duplicate_leakage():
    assert EXPECTED_FOLD_HASH == PRETRAINED_FOLD_HASH
    frame = tiny_frame()
    frame.loc[9, "duplicate_group"] = frame.loc[0, "duplicate_group"]
    for fold in freeze_temporal_folds(frame):
        train_groups = set(frame.iloc[list(fold.train)].duplicate_group)
        validation_groups = set(frame.iloc[list(fold.validation)].duplicate_group)
        assert train_groups.isdisjoint(validation_groups)


def test_conflict_sensitivity_identity_and_privacy_redaction():
    assert len(CONFLICT_IDS) == 5
    assert "example.com" not in redact("a@example.com https://example.com")


@pytest.mark.parametrize(
    "name",
    [
        "test_split.csv",
        "test_predictions.csv",
        "test_metrics.json",
        "held_out_labels.csv",
        "platform.csv",
    ],
)
def test_test_and_other_project_access_rejected(tmp_path: Path, name: str):
    path = tmp_path / name
    path.write_text("must not be read", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hierarchical_development(path)


def test_deterministic_repeated_projection():
    values = np.array([[0.2, 0.9, 0.1, 0.8, 0.4]])
    assert np.array_equal(project_monotonic(values), project_monotonic(values.copy()))


def test_non_overwriting_output_and_no_challenger(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_ordinal_study(tmp_path / "development_split.csv", output)
    root = Path(__file__).resolve().parents[1]
    assert not (root / "configs" / "eclipse_training_mylyn_ordinal_challenger.yaml").exists()
    assert not (root / "configs" / "eclipse_training_mylyn_ordinal_secondary.yaml").exists()
    assert (root / "docs" / "mylyn_ordinal_no_improvement.md").is_file()
