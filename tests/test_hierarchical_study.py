from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defect_classifier.development_study import (
    LABELS,
    folds_fingerprint,
    freeze_temporal_folds,
    redact,
)
from defect_classifier.hierarchical_study import (
    CONFLICT_IDS,
    EXPECTED_FOLD_HASH,
    HIERARCHIES,
    _metrics,
    calculated_class_weights,
    compose_soft_scores,
    decompose_errors,
    fit_node,
    hard_route,
    hierarchy_mapping,
    load_hierarchical_development,
    oracle_route,
    reverse_group_membership,
    run_hierarchical_study,
)
from defect_classifier.pretrained_study import EXPECTED_FOLD_HASH as PRETRAINED_FOLD_HASH


def tiny_frame() -> pd.DataFrame:
    rows = []
    for index, label in enumerate(LABELS * 2):
        rows.append(
            {
                "id": str(index),
                "summary": f"summary {label} group{index % 3} token{index}",
                "description": f"description {label}",
                "severity": label,
                "creation_time": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=index),
                "product": "forbidden",
                "component": "forbidden",
                "duplicate_group": f"MYLYN:{index}",
            }
        )
    return pd.DataFrame(rows)


def test_fixed_labels_and_hierarchy_mappings():
    assert LABELS == ["blocker", "critical", "major", "normal", "minor", "trivial"]
    assert hierarchy_mapping("hierarchy_a") == {
        "blocker": "SEVERE",
        "critical": "SEVERE",
        "major": "NON_SEVERE",
        "normal": "NON_SEVERE",
        "minor": "NON_SEVERE",
        "trivial": "NON_SEVERE",
    }
    assert hierarchy_mapping("hierarchy_b")["major"] == "MEDIUM"
    assert hierarchy_mapping("hierarchy_b")["minor"] == "LOW"
    assert reverse_group_membership("hierarchy_a")["NON_SEVERE"] == (
        "major",
        "normal",
        "minor",
        "trivial",
    )


def test_every_child_node_and_multiclass_non_severe_fit():
    frame = tiny_frame()
    for hierarchy in HIERARCHIES.values():
        for labels in hierarchy.values():
            subset = frame[frame.severity.isin(labels)]
            model = fit_node(subset, subset.severity)
            assert set(model.named_steps["clf"].classes_) == set(labels)
    assert len(HIERARCHIES["hierarchy_a"]["NON_SEVERE"]) == 4


def test_level1_training_uses_fixed_groups():
    frame = tiny_frame()
    target = frame.severity.map(hierarchy_mapping("hierarchy_b"))
    model = fit_node(frame, target)
    assert set(model.named_steps["clf"].classes_) == {"HIGH", "MEDIUM", "LOW"}


def test_single_class_node_fails_and_weights_are_training_only():
    frame = tiny_frame()
    one = frame[frame.severity == "blocker"]
    with pytest.raises(ValueError, match="two training classes"):
        fit_node(one, one.severity)
    weights = calculated_class_weights(pd.Series(["blocker", "blocker", "critical"]))
    assert weights == {"blocker": 0.75, "critical": 1.5}


def test_vectorizer_is_fit_only_on_training_data():
    frame = tiny_frame()
    train = frame.iloc[:6].copy()
    model = fit_node(train, train.severity)
    vocabulary = model.named_steps["features"].vocabulary_
    assert "token11" not in vocabulary


def test_hard_routing_both_hierarchies():
    for groups in [np.array(["SEVERE", "NON_SEVERE"]), np.array(["HIGH", "LOW"])]:
        children = {str(group): np.array([f"{group}0", f"{group}1"]) for group in groups}
        assert hard_route(groups, children).tolist() == [f"{groups[0]}0", f"{groups[1]}1"]


@pytest.mark.parametrize("hierarchy", ["hierarchy_a", "hierarchy_b"])
def test_soft_composition_explicit_class_order_and_final_scores(hierarchy):
    groups = list(HIERARCHIES[hierarchy])
    group_classes = np.array(list(reversed(groups)))
    group_probs = np.array([[0.2 + index * 0.1 for index in range(len(groups))]])
    group_probs = group_probs / group_probs.sum()
    child_probs, child_classes = {}, {}
    for group, labels in HIERARCHIES[hierarchy].items():
        child_classes[group] = np.array(list(reversed(labels)))
        values = np.arange(1, len(labels) + 1, dtype=float)
        child_probs[group] = np.array([values / values.sum()])
    scores = compose_soft_scores(group_probs, group_classes, child_probs, child_classes)
    assert scores.shape == (1, 6)
    assert scores.sum() == pytest.approx(1.0)
    assert LABELS[int(scores.argmax())] in LABELS


def test_oracle_is_separate_and_error_decomposition():
    groups = np.array(["HIGH", "MEDIUM", "LOW"])
    children = {
        "HIGH": np.array(["blocker", "blocker", "blocker"]),
        "MEDIUM": np.array(["normal", "normal", "normal"]),
        "LOW": np.array(["minor", "minor", "minor"]),
    }
    assert oracle_route(groups, children).tolist() == ["blocker", "normal", "minor"]
    truth = np.array(["blocker", "major", "minor"])
    predicted = np.array(["blocker", "normal", "normal"])
    predicted_groups = np.array(["HIGH", "MEDIUM", "MEDIUM"])
    assert decompose_errors(truth, predicted, predicted_groups, "hierarchy_b").tolist() == [
        "correct",
        "within_group_error",
        "routing_error",
    ]


def test_fixed_six_label_metrics_when_predictions_absent():
    result = _metrics(np.array(["normal"]), np.array(["normal"]))
    assert result["macro_f1"] == pytest.approx(1 / 6)
    assert result["recall_blocker"] == 0


def test_conflict_ids_duplicate_policy_and_redaction():
    assert len(CONFLICT_IDS) == 5
    assert len(set(CONFLICT_IDS)) == 5
    value = redact("person@example.com https://example.com/a")
    assert "example.com" not in value


@pytest.mark.parametrize(
    "name", ["test_split.csv", "test_predictions.csv", "test_metrics.json", "platform.csv"]
)
def test_protected_and_other_project_inputs_are_rejected(tmp_path: Path, name: str):
    path = tmp_path / name
    path.write_text("not read", encoding="utf-8")
    with pytest.raises(ValueError):
        load_hierarchical_development(path)


def test_non_overwriting_output(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        run_hierarchical_study(tmp_path / "development_split.csv", output)


def test_deterministic_repeated_node_execution():
    frame = tiny_frame()
    first = fit_node(frame, frame.severity).predict(frame.drop(columns=["severity"]))
    second = fit_node(frame, frame.severity).predict(frame.drop(columns=["severity"]))
    assert np.array_equal(first, second)


def test_frozen_fold_identity_constant_and_no_enabled_challenger():
    assert EXPECTED_FOLD_HASH == "5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903"
    assert EXPECTED_FOLD_HASH == PRETRAINED_FOLD_HASH
    root = Path(__file__).resolve().parents[1]
    challenger = root / "configs" / "eclipse_training_mylyn_hierarchical_challenger.yaml"
    assert not challenger.exists()
    assert (root / "docs" / "mylyn_hierarchical_no_improvement.md").is_file()


def test_duplicate_groups_do_not_cross_frozen_folds():
    frame = tiny_frame()
    frame.loc[6, "duplicate_group"] = frame.loc[0, "duplicate_group"]
    for fold in freeze_temporal_folds(frame):
        train_groups = set(frame.iloc[list(fold.train)].duplicate_group)
        validation_groups = set(frame.iloc[list(fold.validation)].duplicate_group)
        assert train_groups.isdisjoint(validation_groups)
    assert folds_fingerprint(freeze_temporal_folds(frame)) == folds_fingerprint(
        freeze_temporal_folds(frame.copy())
    )
