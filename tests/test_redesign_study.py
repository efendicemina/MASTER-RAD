from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defect_classifier.development_study import freeze_temporal_folds, redact
from defect_classifier.redesign_study import (
    FEATURES,
    S6_LABELS,
    TASK_LABELS,
    _select_candidate,
    apply_rare_categories,
    build_features,
    clean_text_value,
    enforce_metadata_eligibility,
    fit_rare_categories,
    fixed_metrics,
    map_target,
    normalize_severity,
    run_redesign_study,
    structural_values,
)


def tiny_frame() -> pd.DataFrame:
    rows = []
    for index, label in enumerate(S6_LABELS * 3):
        rows.append(
            {
                "id": str(index),
                "summary": f"Summary shared{index % 3} {label} unique{index}",
                "description": f"<b>Description</b> shared{index % 2} bug #{1000 + index}",
                "severity": label,
                "creation_time": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=index),
                "product": "p",
                "component": "c",
                "duplicate_group": f"MYLYN:{index}",
            }
        )
    return pd.DataFrame(rows)


def test_fixed_s6_s3_s2_mappings_and_invalid_rejection():
    values = pd.Series(S6_LABELS)
    assert map_target(values, "s6").tolist() == S6_LABELS
    assert map_target(values, "s3").tolist() == ["HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW", "LOW"]
    assert map_target(values, "s2").tolist() == ["HIGH_IMPACT"] * 3 + ["LOWER_IMPACT"] * 3
    assert TASK_LABELS["s3"] == ["HIGH", "MEDIUM", "LOW"]
    with pytest.raises(ValueError):
        normalize_severity("enhancement")
    with pytest.raises(ValueError):
        map_target(pd.Series(["unknown"]), "s6")


def test_severity_normalization():
    assert normalize_severity(" Critical ") == "critical"
    assert normalize_severity("MAJOR") == "major"


def test_cleaning_unicode_html_entities_urls_email_issue_and_label_masking():
    value = clean_text_value(
        "CRITICAL &amp; café", "<b>See</b> https://example.com a@example.com bug #1234", True
    )
    assert "critical" not in value
    assert "&amp;" not in value and "<b>" not in value
    assert "example.com" not in value
    assert "1234" not in value
    assert "[severity_term]" in value


def test_structural_features_are_initial_text_only_and_deterministic():
    frame = tiny_frame().iloc[:2]
    first = structural_values(frame)
    second = structural_values(frame.copy())
    assert first.shape == (2, 23)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("feature", FEATURES)
def test_all_text_feature_sets_fit_training_only(feature):
    frame = tiny_frame()
    transformer = build_features(feature)
    transformer.fit_transform(frame.iloc[:12])
    transformed = transformer.transform(frame.iloc[12:])
    assert transformed.shape[0] == 6
    if feature in {"F0", "F1"}:
        assert "unique17" not in transformer.named_steps["tfidf"].vocabulary_


def test_metadata_eligibility_and_ineligible_columns():
    audit = pd.DataFrame(
        [
            {"normalized_name": "product", "eligibility_status": "PRIMARY_ELIGIBLE"},
            {"normalized_name": "component", "eligibility_status": "SENSITIVITY_ONLY"},
        ]
    )
    assert enforce_metadata_eligibility(["Product"], audit) == ["product"]
    with pytest.raises(ValueError):
        enforce_metadata_eligibility(["Component"], audit)
    with pytest.raises(ValueError):
        enforce_metadata_eligibility(["Priority"], audit)
    with pytest.raises(ValueError):
        enforce_metadata_eligibility(["History/Activity Log"], audit)


def test_training_only_rare_and_missing_category_handling():
    training = pd.Series(["a", "a", "a", "a", "a", "b", None])
    retained = fit_rare_categories(training)
    assert retained == {"a"}
    validation = apply_rare_categories(pd.Series(["a", "b", "unseen", None]), retained)
    assert validation.tolist() == ["a", "[RARE]", "[RARE]", "[RARE]"]


def test_fixed_label_metrics_keep_missing_predictions_visible():
    result = fixed_metrics(
        np.array(["HIGH", "MEDIUM", "LOW"]), np.array(["MEDIUM"] * 3), TASK_LABELS["s3"]
    )
    assert result["recall_HIGH"] == 0
    assert result["minimum_class_recall"] == 0


def test_s2_high_metrics_and_binary_auc_inputs():
    result = fixed_metrics(
        np.array(["HIGH_IMPACT", "LOWER_IMPACT"]),
        np.array(["HIGH_IMPACT", "LOWER_IMPACT"]),
        TASK_LABELS["s2"],
    )
    assert result["precision_HIGH_IMPACT"] == 1
    assert result["recall_HIGH_IMPACT"] == 1


def test_candidate_tie_breaker_prefers_minimum_recall_then_simple_feature():
    rows = []
    for candidate, feature, recall in [("a", "F0", 0.2), ("b", "F1", 0.3)]:
        rows.append(
            {
                "candidate": candidate,
                "feature": feature,
                "family": "LinearSVC",
                "C": 1.0,
                "class_weight": "balanced",
                "macro_f1": 0.5,
                "minimum_class_recall": recall,
                "recall_HIGH": recall,
                "feature_count": 10,
                "runtime_seconds": 1,
            }
        )
    assert _select_candidate(pd.DataFrame(rows), "s3").candidate == "b"


def test_outer_temporal_duplicate_leakage_prevention():
    frame = tiny_frame()
    frame.loc[9, "duplicate_group"] = frame.loc[0, "duplicate_group"]
    for fold in freeze_temporal_folds(frame):
        train = set(frame.iloc[list(fold.train)].duplicate_group)
        validation = set(frame.iloc[list(fold.validation)].duplicate_group)
        assert train.isdisjoint(validation)


def test_redaction_and_non_overwriting_outputs(tmp_path: Path):
    assert "example.com" not in redact("a@example.com https://example.com")
    audit = tmp_path / "audit"
    audit.mkdir()
    with pytest.raises(FileExistsError):
        run_redesign_study(tmp_path / "development_split.csv", audit, tmp_path / "models")


def test_no_challengers_exist_before_material_result():
    root = Path(__file__).resolve().parents[1]
    for task in ("s6_enhanced", "s3", "s2"):
        assert not (root / "configs" / f"eclipse_training_mylyn_{task}_challenger.yaml").exists()
