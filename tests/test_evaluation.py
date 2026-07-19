from __future__ import annotations

from defect_classifier.evaluation import bootstrap_macro_f1_ci, calculate_metrics
from defect_classifier.training import FixedLabelMacroF1Scorer


class BinaryEstimator:
    def predict(self, features):
        return ["a"] * len(features)


def test_metric_calculation():
    metrics = calculate_metrics(["a", "b", "a"], ["a", "a", "a"], ["a", "b"])
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == 2 / 3


def test_bootstrap_confidence_interval():
    interval = bootstrap_macro_f1_ci(
        ["a", "b", "a"], ["a", "a", "a"], 10, 0.95, 42, labels=["a", "b"]
    )
    assert interval["lower"] <= interval["upper"]


def test_macro_f1_keeps_absent_expected_class():
    metrics = calculate_metrics(["a", "a"], ["a", "a"], ["a", "b"])
    assert metrics["macro_f1"] == 0.5


def test_fixed_label_scorer_handles_binary_temporal_fold():
    scorer = FixedLabelMacroF1Scorer(("a", "b", "c"))
    assert scorer(BinaryEstimator(), [1, 2], ["a", "b"]) == 2 / 9
