"""Evaluation and bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    confusion_matrix: pd.DataFrame
    normalized_confusion_matrix: pd.DataFrame
    classification_report: dict[str, Any]


def calculate_metrics(
    y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray, labels: list[str]
) -> dict[str, Any]:
    """Calculate the standard evaluation metrics used in the thesis."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        "macro_f1": f1_score(
            y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0
        ),
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "balanced_accuracy": balanced_accuracy_score(y_true_arr, y_pred_arr),
        "macro_precision": precision_score(
            y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
        ),
        "per_class_precision": precision_score(
            y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
        ).tolist(),
        "per_class_f1": f1_score(
            y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0
        ).tolist(),
        "per_class_support": classification_report(
            y_true_arr, y_pred_arr, labels=labels, output_dict=True, zero_division=0
        ),
    }


def build_confusion_frames(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build raw and normalized confusion matrix data frames."""

    raw = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return (
        pd.DataFrame(raw, index=labels, columns=labels),
        pd.DataFrame(normalized, index=labels, columns=labels),
    )


def bootstrap_macro_f1_ci(
    y_true: list[str],
    y_pred: list[str],
    n_resamples: int,
    confidence_level: float,
    random_state: int,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """Estimate macro F1 CI with class-stratified bootstrap resampling."""

    rng = np.random.default_rng(random_state)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    scores = []
    fixed_labels = labels or sorted(set(y_true_arr.astype(str)) | set(y_pred_arr.astype(str)))
    class_indices = [np.flatnonzero(y_true_arr == label) for label in fixed_labels]
    class_indices = [indices for indices in class_indices if len(indices)]
    for _ in range(n_resamples):
        indices = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in class_indices]
        )
        score = f1_score(
            y_true_arr[indices],
            y_pred_arr[indices],
            labels=fixed_labels,
            average="macro",
            zero_division=0,
        )
        scores.append(score)
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(scores, alpha))
    upper = float(np.quantile(scores, 1.0 - alpha))
    return {"lower": lower, "upper": upper, "mean": float(np.mean(scores))}


def evaluate_predictions(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> EvaluationResult:
    """Bundle all evaluation outputs for a held-out test set."""

    metrics = calculate_metrics(y_true, y_pred, labels)
    confusion, normalized = build_confusion_frames(y_true, y_pred, labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return EvaluationResult(
        metrics=metrics,
        confusion_matrix=confusion,
        normalized_confusion_matrix=normalized,
        classification_report=report,
    )
