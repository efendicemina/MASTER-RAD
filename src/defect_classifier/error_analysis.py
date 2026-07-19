"""Error analysis helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


def build_error_analysis(
    predictions: pd.DataFrame, summary_column: str, description_column: str
) -> dict[str, Any]:
    """Construct an error analysis summary from prediction rows."""

    errors = predictions.loc[~predictions["correct"]].copy()
    confusions = Counter(zip(errors["true_label"], errors["predicted_label"], strict=True))
    by_true = {
        label: group[
            ["report_id", "true_label", "predicted_label", summary_column, description_column]
        ].to_dict(orient="records")
        for label, group in errors.groupby("true_label")
    }
    by_pred = {
        label: group[
            ["report_id", "true_label", "predicted_label", summary_column, description_column]
        ].to_dict(orient="records")
        for label, group in errors.groupby("predicted_label")
    }

    lengths = predictions.assign(
        text_length=predictions[summary_column].fillna("").astype(str).str.len()
        + predictions[description_column].fillna("").astype(str).str.len()
    )
    stats = {
        "correct_mean": float(lengths.loc[lengths["correct"], "text_length"].mean())
        if lengths["correct"].any()
        else 0.0,
        "incorrect_mean": float(lengths.loc[~lengths["correct"], "text_length"].mean())
        if (~lengths["correct"]).any()
        else 0.0,
        "correct_median": float(lengths.loc[lengths["correct"], "text_length"].median())
        if lengths["correct"].any()
        else 0.0,
        "incorrect_median": float(lengths.loc[~lengths["correct"], "text_length"].median())
        if (~lengths["correct"]).any()
        else 0.0,
    }

    representative = errors.sort_values(["true_label", "predicted_label", "report_id"]).head(20)
    return {
        "most_common_confusions": [
            {"true_label": true_label, "predicted_label": predicted_label, "count": int(count)}
            for (true_label, predicted_label), count in confusions.most_common()
        ],
        "false_predictions_by_true_class": by_true,
        "false_predictions_by_predicted_class": by_pred,
        "representative_misclassified_examples": representative[
            ["report_id", "true_label", "predicted_label", summary_column, description_column]
        ].to_dict(orient="records"),
        "report_length_statistics": stats,
    }
