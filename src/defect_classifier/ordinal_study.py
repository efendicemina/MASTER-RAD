"""Controlled MYLYN development-only ordinal severity study."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.isotonic import isotonic_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_recall_fscore_support,
)

from .development_study import LABELS, FrozenFold, build_study_pipeline, freeze_temporal_folds
from .hierarchical_study import (
    BASELINE_TOLERANCE,
    CONFLICT_IDS,
    EXPECTED_FOLD_HASH,
    MYLYN_DEVELOPMENT_SHA256,
    _metrics,
    _row_hash,
    load_hierarchical_development,
)
from .reporting import save_confusion_matrix_figure
from .utils import ensure_directory, sha256_file, write_csv, write_json

ORDERED_LABELS = ["trivial", "minor", "normal", "major", "critical", "blocker"]
LABEL_TO_RANK = {label: rank for rank, label in enumerate(ORDERED_LABELS)}
RANK_TO_LABEL = {rank: label for label, rank in LABEL_TO_RANK.items()}
THRESHOLDS = tuple(range(5))
BASELINE = 0.2262765
PRIMARY_THRESHOLD = 0.2463
SECONDARY_FLOOR = 0.2163
RIDGE_ALPHA = 1.0
PREPROCESSING_FINGERPRINT = hashlib.sha256(
    b"summary_description|lowercase|remove_html|replace_urls|replace_emails|"
    b"unicode|whitespace|word12|min_df2|max_df.98|max_features50000|sublinear"
).hexdigest()


def label_to_rank(label: str) -> int:
    try:
        return LABEL_TO_RANK[label]
    except KeyError as error:
        raise ValueError(f"Invalid severity label: {label}") from error


def rank_to_label(rank: int) -> str:
    if rank not in RANK_TO_LABEL:
        raise ValueError(f"Invalid severity rank: {rank}")
    return RANK_TO_LABEL[rank]


def cumulative_targets(ranks: np.ndarray) -> np.ndarray:
    values = np.asarray(ranks, dtype=int)
    if np.any((values < 0) | (values > 5)):
        raise ValueError("Ranks must be in 0..5")
    return np.column_stack([values > threshold for threshold in THRESHOLDS]).astype(int)


def binary_class_weights(target: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(target, return_counts=True)
    if len(values) != 2:
        raise ValueError("Every cumulative threshold requires both binary classes")
    return {
        int(value): len(target) / (2 * int(count))
        for value, count in zip(values, counts, strict=True)
    }


def project_monotonic(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("Cumulative probabilities must have shape (n, 5)")
    clipped = np.clip(values, 0.0, 1.0)
    return np.vstack(
        [isotonic_regression(row, increasing=False, y_min=0.0, y_max=1.0) for row in clipped]
    )


def validate_monotonic(probabilities: np.ndarray, tolerance: float = 1e-12) -> bool:
    values = np.asarray(probabilities)
    return bool(
        np.all(values >= -tolerance)
        and np.all(values <= 1 + tolerance)
        and np.all(np.diff(values, axis=1) <= tolerance)
    )


def reconstruct_class_probabilities(cumulative: np.ndarray) -> np.ndarray:
    values = np.asarray(cumulative, dtype=float)
    if not validate_monotonic(values):
        raise ValueError("Cumulative probabilities are not monotonic")
    classes = np.column_stack(
        [
            1 - values[:, 0],
            values[:, 0] - values[:, 1],
            values[:, 1] - values[:, 2],
            values[:, 2] - values[:, 3],
            values[:, 3] - values[:, 4],
            values[:, 4],
        ]
    )
    if np.any(classes < -1e-12) or not np.allclose(classes.sum(axis=1), 1.0, atol=1e-12):
        raise RuntimeError("Invalid reconstructed class probabilities")
    return np.clip(classes, 0.0, 1.0)


def decode_hard(raw_probabilities: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(raw_probabilities) >= 0.5).sum(axis=1), 0, 5).astype(int)


def decode_argmax(class_probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(class_probabilities, axis=1).astype(int)


def round_clip_ranks(values: np.ndarray) -> tuple[np.ndarray, int]:
    raw = np.asarray(values, dtype=float)
    rounded = np.floor(raw + 0.5)
    clipping_count = int(np.sum((rounded < 0) | (rounded > 5)))
    return np.clip(rounded, 0, 5).astype(int), clipping_count


def decode_expected_rank(class_probabilities: np.ndarray) -> np.ndarray:
    expected = np.asarray(class_probabilities) @ np.arange(6)
    return round_clip_ranks(expected)[0]


def ordinal_metrics(true_ranks: np.ndarray, predicted_ranks: np.ndarray) -> dict[str, Any]:
    truth, predicted = np.asarray(true_ranks, dtype=int), np.asarray(predicted_ranks, dtype=int)
    signed = predicted - truth
    absolute = np.abs(signed)
    correlation = spearmanr(truth, predicted).statistic
    result: dict[str, Any] = {
        "mean_absolute_error": float(absolute.mean()),
        "median_absolute_error": float(np.median(absolute)),
        "root_mean_squared_error": float(np.sqrt(mean_squared_error(truth, predicted))),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(truth, predicted, labels=list(range(6)), weights="quadratic")
        ),
        "spearman_correlation": float(0.0 if np.isnan(correlation) else correlation),
        "exact_rank_accuracy": float(np.mean(absolute == 0)),
        "within_one_accuracy": float(np.mean(absolute <= 1)),
        "within_two_accuracy": float(np.mean(absolute <= 2)),
        "mean_signed_error": float(signed.mean()),
        "overestimation_rate": float(np.mean(signed > 0)),
        "underestimation_rate": float(np.mean(signed < 0)),
        "extreme_error_count": int(np.sum(absolute >= 3)),
        "extreme_error_rate": float(np.mean(absolute >= 3)),
    }
    for distance in range(6):
        result[f"distance_{distance}_count"] = int(np.sum(absolute == distance))
        result[f"distance_{distance}_rate"] = float(np.mean(absolute == distance))
    return result


def _fit_representation(train: pd.DataFrame, validation: pd.DataFrame):
    template = build_study_pipeline("summary_description", "word_1_2")
    text = clone(template.named_steps["text"])
    vectorizer = clone(template.named_steps["features"])
    train_text = text.fit_transform(train.drop(columns=["severity"]))
    validation_text = text.transform(validation.drop(columns=["severity"]))
    train_matrix = vectorizer.fit_transform(train_text)
    validation_matrix = vectorizer.transform(validation_text)
    return vectorizer, train_matrix, validation_matrix


def _safe_oof(
    validation: pd.DataFrame,
    fold: int,
    predicted_ranks: np.ndarray,
    continuous: np.ndarray | None = None,
) -> pd.DataFrame:
    truth = validation.severity.map(LABEL_TO_RANK).to_numpy()
    result = pd.DataFrame(
        {
            "id": validation.id.astype(str).to_numpy(),
            "fold": fold,
            "true_label": validation.severity.to_numpy(),
            "true_rank": truth,
            "predicted_label": [rank_to_label(int(value)) for value in predicted_ranks],
            "predicted_rank": predicted_ranks,
            "absolute_rank_error": np.abs(predicted_ranks - truth),
        }
    )
    if continuous is not None:
        result["continuous_rank_prediction"] = continuous
    return result


def _fold_result(
    method: str,
    fold: int,
    truth: np.ndarray,
    predicted: np.ndarray,
    runtime: float,
    features: int,
    size: int,
    rss_delta: int,
) -> dict[str, Any]:
    true_labels = np.asarray([rank_to_label(int(value)) for value in truth])
    predicted_labels = np.asarray([rank_to_label(int(value)) for value in predicted])
    return {
        "method": method,
        "fold": fold,
        **_metrics(true_labels, predicted_labels),
        **ordinal_metrics(truth, predicted),
        "runtime_seconds": runtime,
        "feature_count": features,
        "model_size_bytes": size,
        "rss_delta_bytes": rss_delta,
    }


def _evaluate(
    frame: pd.DataFrame, folds: tuple[FrozenFold, ...]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    rows, oof = [], {name: [] for name in METHOD_NAMES}
    threshold_rows, monotonicity_rows = [], []
    for fold in folds:
        train, validation = frame.iloc[list(fold.train)], frame.iloc[list(fold.validation)]
        train_ranks = train.severity.map(LABEL_TO_RANK).to_numpy()
        true_ranks = validation.severity.map(LABEL_TO_RANK).to_numpy()
        rss_before, started = psutil.Process().memory_info().rss, perf_counter()
        vectorizer, x_train, x_validation = _fit_representation(train, validation)
        features = len(vectorizer.get_feature_names_out())

        flat_start = perf_counter()
        flat = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=42
        ).fit(x_train, train.severity)
        flat_labels = flat.predict(x_validation)
        flat_ranks = np.asarray([label_to_rank(value) for value in flat_labels])
        flat_runtime = perf_counter() - flat_start
        rows.append(
            _fold_result(
                "flat",
                fold.fold,
                true_ranks,
                flat_ranks,
                flat_runtime,
                features,
                len(pickle.dumps((vectorizer, flat))),
                psutil.Process().memory_info().rss - rss_before,
            )
        )
        oof["flat"].append(_safe_oof(validation, fold.fold, flat_ranks))

        targets = cumulative_targets(train_ranks)
        raw_probabilities = np.zeros((len(validation), 5))
        threshold_models, cumulative_start = [], perf_counter()
        for threshold in THRESHOLDS:
            target = targets[:, threshold]
            weights = binary_class_weights(target)
            model = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=42
            )
            fit_start = perf_counter()
            model.fit(x_train, target)
            fit_seconds = perf_counter() - fit_start
            infer_start = perf_counter()
            probability = model.predict_proba(x_validation)[:, list(model.classes_).index(1)]
            predicted_binary = (probability >= 0.5).astype(int)
            inference_seconds = perf_counter() - infer_start
            raw_probabilities[:, threshold] = probability
            threshold_models.append(model)
            precision, recall, binary_f1, _ = precision_recall_fscore_support(
                true_ranks > threshold,
                predicted_binary,
                labels=[0, 1],
                zero_division=0,
            )
            matrix = confusion_matrix(true_ranks > threshold, predicted_binary, labels=[0, 1])
            threshold_rows.append(
                {
                    "fold": fold.fold,
                    "threshold": threshold,
                    "question": f"rank > {threshold}",
                    "binary_macro_f1": f1_score(
                        true_ranks > threshold, predicted_binary, average="macro", zero_division=0
                    ),
                    "balanced_accuracy": balanced_accuracy_score(
                        true_ranks > threshold, predicted_binary
                    ),
                    "positive_precision": precision[1],
                    "positive_recall": recall[1],
                    "positive_f1": binary_f1[1],
                    "train_positive_support": int(target.sum()),
                    "train_negative_support": int(len(target) - target.sum()),
                    "validation_positive_support": int(np.sum(true_ranks > threshold)),
                    "validation_negative_support": int(np.sum(true_ranks <= threshold)),
                    "predicted_positive_rate": float(predicted_binary.mean()),
                    "class_weights": json.dumps(weights, sort_keys=True),
                    "tn": int(matrix[0, 0]),
                    "fp": int(matrix[0, 1]),
                    "fn": int(matrix[1, 0]),
                    "tp": int(matrix[1, 1]),
                    "feature_count": features,
                    "model_size_bytes": len(pickle.dumps(model)),
                    "fit_seconds": fit_seconds,
                    "inference_seconds": inference_seconds,
                }
            )
        corrected = project_monotonic(raw_probabilities)
        correction = np.abs(corrected - raw_probabilities)
        violations = np.any(np.diff(raw_probabilities, axis=1) > 1e-12, axis=1)
        monotonicity_rows.append(
            {
                "fold": fold.fold,
                "validation_rows": len(validation),
                "violating_rows": int(violations.sum()),
                "violation_rate": float(violations.mean()),
                "mean_absolute_correction": float(correction.mean()),
                "maximum_absolute_correction": float(correction.max()),
            }
        )
        class_probabilities = reconstruct_class_probabilities(corrected)
        cumulative_predictions = {
            "cumulative_hard": decode_hard(raw_probabilities),
            "cumulative_argmax": decode_argmax(class_probabilities),
            "cumulative_expected_rank": decode_expected_rank(class_probabilities),
        }
        cumulative_runtime = perf_counter() - cumulative_start
        cumulative_size = len(pickle.dumps((vectorizer, threshold_models)))
        for method, predicted in cumulative_predictions.items():
            rows.append(
                _fold_result(
                    method,
                    fold.fold,
                    true_ranks,
                    predicted,
                    cumulative_runtime,
                    features,
                    cumulative_size,
                    psutil.Process().memory_info().rss - rss_before,
                )
            )
            oof[method].append(_safe_oof(validation, fold.fold, predicted))

        ridge_start = perf_counter()
        ridge = Ridge(alpha=RIDGE_ALPHA, solver="lsqr").fit(x_train, train_ranks)
        continuous = ridge.predict(x_validation)
        ridge_ranks, clipping_count = round_clip_ranks(continuous)
        ridge_runtime = perf_counter() - ridge_start
        ridge_row = _fold_result(
            "ridge_rank",
            fold.fold,
            true_ranks,
            ridge_ranks,
            ridge_runtime,
            features,
            len(pickle.dumps((vectorizer, ridge))),
            psutil.Process().memory_info().rss - rss_before,
        )
        ridge_row["clipping_count"] = clipping_count
        rows.append(ridge_row)
        oof["ridge_rank"].append(
            _safe_oof(validation, fold.fold, ridge_ranks, continuous=continuous)
        )
        _ = perf_counter() - started
    return (
        pd.DataFrame(rows),
        {method: pd.concat(parts, ignore_index=True) for method, parts in oof.items()},
        pd.DataFrame(threshold_rows),
        pd.DataFrame(monotonicity_rows),
    )


METHOD_NAMES = (
    "flat",
    "cumulative_hard",
    "cumulative_argmax",
    "cumulative_expected_rank",
    "ridge_rank",
)


def _aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "weighted_f1",
        "accuracy",
        "minimum_per_class_recall",
        "mean_absolute_error",
        "median_absolute_error",
        "root_mean_squared_error",
        "quadratic_weighted_kappa",
        "spearman_correlation",
        "exact_rank_accuracy",
        "within_one_accuracy",
        "within_two_accuracy",
        "mean_signed_error",
        "overestimation_rate",
        "underestimation_rate",
        "extreme_error_rate",
    ]
    rows = []
    for method, group in folds.groupby("method"):
        row: dict[str, Any] = {
            "method": method,
            "mean_macro_f1": group.macro_f1.mean(),
            "std_macro_f1": group.macro_f1.std(ddof=0),
            "mean_blocker_critical_recall": np.mean(
                [group.recall_blocker.mean(), group.recall_critical.mean()]
            ),
            "mean_blocker_critical_f1": np.mean(
                [group.f1_blocker.mean(), group.f1_critical.mean()]
            ),
            "minimum_mean_class_recall": min(group[f"recall_{label}"].mean() for label in LABELS),
            "runtime_seconds": group.runtime_seconds.sum(),
            "mean_feature_count": group.feature_count.mean(),
            "mean_model_size_bytes": group.model_size_bytes.mean(),
            "maximum_rss_delta_bytes": group.rss_delta_bytes.max(),
        }
        for metric in metrics:
            if metric != "macro_f1":
                row[f"mean_{metric}"] = group[metric].mean()
        for label in LABELS:
            for metric in ("precision", "recall", "f1"):
                row[f"mean_{metric}_{label}"] = group[f"{metric}_{label}"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False)


def _write_matrix(output: Path, method: str, oof: pd.DataFrame) -> None:
    matrix = pd.DataFrame(
        confusion_matrix(oof.true_label, oof.predicted_label, labels=LABELS),
        index=LABELS,
        columns=LABELS,
    )
    normalized = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    write_csv(output / f"{method}_confusion_matrix.csv", matrix.reset_index(names="true_label"))
    write_csv(
        output / f"{method}_confusion_matrix_normalized.csv",
        normalized.reset_index(names="true_label"),
    )
    save_confusion_matrix_figure(
        output / f"{method}_confusion_matrix.png", matrix, f"{method} OOF confusion matrix"
    )


def _analysis_outputs(
    output: Path,
    frame: pd.DataFrame,
    folds: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> None:
    per_class, distributions, distances, extremes, minority, specified = [], [], [], [], [], []
    validation_lookup = frame.set_index(frame.id.astype(str))
    flat = predictions["flat"].set_index(["id", "fold"])
    comparison_rows = []
    for method, oof in predictions.items():
        for fold, group in folds[folds.method == method].groupby("fold"):
            row = group.iloc[0]
            for label in LABELS:
                per_class.append(
                    {
                        "method": method,
                        "fold": fold,
                        "severity": label,
                        "precision": row[f"precision_{label}"],
                        "recall": row[f"recall_{label}"],
                        "f1": row[f"f1_{label}"],
                        "support": row[f"support_{label}"],
                    }
                )
        for label in LABELS:
            distributions.append(
                {
                    "method": method,
                    "severity": label,
                    "predictions": int(np.sum(oof.predicted_label == label)),
                }
            )
        for distance in range(6):
            count = int(np.sum(oof.absolute_rank_error == distance))
            distances.append(
                {"method": method, "distance": distance, "count": count, "rate": count / len(oof)}
            )
        extreme = oof[oof.absolute_rank_error >= 3]
        extremes.append(
            {
                "method": method,
                "extreme_errors": len(extreme),
                "extreme_error_rate": len(extreme) / len(oof),
            }
        )
        for label in ("blocker", "critical"):
            rows = oof[oof.true_label == label]
            minority.append(
                {
                    "method": method,
                    "true_label": label,
                    "support": len(rows),
                    "exact_recall": float(np.mean(rows.predicted_label == label)),
                    "predicted_as_other_high": int(
                        np.sum(
                            rows.predicted_label
                            == ("critical" if label == "blocker" else "blocker")
                        )
                    ),
                    "at_least_two_levels_lower": int(
                        np.sum(rows.predicted_rank <= rows.true_rank - 2)
                    ),
                    "at_least_three_levels_lower": int(
                        np.sum(rows.predicted_rank <= rows.true_rank - 3)
                    ),
                }
            )
        specified.append(
            {
                "method": method,
                "blocker_as_critical": int(
                    np.sum((oof.true_label == "blocker") & (oof.predicted_label == "critical"))
                ),
                "blocker_at_least_two_lower": int(
                    np.sum((oof.true_label == "blocker") & (oof.predicted_rank <= 3))
                ),
                "blocker_at_least_three_lower": int(
                    np.sum((oof.true_label == "blocker") & (oof.predicted_rank <= 2))
                ),
                "critical_as_blocker": int(
                    np.sum((oof.true_label == "critical") & (oof.predicted_label == "blocker"))
                ),
                "critical_at_least_two_lower": int(
                    np.sum((oof.true_label == "critical") & (oof.predicted_rank <= 2))
                ),
                "trivial_at_least_two_higher": int(
                    np.sum((oof.true_label == "trivial") & (oof.predicted_rank >= 2))
                ),
                "major_as_normal": int(
                    np.sum((oof.true_label == "major") & (oof.predicted_label == "normal"))
                ),
                "normal_as_major": int(
                    np.sum((oof.true_label == "normal") & (oof.predicted_label == "major"))
                ),
                "minor_as_normal": int(
                    np.sum((oof.true_label == "minor") & (oof.predicted_label == "normal"))
                ),
                "normal_as_minor": int(
                    np.sum((oof.true_label == "normal") & (oof.predicted_label == "minor"))
                ),
                "normal_predictions": int(np.sum(oof.predicted_label == "normal")),
            }
        )
        if method != "flat":
            joined = oof.set_index(["id", "fold"]).join(
                flat[["absolute_rank_error"]], rsuffix="_flat"
            )
            comparison_rows.append(
                {
                    "method": method,
                    "corrected_vs_flat": int(
                        np.sum(
                            (joined.absolute_rank_error == 0)
                            & (joined.absolute_rank_error_flat > 0)
                        )
                    ),
                    "worsened_vs_flat": int(
                        np.sum(joined.absolute_rank_error > joined.absolute_rank_error_flat)
                    ),
                    "improved_distance_vs_flat": int(
                        np.sum(joined.absolute_rank_error < joined.absolute_rank_error_flat)
                    ),
                }
            )
        enriched = oof.copy()
        source = validation_lookup.loc[enriched.id.astype(str)]
        enriched["text_length_group"] = pd.cut(
            (source.summary.fillna("") + " " + source.description.fillna("")).str.len().to_numpy(),
            [-1, 200, 1000, np.inf],
            labels=["short", "medium", "long"],
        )
        enriched["time_period"] = pd.cut(
            source.creation_time.rank(method="first").to_numpy(),
            [0, len(source) / 3, 2 * len(source) / 3, np.inf],
            labels=["early", "middle", "late"],
        )
        for dimension in ("text_length_group", "time_period"):
            for value, group in enriched.groupby(dimension, observed=True):
                comparison_rows.append(
                    {
                        "method": method,
                        "slice": dimension,
                        "value": str(value),
                        "rows": len(group),
                        "macro_f1": f1_score(
                            group.true_label,
                            group.predicted_label,
                            labels=LABELS,
                            average="macro",
                            zero_division=0,
                        ),
                        "mean_absolute_error": group.absolute_rank_error.mean(),
                    }
                )
    write_csv(output / "per_class_results.csv", pd.DataFrame(per_class))
    write_csv(output / "predicted_class_distributions.csv", pd.DataFrame(distributions))
    write_csv(output / "error_distance_distribution.csv", pd.DataFrame(distances))
    write_csv(output / "extreme_error_summary.csv", pd.DataFrame(extremes))
    write_csv(output / "blocker_critical_analysis.csv", pd.DataFrame(minority))
    write_csv(output / "specified_ordinal_errors.csv", pd.DataFrame(specified))
    write_csv(output / "error_analysis_slices.csv", pd.DataFrame(comparison_rows))


def _verify_folds(frame: pd.DataFrame) -> tuple[tuple[FrozenFold, ...], dict[str, Any]]:
    folds = freeze_temporal_folds(frame)
    from .development_study import folds_fingerprint

    fingerprint = folds_fingerprint(folds)
    if fingerprint != EXPECTED_FOLD_HASH:
        raise RuntimeError("Ordinal fold membership differs from the approved study")
    expected_sizes = [(1916, 1871), (3832, 1855), (5748, 1889)]
    actual_sizes = [(len(fold.train), len(fold.validation)) for fold in folds]
    if actual_sizes != expected_sizes:
        raise RuntimeError("Ordinal fold sizes differ from the approved study")
    return folds, {
        "verified": True,
        "dataset_fingerprint": MYLYN_DEVELOPMENT_SHA256,
        "folds_fingerprint": fingerprint,
        "preprocessing_fingerprint": PREPROCESSING_FINGERPRINT,
        "folds": [
            {
                "fold": fold.fold,
                "train_rows": len(fold.train),
                "validation_rows": len(fold.validation),
                "train_row_hash": _row_hash(frame, fold.train),
                "validation_row_hash": _row_hash(frame, fold.validation),
            }
            for fold in folds
        ],
    }


def _sensitivity(frame: pd.DataFrame, folds: tuple[FrozenFold, ...]) -> pd.DataFrame:
    excluded = frame.id.astype(str).isin(CONFLICT_IDS)
    sensitivity_folds = tuple(
        FrozenFold(
            fold.fold,
            tuple(index for index in fold.train if not excluded.iloc[index]),
            tuple(index for index in fold.validation if not excluded.iloc[index]),
        )
        for fold in folds
    )
    results, _, _, _ = _evaluate(frame, sensitivity_folds)
    aggregate = _aggregate(results)
    aggregate["diagnostic_only"] = True
    aggregate["excluded_known_conflict_rows"] = int(excluded.sum())
    return aggregate


def run_ordinal_study(development_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    ensure_directory(output)
    frame = load_hierarchical_development(development_path)
    folds, identity = _verify_folds(frame)
    write_json(output / "fold_identity_verification.json", identity)
    write_json(
        output / "study_manifest.json",
        {
            "dataset_sha256": sha256_file(development_path),
            "development_rows": len(frame),
            "fixed_severity_order": LABEL_TO_RANK,
            "fold_hash": EXPECTED_FOLD_HASH,
            "ridge_alpha": RIDGE_ALPHA,
            "protocol_frozen": True,
            "held_out_test_accessed": False,
            "other_project_trained": False,
            "additional_main_row_removals": 0,
        },
    )
    fold_results, predictions, thresholds, monotonicity = _evaluate(frame, folds)
    flat_mean = fold_results[fold_results.method == "flat"].macro_f1.mean()
    if abs(flat_mean - BASELINE) > BASELINE_TOLERANCE:
        raise RuntimeError(f"Flat baseline reproduction {flat_mean:.7f} exceeds tolerance")
    write_csv(
        output / "flat_baseline_reproduction.csv",
        fold_results[fold_results.method == "flat"],
    )
    comparison = _aggregate(fold_results)
    write_csv(output / "fold_results.csv", fold_results)
    write_csv(output / "exact_class_metrics.csv", fold_results)
    ordinal_columns = [
        "method",
        "fold",
        "mean_absolute_error",
        "median_absolute_error",
        "root_mean_squared_error",
        "quadratic_weighted_kappa",
        "spearman_correlation",
        "exact_rank_accuracy",
        "within_one_accuracy",
        "within_two_accuracy",
        "mean_signed_error",
        "overestimation_rate",
        "underestimation_rate",
        "extreme_error_count",
        "extreme_error_rate",
    ] + [f"distance_{distance}_{suffix}" for distance in range(6) for suffix in ("count", "rate")]
    write_csv(output / "ordinal_metrics.csv", fold_results[ordinal_columns])
    write_csv(output / "threshold_results.csv", thresholds)
    write_csv(
        output / "threshold_support.csv",
        thresholds[
            [
                "fold",
                "threshold",
                "train_positive_support",
                "train_negative_support",
                "validation_positive_support",
                "validation_negative_support",
                "class_weights",
            ]
        ],
    )
    write_csv(output / "probability_monotonicity_audit.csv", monotonicity)
    for method, oof in predictions.items():
        write_csv(output / f"{method}_oof_predictions.csv", oof)
        _write_matrix(output, method, oof)
    _analysis_outputs(output, frame, fold_results, predictions)
    sensitivity = _sensitivity(frame, folds)
    write_csv(output / "conflict_sensitivity.csv", sensitivity)

    flat = comparison[comparison.method == "flat"].iloc[0]
    candidates = comparison[comparison.method != "flat"].copy()
    candidates["primary_met"] = candidates.mean_macro_f1 >= PRIMARY_THRESHOLD
    candidates["mae_relative_improvement"] = (
        flat.mean_mean_absolute_error - candidates.mean_mean_absolute_error
    ) / flat.mean_mean_absolute_error
    candidates["kappa_gain"] = (
        candidates.mean_quadratic_weighted_kappa - flat.mean_quadratic_weighted_kappa
    )
    candidates["within_one_gain"] = (
        candidates.mean_within_one_accuracy - flat.mean_within_one_accuracy
    )
    candidates["extreme_relative_reduction"] = (
        flat.mean_extreme_error_rate - candidates.mean_extreme_error_rate
    ) / flat.mean_extreme_error_rate
    candidates["bc_recall_change"] = (
        candidates.mean_blocker_critical_recall - flat.mean_blocker_critical_recall
    )
    candidates["maximum_recall_loss"] = candidates.apply(
        lambda row: max(
            flat[f"mean_recall_{label}"] - row[f"mean_recall_{label}"] for label in LABELS
        ),
        axis=1,
    )
    distributions = pd.read_csv(output / "predicted_class_distributions.csv")
    plausible = {}
    for method in candidates.method:
        counts = distributions[distributions.method == method].predictions
        plausible[method] = bool((counts > 0).all() and counts.max() / counts.sum() < 0.85)
    candidates["distribution_plausible"] = candidates.method.map(plausible)
    candidates["secondary_met"] = (
        (candidates.mean_macro_f1 >= SECONDARY_FLOOR)
        & (candidates.mae_relative_improvement >= 0.10)
        & (candidates.kappa_gain >= 0.05)
        & (candidates.within_one_gain >= 0.03)
        & (candidates.extreme_relative_reduction >= 0.20)
        & (candidates.bc_recall_change >= -0.02)
        & (candidates.maximum_recall_loss <= 0.10)
        & candidates.distribution_plausible
    )
    comparison = comparison.merge(
        candidates[
            [
                "method",
                "primary_met",
                "mae_relative_improvement",
                "kappa_gain",
                "within_one_gain",
                "extreme_relative_reduction",
                "bc_recall_change",
                "maximum_recall_loss",
                "distribution_plausible",
                "secondary_met",
            ]
        ],
        on="method",
        how="left",
    )
    write_csv(output / "ordinal_model_comparison.csv", comparison)
    write_csv(
        output / "runtime_summary.csv",
        comparison[
            [
                "method",
                "runtime_seconds",
                "mean_feature_count",
                "mean_model_size_bytes",
                "maximum_rss_delta_bytes",
            ]
        ],
    )
    decision = (
        "primary"
        if candidates.primary_met.any()
        else ("secondary" if candidates.secondary_met.any() else "no_improvement")
    )
    summary = [
        "# MYLYN ordinal study summary",
        "",
        f"Flat baseline reproduced at macro-F1 `{flat_mean:.7f}`.",
        f"Decision: `{decision}`. Held-out test accessed: `false`.",
        "",
        "```csv",
        comparison.to_csv(index=False).strip(),
        "```",
    ]
    (output / "ordinal_study_summary.md").write_text("\n".join(summary), encoding="utf-8")
    analysis = pd.read_csv(output / "error_analysis_slices.csv")
    (output / "ordinal_error_analysis.md").write_text(
        "# MYLYN ordinal development OOF error analysis\n\n```csv\n"
        + analysis.to_csv(index=False).strip()
        + "\n```",
        encoding="utf-8",
    )
    return {
        "decision": decision,
        "flat_mean_macro_f1": flat_mean,
        "comparison": comparison.to_dict("records"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_ordinal_study(arguments.development, arguments.output), indent=2, default=str
        )
    )
