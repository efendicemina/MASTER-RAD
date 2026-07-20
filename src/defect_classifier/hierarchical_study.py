"""Controlled, MYLYN development-only hierarchical severity study."""

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .development_study import (
    LABELS,
    FrozenFold,
    build_study_pipeline,
    folds_fingerprint,
    freeze_temporal_folds,
)
from .pretrained_study import (
    EXPECTED_FOLD_HASH,
    MYLYN_DEVELOPMENT_SHA256,
    load_pretrained_development,
)
from .reporting import save_confusion_matrix_figure
from .utils import ensure_directory, sha256_file, write_csv, write_json

SEED = 42
BASELINE = 0.2263
BASELINE_TOLERANCE = 0.0001
PRIMARY_THRESHOLD = 0.2463
SECONDARY_FLOOR = 0.2163
CONFLICT_IDS = frozenset({"108445", "166615", "265078", "373112", "377134"})
HIERARCHIES: dict[str, dict[str, tuple[str, ...]]] = {
    "hierarchy_a": {
        "SEVERE": ("blocker", "critical"),
        "NON_SEVERE": ("major", "normal", "minor", "trivial"),
    },
    "hierarchy_b": {
        "HIGH": ("blocker", "critical"),
        "MEDIUM": ("major", "normal"),
        "LOW": ("minor", "trivial"),
    },
}
PREPROCESSING_FINGERPRINT = hashlib.sha256(
    b"summary_description|lowercase|remove_html|replace_urls|replace_emails|"
    b"unicode|whitespace|word12|min_df2|max_df.98|max_features50000|sublinear"
).hexdigest()


def hierarchy_mapping(name: str) -> dict[str, str]:
    if name not in HIERARCHIES:
        raise ValueError(f"Unknown hierarchy: {name}")
    return {label: group for group, labels in HIERARCHIES[name].items() for label in labels}


def reverse_group_membership(name: str) -> dict[str, tuple[str, ...]]:
    if name not in HIERARCHIES:
        raise ValueError(f"Unknown hierarchy: {name}")
    return HIERARCHIES[name].copy()


def load_hierarchical_development(path: str | Path) -> pd.DataFrame:
    """Use the same strict immutable-development gate as the pretrained study."""

    source = Path(path)
    if source.name != "development_split.csv" or "test" in source.name.lower():
        raise ValueError("Only immutable MYLYN development_split.csv is permitted")
    frame = load_pretrained_development(source)
    if set(frame.columns) - {
        "id",
        "summary",
        "description",
        "severity",
        "creation_time",
        "product",
        "component",
        "duplicate_group",
    }:
        raise ValueError("Unexpected columns in development artifact")
    return frame


def _row_hash(frame: pd.DataFrame, indices: tuple[int, ...]) -> str:
    payload = "\n".join(f"{index}:{frame.iloc[index]['id']}" for index in indices)
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_folds(frame: pd.DataFrame) -> tuple[tuple[FrozenFold, ...], dict[str, Any]]:
    folds = freeze_temporal_folds(frame)
    fingerprint = folds_fingerprint(folds)
    if fingerprint != EXPECTED_FOLD_HASH:
        raise RuntimeError("Frozen fold membership differs from approved MYLYN study")
    return folds, {
        "verified": True,
        "folds_fingerprint": fingerprint,
        "dataset_fingerprint": MYLYN_DEVELOPMENT_SHA256,
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


def _pipeline():
    return build_study_pipeline("summary_description", "word_1_2")


def calculated_class_weights(target: pd.Series) -> dict[str, float]:
    counts = target.value_counts()
    if len(counts) < 2:
        raise ValueError("Every node requires at least two training classes")
    return {str(label): len(target) / (len(counts) * count) for label, count in counts.items()}


def fit_node(train: pd.DataFrame, target: pd.Series):
    calculated_class_weights(target)
    model = _pipeline()
    model.fit(train.drop(columns=["severity"]), target)
    return model


def compose_soft_scores(
    group_probabilities: np.ndarray,
    group_classes: list[str] | np.ndarray,
    child_probabilities: dict[str, np.ndarray],
    child_classes: dict[str, list[str] | np.ndarray],
) -> np.ndarray:
    """Compose explicit six-class scores independent of estimator class order."""

    scores = np.zeros((len(group_probabilities), len(LABELS)), dtype=float)
    group_index = {str(value): index for index, value in enumerate(group_classes)}
    for group, probabilities in child_probabilities.items():
        if group not in group_index:
            raise ValueError(f"Missing Level-1 group probability: {group}")
        child_index = {str(value): index for index, value in enumerate(child_classes[group])}
        for label in HIERARCHIES[_hierarchy_for_groups(child_probabilities)][group]:
            if label not in child_index:
                raise ValueError(f"Missing child probability for {label}")
            scores[:, LABELS.index(label)] = (
                group_probabilities[:, group_index[group]] * probabilities[:, child_index[label]]
            )
    return scores


def _hierarchy_for_groups(children: dict[str, Any]) -> str:
    groups = set(children)
    for name, mapping in HIERARCHIES.items():
        if groups == set(mapping):
            return name
    raise ValueError("Child groups do not match a fixed hierarchy")


def hard_route(
    predicted_groups: np.ndarray, child_predictions: dict[str, np.ndarray]
) -> np.ndarray:
    return np.asarray(
        [child_predictions[str(group)][index] for index, group in enumerate(predicted_groups)]
    )


def oracle_route(true_groups: np.ndarray, child_predictions: dict[str, np.ndarray]) -> np.ndarray:
    """NON-DEPLOYABLE DIAGNOSTIC ONLY."""

    return hard_route(true_groups, child_predictions)


def decompose_errors(
    truth: np.ndarray, predicted: np.ndarray, predicted_groups: np.ndarray, hierarchy: str
) -> np.ndarray:
    mapping = hierarchy_mapping(hierarchy)
    true_groups = np.asarray([mapping[str(label)] for label in truth])
    return np.where(
        truth == predicted,
        "correct",
        np.where(true_groups != predicted_groups, "routing_error", "within_group_error"),
    )


def _metrics(
    truth: np.ndarray, predicted: np.ndarray, labels: list[str] = LABELS
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    result: dict[str, Any] = {
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "weighted_f1": f1_score(
            truth, predicted, labels=labels, average="weighted", zero_division=0
        ),
        "accuracy": accuracy_score(truth, predicted),
        "minimum_per_class_recall": float(np.min(recall)),
    }
    for index, label in enumerate(labels):
        result.update(
            {
                f"precision_{label}": precision[index],
                f"recall_{label}": recall[index],
                f"f1_{label}": f1[index],
                f"support_{label}": int(support[index]),
                f"predicted_{label}": int(np.sum(predicted == label)),
            }
        )
    return result


def _feature_count(model) -> int:
    return len(model.named_steps["features"].get_feature_names_out())


def _evaluate_flat(frame: pd.DataFrame, folds: tuple[FrozenFold, ...]):
    rows, oof = [], []
    for fold in folds:
        train, validation = frame.iloc[list(fold.train)], frame.iloc[list(fold.validation)]
        started = perf_counter()
        model = fit_node(train, train["severity"])
        predicted = model.predict(validation.drop(columns=["severity"]))
        row = {
            "method": "flat_baseline",
            "fold": fold.fold,
            **_metrics(validation.severity.to_numpy(), predicted),
        }
        row.update(
            runtime_seconds=perf_counter() - started,
            feature_count=_feature_count(model),
            model_size_bytes=len(pickle.dumps(model)),
        )
        rows.append(row)
        oof.append(_safe_predictions(validation, predicted, fold.fold))
    return pd.DataFrame(rows), pd.concat(oof, ignore_index=True)


def _safe_predictions(validation: pd.DataFrame, predicted: np.ndarray, fold: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": validation["id"].astype(str).to_numpy(),
            "fold": fold,
            "true_label": validation["severity"].to_numpy(),
            "predicted_label": predicted,
        }
    )


def _evaluate_hierarchy(frame: pd.DataFrame, folds: tuple[FrozenFold, ...], name: str):
    mapping, groups = hierarchy_mapping(name), HIERARCHIES[name]
    fold_rows, per_class, level1_rows, level2_rows, predictions = [], [], [], [], {}
    for method in ("hard", "soft", "oracle"):
        predictions[method] = []
    for fold in folds:
        train, validation = frame.iloc[list(fold.train)], frame.iloc[list(fold.validation)]
        x_validation = validation.drop(columns=["severity"])
        train_groups = train.severity.map(mapping)
        true_groups = validation.severity.map(mapping).to_numpy()
        started, rss_before = perf_counter(), psutil.Process().memory_info().rss
        level1 = fit_node(train, train_groups)
        predicted_groups = level1.predict(x_validation)
        group_probabilities = level1.predict_proba(x_validation)
        child_predictions, child_probabilities, child_classes = {}, {}, {}
        models = [level1]
        for group, labels in groups.items():
            subset = train[train.severity.isin(labels)]
            child = fit_node(subset, subset.severity)
            models.append(child)
            child_predictions[group] = child.predict(x_validation)
            child_probabilities[group] = child.predict_proba(x_validation)
            child_classes[group] = child.named_steps["clf"].classes_
            valid_subset = validation[validation.severity.isin(labels)]
            node_pred = child.predict(valid_subset.drop(columns=["severity"]))
            node_metrics = _metrics(valid_subset.severity.to_numpy(), node_pred, list(labels))
            level2_rows.append(
                {
                    "hierarchy": name,
                    "fold": fold.fold,
                    "node": group,
                    "training_support": len(subset),
                    "validation_support": len(valid_subset),
                    "training_distribution": json.dumps(
                        subset.severity.value_counts().to_dict(), sort_keys=True
                    ),
                    "validation_distribution": json.dumps(
                        valid_subset.severity.value_counts().to_dict(), sort_keys=True
                    ),
                    "class_weights": json.dumps(
                        calculated_class_weights(subset.severity), sort_keys=True
                    ),
                    "feature_count": _feature_count(child),
                    **node_metrics,
                }
            )
        level1_metrics = _metrics(true_groups, predicted_groups, list(groups))
        level1_rows.append(
            {
                "hierarchy": name,
                "fold": fold.fold,
                "training_support": len(train),
                "validation_support": len(validation),
                "training_distribution": json.dumps(
                    train_groups.value_counts().to_dict(), sort_keys=True
                ),
                "validation_distribution": json.dumps(
                    pd.Series(true_groups).value_counts().to_dict(), sort_keys=True
                ),
                "class_weights": json.dumps(calculated_class_weights(train_groups), sort_keys=True),
                "feature_count": _feature_count(level1),
                **level1_metrics,
            }
        )
        hard = hard_route(predicted_groups, child_predictions)
        scores = compose_soft_scores(
            group_probabilities,
            level1.named_steps["clf"].classes_,
            child_probabilities,
            child_classes,
        )
        soft = np.asarray(LABELS)[np.argmax(scores, axis=1)]
        oracle = oracle_route(true_groups, child_predictions)
        elapsed = perf_counter() - started
        for method, predicted in (("hard", hard), ("soft", soft), ("oracle", oracle)):
            metrics = _metrics(validation.severity.to_numpy(), predicted)
            method_name = f"{name}_{method}"
            fold_rows.append(
                {
                    "method": method_name,
                    "hierarchy": name,
                    "routing": method,
                    "deployable": method != "oracle",
                    "fold": fold.fold,
                    **metrics,
                    "runtime_seconds": elapsed,
                    "feature_count": sum(_feature_count(model) for model in models),
                    "model_size_bytes": sum(len(pickle.dumps(model)) for model in models),
                    "rss_delta_bytes": psutil.Process().memory_info().rss - rss_before,
                }
            )
            safe = _safe_predictions(validation, predicted, fold.fold)
            safe["true_group"] = true_groups
            safe["predicted_group"] = predicted_groups if method != "oracle" else true_groups
            safe["error_type"] = decompose_errors(
                validation.severity.to_numpy(), predicted, safe.predicted_group.to_numpy(), name
            )
            safe["artifact_status"] = (
                "NON-DEPLOYABLE DIAGNOSTIC ONLY"
                if method == "oracle"
                else "deployable development OOF"
            )
            predictions[method].append(safe)
            for label in LABELS:
                per_class.append(
                    {
                        "method": method_name,
                        "fold": fold.fold,
                        "severity": label,
                        "precision": metrics[f"precision_{label}"],
                        "recall": metrics[f"recall_{label}"],
                        "f1": metrics[f"f1_{label}"],
                        "support": metrics[f"support_{label}"],
                    }
                )
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(per_class),
        pd.DataFrame(level1_rows),
        pd.DataFrame(level2_rows),
        {method: pd.concat(parts, ignore_index=True) for method, parts in predictions.items()},
    )


def _aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in folds.groupby("method"):
        row = {
            "method": method,
            "deployable": bool(group.deployable.iloc[0]) if "deployable" in group else True,
            "mean_macro_f1": group.macro_f1.mean(),
            "std_macro_f1": group.macro_f1.std(ddof=0),
            "mean_macro_precision": group.macro_precision.mean(),
            "mean_macro_recall": group.macro_recall.mean(),
            "mean_balanced_accuracy": group.balanced_accuracy.mean(),
            "mean_weighted_f1": group.weighted_f1.mean(),
            "mean_accuracy": group.accuracy.mean(),
            "minimum_per_class_recall": min(group[f"recall_{label}"].mean() for label in LABELS),
            "mean_blocker_critical_recall": np.mean(
                [group.recall_blocker.mean(), group.recall_critical.mean()]
            ),
            "mean_blocker_critical_f1": np.mean(
                [group.f1_blocker.mean(), group.f1_critical.mean()]
            ),
            "runtime_seconds": group.runtime_seconds.sum(),
            "mean_feature_count": group.feature_count.mean(),
            "mean_model_size_bytes": group.model_size_bytes.mean(),
            "mean_rss_delta_bytes": group.rss_delta_bytes.mean()
            if "rss_delta_bytes" in group
            else np.nan,
        }
        for label in LABELS:
            row[f"mean_recall_{label}"] = group[f"recall_{label}"].mean()
            row[f"mean_f1_{label}"] = group[f"f1_{label}"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False)


def _write_matrix(output: Path, stem: str, oof: pd.DataFrame, labels: list[str] = LABELS):
    matrix = pd.DataFrame(
        confusion_matrix(oof.true_label, oof.predicted_label, labels=labels),
        index=labels,
        columns=labels,
    )
    write_csv(output / f"{stem}_confusion_matrix.csv", matrix.reset_index(names="true_label"))
    normalized = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    write_csv(
        output / f"{stem}_confusion_matrix_normalized.csv",
        normalized.reset_index(names="true_label"),
    )
    save_confusion_matrix_figure(
        output / f"{stem}_confusion_matrix.png", matrix, stem.replace("_", " ")
    )


def _error_outputs(output: Path, all_predictions: dict[str, pd.DataFrame]) -> None:
    summary, by_class, within = [], [], []
    for method, oof in all_predictions.items():
        total = len(oof)
        counts = oof.error_type.value_counts()
        summary.append(
            {
                "method": method,
                "correct_count": int(counts.get("correct", 0)),
                "correct_percentage": counts.get("correct", 0) / total,
                "routing_error_count": int(counts.get("routing_error", 0)),
                "routing_error_percentage": counts.get("routing_error", 0) / total,
                "within_group_error_count": int(counts.get("within_group_error", 0)),
                "within_group_error_percentage": counts.get("within_group_error", 0) / total,
            }
        )
        routing = oof[oof.error_type == "routing_error"]
        for (label, group), count in (
            routing.groupby(["true_label", "predicted_group"]).size().items()
        ):
            by_class.append(
                {"method": method, "true_label": label, "predicted_group": group, "errors": count}
            )
        inside = oof[oof.error_type == "within_group_error"]
        for (group, truth, predicted), count in (
            inside.groupby(["true_group", "true_label", "predicted_label"]).size().items()
        ):
            within.append(
                {
                    "method": method,
                    "child_node": group,
                    "true_label": truth,
                    "predicted_label": predicted,
                    "errors": count,
                }
            )
    write_csv(output / "routing_error_summary.csv", pd.DataFrame(summary))
    write_csv(output / "routing_errors_by_class.csv", pd.DataFrame(by_class))
    write_csv(output / "within_group_errors.csv", pd.DataFrame(within))


def _sensitivity(frame: pd.DataFrame, folds: tuple[FrozenFold, ...]) -> pd.DataFrame:
    excluded = frame.id.astype(str).isin(CONFLICT_IDS)
    rows = []
    for fold in folds:
        train_indices = tuple(index for index in fold.train if not excluded.iloc[index])
        validation_indices = tuple(index for index in fold.validation if not excluded.iloc[index])
        subset_fold = (FrozenFold(fold.fold, train_indices, validation_indices),)
        flat, _ = _evaluate_flat(frame, subset_fold)
        rows.append(
            {"method": "flat_baseline", "fold": fold.fold, "macro_f1": flat.iloc[0].macro_f1}
        )
        for name in HIERARCHIES:
            evaluated, _, _, _, _ = _evaluate_hierarchy(frame, subset_fold, name)
            for _, result in evaluated[evaluated.routing.isin(["hard", "soft"])].iterrows():
                rows.append(
                    {"method": result.method, "fold": fold.fold, "macro_f1": result.macro_f1}
                )
    result = pd.DataFrame(rows)
    result["diagnostic_only"] = True
    result["excluded_known_conflict_rows"] = int(excluded.sum())
    return result


def run_hierarchical_study(development_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    ensure_directory(output)
    frame = load_hierarchical_development(development_path)
    folds, identity = verify_folds(frame)
    write_json(output / "fold_identity_verification.json", identity)
    write_json(
        output / "study_manifest.json",
        {
            "dataset_sha256": sha256_file(development_path),
            "development_rows": len(frame),
            "labels": LABELS,
            "hierarchies": HIERARCHIES,
            "fold_hash": EXPECTED_FOLD_HASH,
            "protocol_frozen": True,
            "held_out_test_accessed": False,
            "other_project_trained": False,
        },
    )
    flat, flat_oof = _evaluate_flat(frame, folds)
    flat_mean = flat.macro_f1.mean()
    baseline_ok = abs(flat_mean - BASELINE) <= BASELINE_TOLERANCE
    write_csv(output / "flat_baseline_reproduction.csv", flat)
    if not baseline_ok:
        raise RuntimeError(f"Flat baseline reproduction {flat_mean:.6f} exceeds frozen tolerance")

    flat_per_class = []
    for _, row in flat.iterrows():
        for label in LABELS:
            flat_per_class.append(
                {
                    "method": "flat_baseline",
                    "fold": row.fold,
                    "severity": label,
                    "precision": row[f"precision_{label}"],
                    "recall": row[f"recall_{label}"],
                    "f1": row[f"f1_{label}"],
                    "support": row[f"support_{label}"],
                }
            )
    fold_parts, class_parts, level1_parts, level2_parts = (
        [flat],
        [pd.DataFrame(flat_per_class)],
        [],
        [],
    )
    all_predictions: dict[str, pd.DataFrame] = {}
    for name in HIERARCHIES:
        results, classes, level1, level2, predictions = _evaluate_hierarchy(frame, folds, name)
        fold_parts.append(results)
        class_parts.append(classes)
        level1_parts.append(level1)
        level2_parts.append(level2)
        for routing, oof in predictions.items():
            method = f"{name}_{routing}"
            all_predictions[method] = oof
            write_csv(output / f"{method}_oof_predictions.csv", oof)
            if routing in {"hard", "soft"}:
                _write_matrix(output, method, oof)
        level1_oof = predictions["hard"][["true_group", "predicted_group"]].rename(
            columns={"true_group": "true_label", "predicted_group": "predicted_label"}
        )
        _write_matrix(output, f"{name}_level1", level1_oof, list(HIERARCHIES[name]))

    fold_results = pd.concat(fold_parts, ignore_index=True)
    per_class = pd.concat(class_parts, ignore_index=True)
    level1 = pd.concat(level1_parts, ignore_index=True)
    level2 = pd.concat(level2_parts, ignore_index=True)
    comparison = _aggregate(fold_results)
    write_csv(output / "fold_results.csv", fold_results)
    write_csv(output / "per_class_results.csv", per_class)
    write_csv(output / "level1_results.csv", level1)
    write_csv(output / "level2_node_results.csv", level2)
    write_csv(output / "hierarchy_comparison.csv", comparison)
    distributions = []
    for method, oof in {"flat_baseline": flat_oof, **all_predictions}.items():
        for label in LABELS:
            distributions.append(
                {
                    "method": method,
                    "severity": label,
                    "predictions": int((oof.predicted_label == label).sum()),
                }
            )
    write_csv(output / "predicted_class_distributions.csv", pd.DataFrame(distributions))
    _error_outputs(output, all_predictions)
    sensitivity = _sensitivity(frame, folds)
    write_csv(output / "duplicate_conflict_sensitivity.csv", sensitivity)
    write_csv(
        output / "runtime_summary.csv",
        comparison[
            [
                "method",
                "runtime_seconds",
                "mean_feature_count",
                "mean_model_size_bytes",
                "mean_rss_delta_bytes",
            ]
        ],
    )

    flat_bc = comparison.loc[
        comparison.method == "flat_baseline", "mean_blocker_critical_recall"
    ].iloc[0]
    deployable = comparison[comparison.deployable & (comparison.method != "flat_baseline")].copy()
    deployable["primary_met"] = deployable.mean_macro_f1 >= PRIMARY_THRESHOLD
    deployable["bc_recall_gain"] = deployable.mean_blocker_critical_recall - flat_bc
    deployable["max_recall_loss"] = deployable.apply(
        lambda row: max(
            comparison.loc[comparison.method == "flat_baseline", f"mean_recall_{label}"].iloc[0]
            - row[f"mean_recall_{label}"]
            for label in LABELS
        ),
        axis=1,
    )
    deployable["minority_benefit_met"] = (
        (deployable.mean_macro_f1 >= SECONDARY_FLOOR)
        & (deployable.bc_recall_gain >= 0.05)
        & (deployable.max_recall_loss <= 0.10)
    )
    write_csv(
        output / "hierarchy_comparison.csv",
        comparison.merge(
            deployable[
                [
                    "method",
                    "primary_met",
                    "bc_recall_gain",
                    "max_recall_loss",
                    "minority_benefit_met",
                ]
            ],
            on="method",
            how="left",
        ),
    )
    decision = (
        "primary"
        if deployable.primary_met.any()
        else ("secondary" if deployable.minority_benefit_met.any() else "no_improvement")
    )
    lines = [
        "# MYLYN hierarchical study summary",
        "",
        f"Flat baseline reproduced at mean macro-F1 `{flat_mean:.6f}`.",
        f"Decision: `{decision}`. Held-out test accessed: `false`.",
        "",
        "```csv\n" + comparison.to_csv(index=False).strip() + "\n```",
    ]
    (output / "hierarchical_study_summary.md").write_text("\n".join(lines), encoding="utf-8")
    error_summary = pd.read_csv(output / "routing_error_summary.csv")
    (output / "hierarchical_error_analysis.md").write_text(
        "# Hierarchical development OOF error analysis\n\n"
        "Oracle rows are NON-DEPLOYABLE DIAGNOSTIC ONLY.\n\n"
        + "```csv\n"
        + error_summary.to_csv(index=False).strip()
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
    args = parser.parse_args()
    print(json.dumps(run_hierarchical_study(args.development, args.output), indent=2, default=str))
