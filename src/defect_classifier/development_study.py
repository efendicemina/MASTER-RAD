"""Controlled, development-only MYLYN model improvement study."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .preprocessing import TextCombiner
from .reporting import save_confusion_matrix_figure
from .splitting import PurgedTimeSeriesSplit
from .utils import ensure_directory, write_csv, write_json

LABELS = ["blocker", "critical", "major", "normal", "minor", "trivial"]
INPUT_VARIANTS = ["summary", "description", "summary_description"]
FEATURE_VARIANTS = ["word_1_1", "word_1_2", "char_wb", "word_char"]
EMAIL = re.compile(r"\S*@\S*")
URL = re.compile(r"https?://\S*|www\.\S+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FrozenFold:
    fold: int
    train: tuple[int, ...]
    validation: tuple[int, ...]


def load_development_only(path: str | Path) -> pd.DataFrame:
    """Load only an explicitly named development split and normalize its order."""

    source = Path(path)
    if source.name != "development_split.csv":
        raise ValueError("Development study accepts only development_split.csv")
    frame = pd.read_csv(source)
    frame["creation_time"] = pd.to_datetime(frame["creation_time"], utc=True, errors="raise")
    return frame.sort_values("creation_time", kind="mergesort").reset_index(drop=True)


def freeze_temporal_folds(frame: pd.DataFrame, n_splits: int = 3) -> tuple[FrozenFold, ...]:
    """Materialize deterministic, duplicate-purged temporal folds once."""

    splitter = PurgedTimeSeriesSplit(n_splits=n_splits)
    features = frame.drop(columns=["severity"])
    return tuple(
        FrozenFold(fold, tuple(map(int, train)), tuple(map(int, validation)))
        for fold, (train, validation) in enumerate(splitter.split(features), start=1)
    )


def folds_fingerprint(folds: tuple[FrozenFold, ...]) -> str:
    payload = json.dumps(
        [{"fold": f.fold, "train": f.train, "validation": f.validation} for f in folds]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_study_pipeline(
    input_variant: str,
    feature_variant: str,
    model_name: str = "LogisticRegression",
    class_weight: None | str | dict[str, float] = "balanced",
    c_value: float = 1.0,
) -> Pipeline:
    """Build a bounded sparse pipeline for an approved ablation configuration."""

    if input_variant not in INPUT_VARIANTS:
        raise ValueError(f"Unsupported input variant: {input_variant}")
    text = TextCombiner(
        "summary",
        "description",
        input_variant,
        {
            "lowercase": True,
            "remove_html": True,
            "replace_urls": True,
            "replace_emails": True,
            "remove_code_blocks": False,
            "remove_stack_traces": False,
            "normalize_unicode": True,
            "normalize_whitespace": True,
        },
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.98,
        max_features=35_000,
        sublinear_tf=True,
    )
    if feature_variant == "word_1_1":
        vectorizer: Any = word
    elif feature_variant == "word_1_2":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=50_000,
            sublinear_tf=True,
        )
    elif feature_variant == "char_wb":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_df=0.995,
            max_features=40_000,
            sublinear_tf=True,
        )
    elif feature_variant == "word_char":
        vectorizer = FeatureUnion(
            [
                ("word", word),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_df=0.995,
                        max_features=30_000,
                        sublinear_tf=True,
                    ),
                ),
            ],
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported feature variant: {feature_variant}")
    classifier: Any
    if model_name == "DummyClassifier":
        classifier = DummyClassifier(strategy="most_frequent")
    elif model_name == "MultinomialNB":
        classifier = MultinomialNB(alpha=0.5)
    elif model_name == "LogisticRegression":
        classifier = LogisticRegression(
            C=c_value, class_weight=class_weight, max_iter=2000, random_state=42
        )
    elif model_name == "LinearSVC":
        classifier = LinearSVC(C=c_value, class_weight=class_weight, random_state=42)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("text", text), ("features", vectorizer), ("clf", classifier)])


def validate_custom_class_weight(weights: dict[str, float]) -> dict[str, float]:
    if set(weights) != set(LABELS):
        raise ValueError("Custom class weights must define exactly the six fixed labels")
    normalized = {label: float(weights[label]) for label in LABELS}
    if any(not np.isfinite(value) or value <= 0 or value > 10 for value in normalized.values()):
        raise ValueError("Custom class weights must be finite and in (0, 10]")
    return normalized


def derived_class_weights(target: pd.Series) -> dict[str, dict[str, float]]:
    """Derive conservative candidates from development labels only."""

    counts = target.value_counts()
    balanced = {label: len(target) / (len(LABELS) * counts[label]) for label in LABELS}
    sqrt_weights = {label: float(np.sqrt(value)) for label, value in balanced.items()}
    capped = {label: float(min(value, 3.0)) for label, value in balanced.items()}
    return {
        "sqrt_inverse_frequency": validate_custom_class_weight(sqrt_weights),
        "capped_balanced_3": validate_custom_class_weight(capped),
    }


def evaluate_folds(
    frame: pd.DataFrame,
    folds: tuple[FrozenFold, ...],
    pipeline: Pipeline,
    configuration: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline]:
    """Evaluate one configuration on frozen folds and return OOF predictions."""

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    rss_before = psutil.Process().memory_info().rss
    started = perf_counter()
    last_model: Pipeline | None = None
    for fold in folds:
        train = frame.iloc[list(fold.train)]
        validation = frame.iloc[list(fold.validation)]
        fitted = clone(pipeline)
        fit_start = perf_counter()
        fitted.fit(train.drop(columns=["severity"]), train["severity"])
        fit_seconds = perf_counter() - fit_start
        infer_start = perf_counter()
        predicted = fitted.predict(validation.drop(columns=["severity"]))
        infer_seconds = perf_counter() - infer_start
        precision, recall, per_f1, _ = precision_recall_fscore_support(
            validation["severity"], predicted, labels=LABELS, zero_division=0
        )
        row: dict[str, Any] = {
            "fold": fold.fold,
            "validation_start": validation["creation_time"].min().isoformat(),
            "validation_end": validation["creation_time"].max().isoformat(),
            "macro_f1": f1_score(
                validation["severity"],
                predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(validation["severity"], predicted),
            "weighted_f1": f1_score(
                validation["severity"],
                predicted,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            ),
            "accuracy": accuracy_score(validation["severity"], predicted),
            "fit_seconds": fit_seconds,
            "inference_seconds": infer_seconds,
        }
        for index, label in enumerate(LABELS):
            row[f"precision_{label}"] = precision[index]
            row[f"recall_{label}"] = recall[index]
            row[f"f1_{label}"] = per_f1[index]
            row[f"predicted_{label}"] = int(np.sum(predicted == label))
        rows.append(row)
        fold_predictions = validation[
            ["id", "creation_time", "severity", "summary", "description", "product", "component"]
        ].copy()
        fold_predictions["predicted"] = predicted
        fold_predictions["fold"] = fold.fold
        predictions.append(fold_predictions)
        last_model = fitted
    fold_frame = pd.DataFrame(rows)
    assert last_model is not None
    feature_count = len(last_model.named_steps["features"].get_feature_names_out())
    aggregate: dict[str, Any] = {
        **configuration,
        "folds_fingerprint": folds_fingerprint(folds),
        "feature_count": feature_count,
        "mean_cv_macro_f1": fold_frame["macro_f1"].mean(),
        "std_cv_macro_f1": fold_frame["macro_f1"].std(ddof=0),
        "mean_balanced_accuracy": fold_frame["balanced_accuracy"].mean(),
        "mean_weighted_f1": fold_frame["weighted_f1"].mean(),
        "fit_seconds": fold_frame["fit_seconds"].sum(),
        "inference_seconds": fold_frame["inference_seconds"].sum(),
        "total_seconds": perf_counter() - started,
        "rss_delta_bytes": psutil.Process().memory_info().rss - rss_before,
        "model_size_bytes": len(pickle.dumps(last_model)),
    }
    for label in LABELS:
        for metric in ("precision", "recall", "f1"):
            aggregate[f"mean_{metric}_{label}"] = fold_frame[f"{metric}_{label}"].mean()
        aggregate[f"predicted_{label}"] = int(fold_frame[f"predicted_{label}"].sum())
    return aggregate, pd.concat(predictions, ignore_index=True), last_model


def chronological_learning_subsets(
    frame: pd.DataFrame, fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
) -> list[tuple[float, pd.DataFrame]]:
    if not fractions or any(value <= 0 or value > 1 for value in fractions):
        raise ValueError("Learning fractions must be in (0, 1]")
    return [
        (fraction, frame.iloc[: max(4, int(len(frame) * fraction))].copy())
        for fraction in fractions
    ]


def redact(value: Any, limit: int | None = None) -> str:
    text = URL.sub("[URL]", EMAIL.sub("[EMAIL]", "" if pd.isna(value) else str(value)))
    return text if limit is None else text[:limit]


def _write_review_files(frame: pd.DataFrame, parquet_path: Path, output: Path) -> None:
    raw = pd.read_parquet(parquet_path)
    raw = raw[raw["severity"].isin(LABELS)].copy()
    conflicts = raw.groupby("exact_text_hash")["severity"].nunique()
    conflict_hashes = conflicts[conflicts > 1].index
    review = raw[raw["exact_text_hash"].isin(conflict_hashes)].copy()
    review = review.sort_values(["exact_text_hash", "creation_time"])
    review_out = pd.DataFrame(
        {
            "global_report_key": review["global_report_key"],
            "issue_id": review["issue_id"],
            "creation_time": review["creation_time"],
            "severity": review["severity"],
            "summary": review["summary"].map(redact),
            "description": review["description"].map(lambda value: redact(value, 1000)),
            "duplicate_group_identifier": review["exact_text_hash"],
            "recommended_review_reason": "identical text has conflicting severity labels",
            "human_decision": "",
            "human_notes": "",
        }
    )
    write_csv(output / "conflicting_duplicate_review.csv", review_out)
    minority = frame[frame["severity"].isin(["blocker", "critical", "minor", "trivial"])].copy()
    minority["time_period"] = pd.qcut(
        minority["creation_time"].rank(method="first"), 3, labels=["early", "middle", "late"]
    )
    sample = pd.concat(
        [
            group.sample(min(10, len(group)), random_state=42)
            for _, group in minority.groupby(["severity", "time_period"], observed=True)
        ],
        ignore_index=True,
    )
    sample["summary"] = sample["summary"].map(redact)
    sample["description"] = sample["description"].map(lambda value: redact(value, 1000))
    columns = [
        "id",
        "creation_time",
        "severity",
        "time_period",
        "summary",
        "description",
        "duplicate_group",
        "human_label_valid",
        "human_notes",
    ]
    sample["human_label_valid"] = ""
    sample["human_notes"] = ""
    write_csv(output / "minority_class_review_sample.csv", sample[columns])


def _learning_curves(frame: pd.DataFrame, output: Path) -> None:
    rows, per_class = [], []
    baseline = build_study_pipeline("summary_description", "word_1_2")
    for fraction, subset in chronological_learning_subsets(frame):
        folds = freeze_temporal_folds(subset)
        result, _, _ = evaluate_folds(subset, folds, baseline, {"training_fraction": fraction})
        counts = subset["severity"].value_counts()
        row = {
            "training_fraction": fraction,
            "available_reports": len(subset),
            "mean_cv_macro_f1": result["mean_cv_macro_f1"],
            "std_cv_macro_f1": result["std_cv_macro_f1"],
            "balanced_accuracy": result["mean_balanced_accuracy"],
            "training_duration_seconds": result["fit_seconds"],
            "feature_count": result["feature_count"],
        }
        for label in LABELS:
            row[f"count_{label}"] = int(counts.get(label, 0))
            per_class.append(
                {
                    "training_fraction": fraction,
                    "severity": label,
                    "mean_cv_f1": result[f"mean_f1_{label}"],
                }
            )
        rows.append(row)
    curve = pd.DataFrame(rows)
    write_csv(output / "learning_curve.csv", curve)
    write_csv(output / "learning_curve_per_class.csv", pd.DataFrame(per_class))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["available_reports"], curve["mean_cv_macro_f1"], marker="o")
    ax.set(xlabel="Chronological development reports", ylabel="Mean CV macro F1")
    fig.tight_layout()
    fig.savefig(output / "learning_curve_macro_f1.png", dpi=200)
    plt.close(fig)


def _temporal_drift(frame: pd.DataFrame, fold_rows: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    data["year"] = data["creation_time"].dt.year
    data["text_length"] = (
        data["summary"].fillna("") + " " + data["description"].fillna("")
    ).str.len()
    yearly = (
        data.groupby("year")
        .agg(
            reports=("severity", "size"),
            mean_text_length=("text_length", "mean"),
            median_text_length=("text_length", "median"),
            products=("product", "nunique"),
            components=("component", "nunique"),
        )
        .reset_index()
    )
    distributions = []
    for year, group in data.groupby("year"):
        distributions.append(
            {
                "year": year,
                "source_project": "MYLYN",
                "product_distribution": json.dumps(
                    group["product"].fillna("[missing]").value_counts().to_dict(),
                    sort_keys=True,
                ),
                "component_distribution": json.dumps(
                    group["component"].fillna("[missing]").value_counts().to_dict(),
                    sort_keys=True,
                ),
            }
        )
    yearly = yearly.merge(pd.DataFrame(distributions), on="year", how="left")
    severity = data.groupby(["year", "severity"]).size().unstack(fill_value=0).reset_index()
    yearly = yearly.merge(severity, on="year", how="left")
    vocab = []
    previous: set[str] | None = None
    for year, group in data.groupby("year"):
        tokens = set(" ".join(group["summary"].fillna("").str.lower()).split())
        overlap = (
            np.nan if previous is None else len(tokens & previous) / max(len(tokens | previous), 1)
        )
        vocab.append(
            {"year": year, "summary_vocabulary": len(tokens), "jaccard_vs_previous": overlap}
        )
        previous = tokens
    yearly = yearly.merge(pd.DataFrame(vocab), on="year", how="left")
    write_csv(output / "temporal_drift_summary.csv", yearly)
    write_csv(output / "temporal_cv_performance.csv", fold_rows)
    lines = [
        "# MYLYN temporal drift",
        "",
        "Development period: "
        f"{data.creation_time.min().date()} to {data.creation_time.max().date()}.",
        f"Years covered: {data.year.nunique()}; reports: {len(data)}.",
        "",
        f"Annual volume peaks in {int(yearly.loc[yearly.reports.idxmax(), 'year'])} "
        f"at {int(yearly.reports.max())} reports.",
        f"Mean text length ranges from {yearly.mean_text_length.min():.1f} to "
        f"{yearly.mean_text_length.max():.1f} characters by year.",
        "Adjacent-year Summary vocabulary Jaccard ranges from "
        f"{yearly.jaccard_vs_previous.min():.3f} to "
        f"{yearly.jaccard_vs_previous.max():.3f}.",
        "",
        "Vocabulary Jaccard similarity compares each year's Summary vocabulary "
        "with the previous year.",
        "Product and Component are used only as aggregated diagnostics, never predictive features.",
    ]
    (output / "temporal_drift.md").write_text("\n".join(lines), encoding="utf-8")


def _development_error_analysis(oof: pd.DataFrame, model: Pipeline, output: Path) -> None:
    true, predicted = oof["severity"], oof["predicted"]
    matrix = pd.DataFrame(
        confusion_matrix(true, predicted, labels=LABELS), index=LABELS, columns=LABELS
    )
    normalized = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    write_csv(output / "development_oof_confusion_matrix.csv", matrix)
    write_csv(output / "development_oof_confusion_matrix_normalized.csv", normalized)
    save_confusion_matrix_figure(
        output / "development_oof_confusion_matrix.png", matrix, "Development OOF confusion matrix"
    )
    errors = oof[true != predicted].copy()
    errors["year"] = errors["creation_time"].dt.year
    errors["text_length"] = (
        errors["summary"].fillna("") + errors["description"].fillna("")
    ).str.len()
    errors["text_length_group"] = pd.cut(
        errors["text_length"], [-1, 200, 1000, np.inf], labels=["short", "medium", "long"]
    )
    for column in ["severity", "predicted", "text_length_group", "year", "product", "component"]:
        table = errors.groupby(column, dropna=False).size().reset_index(name="errors")
        write_csv(output / f"development_errors_by_{column}.csv", table)
    confusions = (
        errors.groupby(["severity", "predicted"])
        .size()
        .reset_index(name="errors")
        .sort_values("errors", ascending=False)
    )
    write_csv(output / "development_common_confusions.csv", confusions)
    examples = errors.sort_values(["severity", "creation_time"]).groupby("severity").head(10).copy()
    examples["summary"] = examples["summary"].map(redact)
    examples["description"] = examples["description"].map(lambda value: redact(value, 1000))
    write_csv(output / "development_representative_errors.csv", examples)
    classifier = model.named_steps["clf"]
    if hasattr(classifier, "coef_"):
        names = model.named_steps["features"].get_feature_names_out()
        feature_rows = []
        for label, coefficients in zip(classifier.classes_, classifier.coef_, strict=True):
            for rank, index in enumerate(coefficients.argsort()[-30:][::-1], start=1):
                feature_rows.append(
                    {
                        "severity": label,
                        "rank": rank,
                        "feature": names[index],
                        "coefficient": coefficients[index],
                    }
                )
        write_csv(output / "development_influential_features.csv", pd.DataFrame(feature_rows))


def run_study(development_path: Path, parquet_path: Path, output: Path) -> dict[str, Any]:
    """Execute all approved comparisons without accepting any held-out-test input."""

    ensure_directory(output)
    frame = load_development_only(development_path)
    folds = freeze_temporal_folds(frame)
    write_json(
        output / "study_manifest.json",
        {
            "development_path": str(development_path),
            "development_rows": len(frame),
            "folds_fingerprint": folds_fingerprint(folds),
            "labels": LABELS,
            "held_out_test_accessed": False,
        },
    )
    _write_review_files(frame, parquet_path, output)
    _learning_curves(frame, output)

    ablation_rows = []
    for input_variant in INPUT_VARIANTS:
        for feature_variant in FEATURE_VARIANTS:
            result, _, _ = evaluate_folds(
                frame,
                folds,
                build_study_pipeline(input_variant, feature_variant),
                {
                    "input_variant": input_variant,
                    "feature_variant": feature_variant,
                    "model": "LogisticRegression",
                    "class_weight": "balanced",
                },
            )
            ablation_rows.append(result)
    ablation = pd.DataFrame(ablation_rows).sort_values(
        ["mean_cv_macro_f1", "std_cv_macro_f1"], ascending=[False, True]
    )
    write_csv(output / "feature_ablation.csv", ablation)
    best_feature = ablation.iloc[0]
    best_input, best_variant = best_feature["input_variant"], best_feature["feature_variant"]

    weight_rows = []
    weights: list[tuple[str, Any]] = [("none", None), ("balanced", "balanced")]
    weights.extend(derived_class_weights(frame["severity"]).items())
    for model_name in ["LogisticRegression", "LinearSVC"]:
        for weight_name, weight in weights:
            result, _, _ = evaluate_folds(
                frame,
                folds,
                build_study_pipeline(best_input, best_variant, model_name, weight),
                {
                    "model": model_name,
                    "class_weight": weight_name,
                    "input_variant": best_input,
                    "feature_variant": best_variant,
                },
            )
            weight_rows.append(result)
    weight_comparison = pd.DataFrame(weight_rows).sort_values("mean_cv_macro_f1", ascending=False)
    write_csv(output / "class_weight_comparison.csv", weight_comparison)

    best_weight_row = weight_comparison.iloc[0]
    chosen_weight_name = best_weight_row["class_weight"]
    chosen_weight = dict(weights)[chosen_weight_name]
    model_rows, model_outputs = [], {}
    for model_name in ["DummyClassifier", "MultinomialNB", "LogisticRegression", "LinearSVC"]:
        applicable_weight = (
            chosen_weight if model_name in {"LogisticRegression", "LinearSVC"} else None
        )
        result, oof, fitted = evaluate_folds(
            frame,
            folds,
            build_study_pipeline(best_input, best_variant, model_name, applicable_weight),
            {
                "model": model_name,
                "class_weight": chosen_weight_name if applicable_weight is not None else "n/a",
                "input_variant": best_input,
                "feature_variant": best_variant,
            },
        )
        model_rows.append(result)
        model_outputs[model_name] = (oof, fitted)
    models = pd.DataFrame(model_rows).sort_values(
        ["mean_cv_macro_f1", "std_cv_macro_f1"], ascending=[False, True]
    )
    write_csv(output / "model_comparison.csv", models)
    selected = models.iloc[0]
    selected_oof, selected_model = model_outputs[selected["model"]]
    temporal_rows = []
    for fold, group in selected_oof.groupby("fold"):
        metrics = {
            "fold": fold,
            "validation_start": group.creation_time.min(),
            "validation_end": group.creation_time.max(),
        }
        metrics["macro_f1"] = f1_score(
            group.severity, group.predicted, labels=LABELS, average="macro", zero_division=0
        )
        _, recall, _, _ = precision_recall_fscore_support(
            group.severity, group.predicted, labels=LABELS, zero_division=0
        )
        metrics.update({f"recall_{label}": recall[index] for index, label in enumerate(LABELS)})
        temporal_rows.append(metrics)
    temporal_frame = pd.DataFrame(temporal_rows)
    _temporal_drift(frame, temporal_frame, output)
    _development_error_analysis(selected_oof, selected_model, output)
    write_json(output / "selected_challenger.json", selected.to_dict())
    return selected.to_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_study(arguments.development, arguments.parquet, arguments.output),
            indent=2,
            default=str,
        )
    )
