"""Training and experiment orchestration."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import psutil
from sklearn.metrics import f1_score

from .config import load_config
from .data_loading import file_fingerprint, load_raw_csv, resolve_columns
from .data_validation import (
    add_duplicate_group_column,
    coerce_dates,
    drop_exact_duplicate_reports,
    filter_dataset,
    inspect_dataset,
)
from .error_analysis import build_error_analysis
from .evaluation import bootstrap_macro_f1_ci, evaluate_predictions
from .models import build_pipeline, build_search, supported_models
from .pilot_preparation import prepare_pilot_dataframe
from .reporting import (
    make_experiment_dir,
    save_confusion_matrix_figure,
    save_frame,
    save_joblib,
    save_payload,
)
from .splitting import (
    SplitResult,
    chronological_holdout,
    cv_fold_distribution,
    purge_duplicate_group_overlap,
    stratified_random_holdout,
)
from .utils import (
    git_commit_hash,
    package_versions,
    safe_filename,
    system_metadata,
    utc_now_iso,
    write_csv,
    write_json,
)


@dataclass(slots=True)
class TrainingOutcome:
    experiment_dir: Path
    model_path: Path
    split_path: Path
    metadata_path: Path
    selected_model: str
    selected_params: dict[str, Any]
    labels: list[str]


@dataclass(frozen=True, slots=True)
class FixedLabelMacroF1Scorer:
    """Score predictions against a stable class set, including temporarily absent classes."""

    labels: tuple[str, ...]

    def __call__(self, estimator, features, target) -> float:
        predictions = estimator.predict(features)
        return float(
            f1_score(
                target,
                predictions,
                labels=list(self.labels),
                average="macro",
                zero_division=0,
            )
        )


def prepare_dataframe(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    """Load, normalize, and filter the dataset."""

    raw_path = Path(config["dataset"]["path"])
    frame = load_raw_csv(raw_path)
    resolved_frame, resolved_columns = resolve_columns(frame, config["source_columns"])
    inspection = inspect_dataset(resolved_frame, config)
    cleaned = filter_dataset(resolved_frame, config)
    cleaned = coerce_dates(cleaned)
    cleaned = add_duplicate_group_column(cleaned)
    cleaned, duplicate_count = drop_exact_duplicate_reports(
        cleaned, [config["text_columns"]["summary"], config["text_columns"]["description"]]
    )
    inspection.summary["duplicate_reports_removed_before_split"] = duplicate_count
    inspection.summary["retained_rows_after_filtering"] = int(len(cleaned))
    return cleaned, resolved_columns, inspection.summary


def split_dataset(config: dict[str, Any], frame: pd.DataFrame) -> SplitResult:
    """Split the dataset according to the configured strategy."""

    split_config = config["split"]
    strategy = str(split_config["strategy"])
    if strategy == "chronological":
        result = chronological_holdout(
            frame,
            date_column="creation_time",
            target_column=config["target_column"],
            train_fraction=float(split_config["train_fraction"]),
            allow_missing_dates_fallback=bool(
                split_config.get("allow_missing_dates_fallback", False)
            ),
        )
        result, _ = purge_duplicate_group_overlap(result, "duplicate_group")
        return result
    if strategy == "stratified_random":
        result = stratified_random_holdout(
            frame,
            target_column=config["target_column"],
            train_fraction=float(split_config["train_fraction"]),
            random_state=int(split_config.get("random_state", 42)),
        )
        result, _ = purge_duplicate_group_overlap(result, "duplicate_group")
        return result
    raise ValueError(f"Unknown split strategy: {strategy}")


def train_experiment(config_path: str | Path) -> TrainingOutcome:
    """Run data preparation, model comparison, and final fitting."""

    config = load_config(config_path)
    if not config.get("training", {}).get("enabled", True):
        raise ValueError(
            "Training is disabled for this ingestion/audit configuration; "
            "create an explicit training configuration after readiness review"
        )
    preparation = None
    if config.get("training", {}).get("controlled_pilot", False):
        preparation = prepare_pilot_dataframe(config)
        cleaned_frame = preparation.frame
        inspection_summary = preparation.filtering_summary
    else:
        cleaned_frame, _, inspection_summary = prepare_dataframe(config)
    split_result = split_dataset(config, cleaned_frame)

    labels = list(config.get("target_labels", [])) or sorted(
        split_result.development[config["target_column"]].astype(str).unique().tolist()
    )
    if not labels:
        raise ValueError("No target labels remain after filtering")
    missing_development_labels = sorted(
        set(labels) - set(split_result.development[config["target_column"]])
    )
    if missing_development_labels:
        raise ValueError(f"Approved labels missing from development: {missing_development_labels}")

    timestamp = utc_now_iso().replace(":", "").replace("-", "")
    experiment_id = f"{safe_filename(Path(config_path).stem)}_{timestamp}"
    experiment_dir = make_experiment_dir(config["output"]["root_dir"], experiment_id)
    save_payload(experiment_dir / "tables" / "inspection.json", inspection_summary)
    write_json(experiment_dir / "config_snapshot.json", config)
    if preparation is not None:
        save_payload(
            experiment_dir / "metrics" / "data_filtering_summary.json",
            preparation.filtering_summary,
        )
        save_frame(
            experiment_dir / "tables" / "duplicate_removals.csv",
            preparation.duplicate_removals,
        )
        save_frame(
            experiment_dir / "tables" / "conflicting_duplicate_labels.csv",
            preparation.conflicting_duplicate_labels,
        )

    split_columns = _research_split_columns(split_result.development, config)
    persisted_development = _redact_text_frame(split_result.development[split_columns], config)
    persisted_test = _redact_text_frame(split_result.test[split_columns], config)
    write_csv(
        experiment_dir / "tables" / "development_split.csv",
        persisted_development,
    )
    write_csv(experiment_dir / "tables" / "test_split.csv", persisted_test)
    save_frame(
        experiment_dir / "tables" / "development_class_distribution.csv",
        _class_distribution(split_result.development, config["target_column"], labels),
    )
    save_frame(
        experiment_dir / "tables" / "test_class_distribution.csv",
        _class_distribution(split_result.test, config["target_column"], labels),
    )
    fold_distribution = cv_fold_distribution(
        split_result.development,
        config["target_column"],
        int(config["cv"]["n_splits"]),
    )
    save_frame(experiment_dir / "tables" / "cv_fold_distribution.csv", fold_distribution)

    scoring = _build_scorer(str(config["primary_metric"]), labels)
    best_result: dict[str, Any] | None = None
    comparison_rows: list[dict[str, Any]] = []
    dev_features = split_result.development.drop(columns=[config["target_column"]])
    dev_target = (
        split_result.development[config["target_column"]].astype(str).str.strip().str.lower()
    )
    resource_before = _resource_snapshot()
    minimum_available_gb = float(
        config.get("resources", {}).get("minimum_available_memory_gb", 2.0)
    )
    if resource_before["available_memory_gb"] < minimum_available_gb:
        raise MemoryError(
            f"Available memory {resource_before['available_memory_gb']:.2f} GiB is below "
            f"configured minimum {minimum_available_gb:.2f} GiB"
        )
    feature_estimate = _estimate_tfidf_dimensions(dev_features, config)
    save_payload(
        experiment_dir / "metrics" / "resource_estimate.json",
        {"before_training": resource_before, "tfidf_development_estimate": feature_estimate},
    )

    for model_name in supported_models(config):
        pipeline = build_pipeline(model_name, config)
        search = build_search(config, model_name, pipeline, scoring=scoring)
        start = perf_counter()
        process = psutil.Process()
        rss_before = process.memory_info().rss
        search.fit(dev_features, dev_target)
        rss_after = process.memory_info().rss
        best_score = float(search.best_score_)
        if not isfinite(best_score):
            raise ValueError(
                f"Cross-validation produced a non-finite score for {model_name}; "
                "inspect class coverage in temporal folds"
            )
        best_params = search.best_params_
        best_index = int(search.best_index_)
        score_std = float(search.cv_results_["std_test_score"][best_index])
        duration = perf_counter() - start
        cv_results = pd.DataFrame(search.cv_results_)
        save_frame(
            experiment_dir / "tables" / f"cv_results_{safe_filename(model_name)}.csv",
            cv_results,
        )
        model_features = _extract_linear_features(search.best_estimator_, labels)
        if model_features is not None:
            save_frame(
                experiment_dir
                / "tables"
                / f"influential_features_{safe_filename(model_name)}.csv",
                model_features,
            )
        comparison_rows.append(
            {
                "model": model_name,
                "cv_macro_f1": best_score,
                "cv_macro_f1_std": score_std,
                "training_seconds": duration,
                "best_params": best_params,
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "rss_delta_bytes": rss_after - rss_before,
            }
        )
        if best_result is None or best_score > best_result["score"]:
            best_result = {"model": model_name, "score": best_score, "params": best_params}

    assert best_result is not None
    selected_pipeline = build_pipeline(best_result["model"], config)
    selected_pipeline.set_params(
        **{key: value for key, value in best_result["params"].items() if "__" in key}
    )
    fit_start = perf_counter()
    selected_pipeline.fit(dev_features, dev_target)
    fit_duration = perf_counter() - fit_start

    model_path = experiment_dir / "artifacts" / "model.joblib"
    save_joblib(model_path, selected_pipeline)
    feature_table = _extract_linear_features(selected_pipeline, labels)
    if feature_table is not None:
        save_frame(experiment_dir / "tables" / "selected_model_features.csv", feature_table)

    metadata = {
        "utc_timestamp": utc_now_iso(),
        "experiment_id": experiment_id,
        "config_snapshot": config,
        "git_commit": git_commit_hash(Path.cwd()),
        "system": system_metadata(),
        "package_versions": package_versions(
            [
                "pandas",
                "numpy",
                "scikit-learn",
                "matplotlib",
                "PyYAML",
                "joblib",
                "scipy",
                "pyarrow",
                "psutil",
            ]
        ),
        "random_seed": int(config["split"].get("random_state", 42)),
        "input_file_name": str(config["dataset"]["path"]),
        "input_file_sha256": file_fingerprint(config["dataset"]["path"]),
        "raw_rows": int(len(load_raw_csv(config["dataset"]["path"]))),
        "retained_rows": int(len(cleaned_frame)),
        "development_rows": int(len(split_result.development)),
        "test_rows": int(len(split_result.test)),
        "duplicate_overlap_rows_removed": split_result.duplicate_overlap_rows_removed,
        "development_class_counts": dev_target.value_counts().sort_index().to_dict(),
        "test_class_counts": (
            split_result.test[config["target_column"]]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "selected_model": best_result["model"],
        "selected_hyperparameters": best_result["params"],
        "comparison_rows": comparison_rows,
        "training_seconds_full_dev_fit": fit_duration,
        "labels": labels,
        "filtering_summary": inspection_summary,
        "resource_before_training": resource_before,
        "resource_after_training": _resource_snapshot(),
        "tfidf_development_estimate": feature_estimate,
        "input_provenance": _input_provenance(config),
        "test_evaluation_count": 0,
    }
    metadata_path = experiment_dir / "metadata.json"
    write_json(metadata_path, metadata)
    save_frame(experiment_dir / "tables" / "model_comparison.csv", pd.DataFrame(comparison_rows))
    save_frame(experiment_dir / "tables" / "selected_labels.csv", pd.DataFrame({"label": labels}))
    return TrainingOutcome(
        experiment_dir=experiment_dir,
        model_path=model_path,
        split_path=experiment_dir,
        metadata_path=metadata_path,
        selected_model=best_result["model"],
        selected_params=best_result["params"],
        labels=labels,
    )


def evaluate_experiment(experiment_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Evaluate the saved model on the held-out test split."""

    experiment_path = Path(experiment_dir)
    metrics_path = experiment_path / "metrics" / "test_metrics.json"
    if metrics_path.exists():
        metadata = _read_json(experiment_path / "metadata.json")
        if int(metadata.get("test_evaluation_count", 0)) == 0:
            return _finalize_saved_evaluation(experiment_path, metadata)
        raise ValueError(
            "Held-out test metrics already exist; final test evaluation is intentionally one-shot"
        )
    config = load_config(config_path)
    model = _load_model(experiment_path / "artifacts" / "model.joblib")
    test_split = pd.read_csv(experiment_path / "tables" / "test_split.csv")
    test_split = coerce_dates(test_split)
    metadata = _read_json(experiment_path / "metadata.json")
    labels = list(metadata["labels"])
    features = test_split.drop(columns=[config["target_column"]])
    target = test_split[config["target_column"]].astype(str).str.strip().str.lower().tolist()

    inference_start = perf_counter()
    predictions = model.predict(features)
    inference_seconds = perf_counter() - inference_start

    evaluation = evaluate_predictions(target, list(predictions), labels)
    bootstrap = bootstrap_macro_f1_ci(
        target,
        list(predictions),
        n_resamples=int(config["bootstrap"]["n_resamples"]),
        confidence_level=float(config["bootstrap"]["confidence_level"]),
        random_state=int(config["bootstrap"]["random_state"]),
        labels=labels,
    )
    evaluation.metrics["bootstrap_macro_f1_ci"] = bootstrap
    evaluation.metrics["inference_seconds"] = inference_seconds

    predictions_frame = pd.DataFrame(
        {
            "report_id": test_split["id"]
            if "id" in test_split.columns
            else test_split.index.astype(str),
            "true_label": target,
            "predicted_label": list(predictions),
            "correct": [
                str(true) == str(pred) for true, pred in zip(target, predictions, strict=True)
            ],
            "summary": test_split[config["text_columns"]["summary"]],
            "description": test_split[config["text_columns"]["description"]]
            .astype(str)
            .str.slice(0, 1000),
        }
    )
    save_frame(experiment_path / "predictions" / "test_predictions.csv", predictions_frame)
    save_payload(metrics_path, evaluation.metrics)
    save_payload(
        experiment_path / "metrics" / "classification_report.json",
        evaluation.classification_report,
    )
    report_frame = pd.DataFrame(evaluation.classification_report).transpose().reset_index()
    report_frame = report_frame.rename(columns={"index": "label"})
    save_frame(experiment_path / "tables" / "classification_report.csv", report_frame)
    save_frame(experiment_path / "tables" / "confusion_matrix.csv", evaluation.confusion_matrix)
    save_frame(
        experiment_path / "tables" / "confusion_matrix_normalized.csv",
        evaluation.normalized_confusion_matrix,
    )
    finalized = _finalize_saved_evaluation(experiment_path, metadata)
    finalized["evaluation"] = evaluation
    return finalized


def _finalize_saved_evaluation(
    experiment_path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Finish reporting from persisted predictions without repeating test inference."""

    predictions_frame = pd.read_csv(experiment_path / "predictions" / "test_predictions.csv")
    metrics = _read_json(experiment_path / "metrics" / "test_metrics.json")
    confusion = pd.read_csv(experiment_path / "tables" / "confusion_matrix.csv", index_col=0)
    normalized = pd.read_csv(
        experiment_path / "tables" / "confusion_matrix_normalized.csv", index_col=0
    )
    save_confusion_matrix_figure(
        experiment_path / "figures" / "confusion_matrix.png", confusion, "Confusion Matrix"
    )
    save_confusion_matrix_figure(
        experiment_path / "figures" / "confusion_matrix_normalized.png",
        normalized,
        "Normalized Confusion Matrix",
    )
    error_analysis = build_error_analysis(predictions_frame, "summary", "description")
    save_payload(experiment_path / "metrics" / "error_analysis.json", error_analysis)
    metadata["test_evaluation_count"] = int(metadata.get("test_evaluation_count", 0)) + 1
    metadata["test_evaluated_utc"] = utc_now_iso()
    metadata["resource_after_evaluation"] = _resource_snapshot()
    metadata["artifact_sizes_bytes"] = _artifact_sizes(experiment_path)
    write_json(experiment_path / "metadata.json", metadata)
    save_payload(
        experiment_path / "metrics" / "runtime_resource_summary.json",
        {
            "model_comparison": metadata["comparison_rows"],
            "final_fit_seconds": metadata["training_seconds_full_dev_fit"],
            "inference_seconds": metrics["inference_seconds"],
            "resource_before_training": metadata["resource_before_training"],
            "resource_after_training": metadata["resource_after_training"],
            "resource_after_evaluation": metadata["resource_after_evaluation"],
            "artifact_sizes_bytes": metadata["artifact_sizes_bytes"],
        },
    )
    return {"metrics": metrics, "predictions": predictions_frame}


def _load_model(path: Path):
    import joblib

    return joblib.load(path)


def _sklearn_scoring_name(metric_name: str) -> str:
    mapping = {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }
    return mapping.get(metric_name, metric_name)


def _build_scorer(metric_name: str, labels: list[str]):
    if metric_name == "macro_f1":
        return FixedLabelMacroF1Scorer(tuple(labels))
    return _sklearn_scoring_name(metric_name)


def _extract_linear_features(pipeline, labels: list[str], top_n: int = 30) -> pd.DataFrame | None:
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return None
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    rows: list[dict[str, Any]] = []
    coefficients = classifier.coef_
    model_labels = list(classifier.classes_)
    coefficient_labels = model_labels if len(coefficients) > 1 else [model_labels[-1]]
    for label, values in zip(coefficient_labels, coefficients, strict=True):
        for rank, index in enumerate(values.argsort()[-top_n:][::-1], start=1):
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "feature": feature_names[index],
                    "coefficient": float(values[index]),
                }
            )
    return pd.DataFrame(rows)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _research_split_columns(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Limit persisted splits to fields needed for reproducibility and evaluation."""

    candidates = [
        "id",
        config["text_columns"]["summary"],
        config["text_columns"]["description"],
        config["target_column"],
        "creation_time",
        "product",
        "component",
        "duplicate_group",
    ]
    return list(dict.fromkeys(column for column in candidates if column in frame.columns))


def _class_distribution(frame: pd.DataFrame, target: str, labels: list[str]) -> pd.DataFrame:
    counts = frame[target].value_counts()
    return pd.DataFrame(
        {
            "severity": labels,
            "count": [int(counts.get(label, 0)) for label in labels],
            "percentage": [float(counts.get(label, 0) / max(len(frame), 1)) for label in labels],
        }
    )


def _resource_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    process = psutil.Process()
    return {
        "available_memory_gb": memory.available / 2**30,
        "total_memory_gb": memory.total / 2**30,
        "process_rss_bytes": process.memory_info().rss,
        "disk_free_gb": disk.free / 2**30,
        "disk_total_gb": disk.total / 2**30,
    }


def _estimate_tfidf_dimensions(features: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    from .features import build_tfidf_vectorizer
    from .preprocessing import TextCombiner

    feature_config = dict(config["features"])
    configured_ranges = []
    for grid in config["models"].get("param_grids", {}).values():
        configured_ranges.extend(grid.get("tfidf__ngram_range", []))
    if configured_ranges:
        feature_config["ngram_range"] = max(configured_ranges, key=lambda value: tuple(value))
    combiner = TextCombiner(
        config["text_columns"]["summary"],
        config["text_columns"]["description"],
        config["features"]["representation"],
        config["cleaning"],
    )
    texts = combiner.fit_transform(features)
    vectorizer = build_tfidf_vectorizer(feature_config)
    matrix = vectorizer.fit_transform(texts)
    sparse_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return {
        "development_rows": int(matrix.shape[0]),
        "estimated_max_vocabulary_features": int(matrix.shape[1]),
        "nonzero_entries": int(matrix.nnz),
        "sparse_matrix_bytes": int(sparse_bytes),
        "ngram_range_used_for_estimate": list(feature_config["ngram_range"]),
        "fit_scope": "development_only",
    }


def _redact_text_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    email_pattern = re.compile(r"\S*@\S*")
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    for column in [config["text_columns"]["summary"], config["text_columns"]["description"]]:
        if column in result:
            result[column] = (
                result[column]
                .fillna("")
                .astype(str)
                .map(lambda value: url_pattern.sub("[URL]", email_pattern.sub("[EMAIL]", value)))
            )
    return result


def _input_provenance(config: dict[str, Any]) -> dict[str, Any]:
    parquet_path = Path(config["dataset"]["path"])
    provenance: dict[str, Any] = {
        "parquet_path": str(parquet_path),
        "parquet_sha256": file_fingerprint(parquet_path),
        "parquet_size_bytes": parquet_path.stat().st_size,
    }
    manifest_path = parquet_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(
            (
                row
                for row in manifest
                if Path(row["output_parquet_path"]).resolve() == parquet_path.resolve()
            ),
            None,
        )
        if entry:
            provenance["raw_source_path"] = entry["source_path"]
            provenance["raw_source_sha256"] = entry["source_sha256"]
            provenance["raw_source_size_bytes"] = entry["source_file_size"]
            provenance["ingestion_configuration_fingerprint"] = entry[
                "configuration_fingerprint"
            ]
    return provenance


def _artifact_sizes(experiment_path: Path) -> dict[str, int]:
    return {
        str(path.relative_to(experiment_path)): path.stat().st_size
        for path in experiment_path.rglob("*")
        if path.is_file()
    }
