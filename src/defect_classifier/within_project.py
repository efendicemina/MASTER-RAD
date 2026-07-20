"""Frozen, resumable within-project Eclipse experiments."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from scipy.stats import friedmanchisquare, pearsonr
from sklearn.base import clone

from .config import load_config
from .development_study import (
    LABELS,
    build_study_pipeline,
    evaluate_folds,
    freeze_temporal_folds,
    redact,
)
from .evaluation import bootstrap_macro_f1_ci, evaluate_predictions
from .pilot_preparation import prepare_pilot_dataframe
from .reporting import save_confusion_matrix_figure
from .training import split_dataset
from .utils import (
    ensure_directory,
    git_commit_hash,
    package_versions,
    sha256_file,
    system_metadata,
    utc_now_iso,
    write_csv,
    write_json,
)

PROJECT_ORDER = ["TPTP", "Papyrus", "PDE", "Equinox", "BIRT", "CDT", "JDT", "Platform"]
MODELS = ["DummyClassifier", "MultinomialNB", "LogisticRegression", "LinearSVC"]
EXPECTED_FILES = [
    "config_snapshot.json",
    "metadata.json",
    "metrics/LogisticRegression.json",
    "tables/model_comparison.csv",
    "tables/development_class_distribution.csv",
    "tables/test_class_distribution.csv",
    "tables/cv_fold_distribution.csv",
    "tables/duplicate_removals.csv",
    "tables/classification_report_LogisticRegression.csv",
    "tables/confusion_matrix_LogisticRegression.csv",
    "tables/confusion_matrix_normalized_LogisticRegression.csv",
    "predictions/LogisticRegression.csv",
    "figures/confusion_matrix_LogisticRegression.png",
    "figures/confusion_matrix_normalized_LogisticRegression.png",
    "tables/influential_features_LogisticRegression.csv",
]


def frozen_project_config(project: str, root: Path) -> dict[str, Any]:
    if project not in PROJECT_ORDER:
        raise ValueError(f"Project must be one of {PROJECT_ORDER}")
    config = load_config(root / "configs" / "eclipse_training_mylyn_challenger.yaml")
    parquet = (root / "data" / "processed" / "eclipse_core" / f"{project}.parquet").resolve()
    config = deepcopy(config)
    config["dataset"]["path"] = str(parquet)
    config["dataset"]["resolved_paths"] = [str(parquet)]
    config["training"]["enabled"] = True
    config["training"]["purpose"] = "frozen_within_project"
    config["project"] = project
    return config


def _distribution(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame["severity"].value_counts()
    return pd.DataFrame(
        {
            "severity": LABELS,
            "count": [int(counts.get(label, 0)) for label in LABELS],
            "share": [float(counts.get(label, 0) / len(frame)) for label in LABELS],
        }
    )


def preflight_project(project: str, root: Path) -> tuple[dict[str, Any], Any, Any, Any]:
    config = frozen_project_config(project, root)
    parquet = Path(config["dataset"]["path"])
    required = {
        "issue_id",
        "summary",
        "description",
        "severity",
        "creation_time",
        "global_report_key",
        "exact_text_hash",
        "duplicate_group_id",
    }
    columns = set(pq.read_schema(parquet).names)
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{project}: missing required Parquet columns {missing}")
    preparation = prepare_pilot_dataframe(config)
    split = split_dataset(config, preparation.frame)
    folds = freeze_temporal_folds(split.development, n_splits=3)
    for scope, frame in [("development", split.development), ("test", split.test)]:
        absent = sorted(set(LABELS) - set(frame["severity"]))
        if absent:
            raise ValueError(f"{project}: {scope} lacks fixed classes {absent}")
    fold_rows = []
    for fold in folds:
        for partition, indices in [("train", fold.train), ("validation", fold.validation)]:
            counts = split.development.iloc[list(indices)]["severity"].value_counts()
            absent = sorted(set(LABELS) - set(counts.index))
            if absent:
                raise ValueError(f"{project}: fold {fold.fold} {partition} lacks {absent}")
            for label in LABELS:
                fold_rows.append(
                    {
                        "fold": fold.fold,
                        "partition": partition,
                        "severity": label,
                        "count": int(counts[label]),
                        "rows": len(indices),
                    }
                )
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(root))
    if disk.free < 80 * 2**30:
        raise OSError(f"{project}: free disk below 80 GiB")
    if memory.available < 2 * 2**30:
        raise MemoryError(f"{project}: available memory below 2 GiB")
    resources = {
        "available_memory_gb": memory.available / 2**30,
        "free_disk_gb": disk.free / 2**30,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "fold_distribution": pd.DataFrame(fold_rows),
    }
    return config, preparation, split, (folds, resources)


def _raw_provenance(parquet: Path) -> dict[str, Any]:
    result = {"parquet_path": str(parquet), "parquet_sha256": sha256_file(parquet)}
    manifest = json.loads((parquet.parent / "manifest.json").read_text(encoding="utf-8"))
    entry = next(row for row in manifest if Path(row["output_parquet_path"]).resolve() == parquet)
    result.update(
        {
            "raw_path": entry["source_path"],
            "raw_sha256": entry["source_sha256"],
            "configuration_fingerprint": entry["configuration_fingerprint"],
        }
    )
    return result


def _linear_features(model, limit: int = 30) -> pd.DataFrame:
    classifier = model.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return pd.DataFrame(columns=["severity", "rank", "feature", "coefficient"])
    names = model.named_steps["features"].get_feature_names_out()
    rows = []
    for label, coefficients in zip(classifier.classes_, classifier.coef_, strict=True):
        for rank, index in enumerate(coefficients.argsort()[-limit:][::-1], start=1):
            rows.append(
                {
                    "severity": label,
                    "rank": rank,
                    "feature": names[index],
                    "coefficient": float(coefficients[index]),
                }
            )
    return pd.DataFrame(rows)


def _update_progress(output_root: Path, row: dict[str, Any]) -> None:
    path = output_root / "processing_progress.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty and row["project"] in set(existing["project"]):
        existing = existing[existing["project"] != row["project"]]
    write_csv(path, pd.concat([existing, pd.DataFrame([row])], ignore_index=True))


def run_project(project: str, root: Path, output_root: Path) -> Path:
    """Run one immutable project experiment and one project-level test access event."""

    completed = sorted(output_root.glob(f"experiments/{project}_*/COMPLETED.json"))
    if completed:
        return completed[-1].parent
    config, preparation, split, (folds, resources) = preflight_project(project, root)
    experiment_id = f"{project}_{utc_now_iso().replace(':', '').replace('-', '')}"
    experiment = output_root / "experiments" / experiment_id
    if experiment.exists():
        raise FileExistsError(experiment)
    for directory in ["artifacts", "metrics", "predictions", "figures", "tables"]:
        ensure_directory(experiment / directory)
    write_json(experiment / "config_snapshot.json", config)
    write_json(experiment / "tables" / "filtering_summary.json", preparation.filtering_summary)
    write_csv(experiment / "tables" / "duplicate_removals.csv", preparation.duplicate_removals)
    write_csv(
        experiment / "tables" / "conflicting_duplicate_labels.csv",
        preparation.conflicting_duplicate_labels,
    )
    write_csv(
        experiment / "tables" / "development_class_distribution.csv",
        _distribution(split.development),
    )
    write_csv(experiment / "tables" / "test_class_distribution.csv", _distribution(split.test))
    write_csv(
        experiment / "tables" / "cv_fold_distribution.csv", resources.pop("fold_distribution")
    )
    metadata: dict[str, Any] = {
        "project": project,
        "experiment_id": experiment_id,
        "status": "development_cv",
        "git_commit": git_commit_hash(root),
        "created_utc": utc_now_iso(),
        "protocol": "docs/eclipse_within_project_protocol.md",
        "labels": LABELS,
        "provenance": _raw_provenance(Path(config["dataset"]["path"])),
        "filtering": preparation.filtering_summary,
        "development_rows": len(split.development),
        "test_rows": len(split.test),
        "duplicate_overlap_rows_removed": split.duplicate_overlap_rows_removed,
        "preflight_resources": resources,
        "test_evaluation_count": 0,
        "system": system_metadata(),
        "packages": package_versions(["pandas", "numpy", "scikit-learn", "scipy", "psutil"]),
        "rare_class_warning": project == "TPTP",
    }
    write_json(experiment / "metadata.json", metadata)
    dev_rows = []
    for model_name in MODELS:
        pipeline = build_study_pipeline(
            "summary_description",
            "word_1_2",
            model_name,
            "balanced" if model_name in {"LogisticRegression", "LinearSVC"} else None,
        )
        result, _, _ = evaluate_folds(
            split.development,
            folds,
            pipeline,
            {"project": project, "model": model_name},
        )
        dev_rows.append(result)
        full = clone(pipeline)
        start = perf_counter()
        full.fit(split.development.drop(columns=["severity"]), split.development["severity"])
        result["full_development_fit_seconds"] = perf_counter() - start
        joblib.dump(full, experiment / "artifacts" / f"{model_name}.joblib")
        write_csv(
            experiment / "tables" / f"influential_features_{model_name}.csv",
            _linear_features(full),
        )
        del full
    write_csv(experiment / "tables" / "development_model_comparison.csv", pd.DataFrame(dev_rows))
    metadata["status"] = "test_access_started"
    metadata["test_access_started_utc"] = utc_now_iso()
    write_json(experiment / "metadata.json", metadata)

    test_features = split.test.drop(columns=["severity"])
    test_target = split.test["severity"].tolist()
    model_rows = []
    for model_name in MODELS:
        fitted_model = joblib.load(experiment / "artifacts" / f"{model_name}.joblib")
        start = perf_counter()
        predicted = fitted_model.predict(test_features)
        inference = perf_counter() - start
        del fitted_model
        evaluation = evaluate_predictions(test_target, list(predicted), LABELS)
        ci = bootstrap_macro_f1_ci(test_target, list(predicted), 1000, 0.95, 42, LABELS)
        evaluation.metrics["bootstrap_macro_f1_ci"] = ci
        evaluation.metrics["inference_seconds"] = inference
        write_json(experiment / "metrics" / f"{model_name}.json", evaluation.metrics)
        report = pd.DataFrame(evaluation.classification_report).transpose().reset_index()
        write_csv(experiment / "tables" / f"classification_report_{model_name}.csv", report)
        write_csv(
            experiment / "tables" / f"confusion_matrix_{model_name}.csv",
            evaluation.confusion_matrix,
        )
        write_csv(
            experiment / "tables" / f"confusion_matrix_normalized_{model_name}.csv",
            evaluation.normalized_confusion_matrix,
        )
        save_confusion_matrix_figure(
            experiment / "figures" / f"confusion_matrix_{model_name}.png",
            evaluation.confusion_matrix,
            f"{project} {model_name}",
        )
        save_confusion_matrix_figure(
            experiment / "figures" / f"confusion_matrix_normalized_{model_name}.png",
            evaluation.normalized_confusion_matrix,
            f"{project} {model_name} normalized",
        )
        predictions = pd.DataFrame(
            {
                "report_id": split.test["id"],
                "true_label": test_target,
                "predicted_label": predicted,
                "correct": np.asarray(test_target) == predicted,
                "summary": split.test["summary"].map(redact),
                "description": split.test["description"].map(lambda value: redact(value, 1000)),
            }
        )
        write_csv(experiment / "predictions" / f"{model_name}.csv", predictions)
        errors = (
            predictions[~predictions["correct"]]
            .groupby(["true_label", "predicted_label"])
            .size()
            .reset_index(name="errors")
            .sort_values("errors", ascending=False)
        )
        write_csv(experiment / "tables" / f"error_analysis_{model_name}.csv", errors)
        write_csv(
            experiment / "tables" / f"error_examples_{model_name}.csv",
            predictions[~predictions["correct"]].head(100),
        )
        dev = next(row for row in dev_rows if row["model"] == model_name)
        model_rows.append(
            {
                "project": project,
                "model": model_name,
                "cv_macro_f1_mean": dev["mean_cv_macro_f1"],
                "cv_macro_f1_std": dev["std_cv_macro_f1"],
                **evaluation.metrics,
                "bootstrap_lower": ci["lower"],
                "bootstrap_upper": ci["upper"],
                "full_fit_seconds": dev["full_development_fit_seconds"],
                "cv_fit_seconds": dev["fit_seconds"],
                "inference_seconds": inference,
                "model_size_bytes": (experiment / "artifacts" / f"{model_name}.joblib")
                .stat()
                .st_size,
                "feature_count": dev["feature_count"],
            }
        )
    comparison = pd.DataFrame(model_rows)
    write_csv(experiment / "tables" / "model_comparison.csv", comparison)
    metadata.update(
        {
            "status": "complete",
            "test_evaluation_count": 1,
            "test_evaluated_utc": utc_now_iso(),
            "runtime_resources": {
                "process_rss_bytes": psutil.Process().memory_info().rss,
                "available_memory_gb": psutil.virtual_memory().available / 2**30,
                "free_disk_gb": psutil.disk_usage(str(root)).free / 2**30,
            },
            "artifact_count": len(list(experiment.rglob("*.*"))),
        }
    )
    write_json(experiment / "metadata.json", metadata)
    missing = [name for name in EXPECTED_FILES if not (experiment / name).is_file()]
    if missing:
        raise RuntimeError(f"{project}: incomplete artifacts {missing}")
    write_json(experiment / "COMPLETED.json", {"project": project, "completed_utc": utc_now_iso()})
    primary = comparison[comparison["model"] == "LogisticRegression"].iloc[0]
    _update_progress(
        output_root,
        {
            "project": project,
            "experiment_id": experiment_id,
            "status": "complete",
            "test_evaluation_count": 1,
            "test_macro_f1": primary["macro_f1"],
            "warning": "rare trivial support" if project == "TPTP" else "",
            "completed_utc": utc_now_iso(),
        },
    )
    return experiment


def _latest_completed(output_root: Path, project: str) -> Path:
    matches = sorted(output_root.glob(f"experiments/{project}_*/COMPLETED.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one completed {project} experiment, found {len(matches)}")
    return matches[0].parent


def aggregate_results(root: Path, output_root: Path, mylyn_experiment: Path) -> None:
    all_models, projects, per_class, intervals, runtimes, distributions, confusions = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for project in PROJECT_ORDER:
        experiment = _latest_completed(output_root, project)
        comparison = pd.read_csv(experiment / "tables" / "model_comparison.csv")
        comparison["experiment_id"] = experiment.name
        all_models.append(comparison)
        primary = comparison[comparison.model == "LogisticRegression"].iloc[0]
        metadata = json.loads((experiment / "metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((experiment / "metrics" / "LogisticRegression.json").read_text())
        projects.append(
            {
                "project": project,
                "eligible_rows": metadata["filtering"][
                    "modeling_rows_after_filtering_and_deduplication"
                ],
                "development_size": metadata["development_rows"],
                "test_size": metadata["test_rows"],
                "cv_macro_f1_mean": primary.cv_macro_f1_mean,
                "cv_macro_f1_std": primary.cv_macro_f1_std,
                "test_macro_f1": primary.macro_f1,
                "bootstrap_lower": primary.bootstrap_lower,
                "bootstrap_upper": primary.bootstrap_upper,
                "weighted_f1": primary.weighted_f1,
                "balanced_accuracy": primary.balanced_accuracy,
                "accuracy": primary.accuracy,
                "selected_model": "LogisticRegression",
                "runtime_seconds": primary.cv_fit_seconds
                + primary.full_fit_seconds
                + primary.inference_seconds,
                "warning_status": "rare trivial support" if project == "TPTP" else "",
            }
        )
        for index, label in enumerate(LABELS):
            support = metrics["per_class_support"][label]["support"]
            per_class.append(
                {
                    "project": project,
                    "severity": label,
                    "precision": metrics["per_class_precision"][index],
                    "recall": metrics["per_class_recall"][index],
                    "f1": metrics["per_class_f1"][index],
                    "support": support,
                }
            )
        intervals.append(
            {
                "project": project,
                "lower": primary.bootstrap_lower,
                "mean": metrics["bootstrap_macro_f1_ci"]["mean"],
                "upper": primary.bootstrap_upper,
            }
        )
        runtimes.extend(
            comparison[
                [
                    "project",
                    "model",
                    "cv_fit_seconds",
                    "full_fit_seconds",
                    "inference_seconds",
                    "model_size_bytes",
                    "feature_count",
                ]
            ].to_dict("records")
        )
        dist = pd.read_csv(experiment / "tables" / "test_class_distribution.csv")
        dist["project"] = project
        distributions.extend(dist.to_dict("records"))
        matrix = pd.read_csv(
            experiment / "tables" / "confusion_matrix_LogisticRegression.csv", index_col=0
        )
        for true_label in LABELS:
            for predicted_label in LABELS:
                confusions.append(
                    {
                        "project": project,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix.loc[true_label, predicted_label]),
                    }
                )
    mylyn_meta = json.loads((mylyn_experiment / "metadata.json").read_text())
    mylyn_metrics = json.loads((mylyn_experiment / "metrics" / "test_metrics.json").read_text())
    mylyn_cv = pd.read_csv(mylyn_experiment / "tables" / "model_comparison.csv")
    mylyn_selected = mylyn_cv[mylyn_cv.model == "LogisticRegression"].iloc[0]
    projects.append(
        {
            "project": "MYLYN",
            "eligible_rows": mylyn_meta["retained_rows"],
            "development_size": mylyn_meta["development_rows"],
            "test_size": mylyn_meta["test_rows"],
            "cv_macro_f1_mean": mylyn_selected.cv_macro_f1,
            "cv_macro_f1_std": mylyn_selected.cv_macro_f1_std,
            "test_macro_f1": mylyn_metrics["macro_f1"],
            "bootstrap_lower": mylyn_metrics["bootstrap_macro_f1_ci"]["lower"],
            "bootstrap_upper": mylyn_metrics["bootstrap_macro_f1_ci"]["upper"],
            "weighted_f1": mylyn_metrics["weighted_f1"],
            "balanced_accuracy": mylyn_metrics["balanced_accuracy"],
            "accuracy": mylyn_metrics["accuracy"],
            "selected_model": "LogisticRegression",
            "runtime_seconds": mylyn_selected.training_seconds
            + mylyn_meta["training_seconds_full_dev_fit"]
            + mylyn_metrics["inference_seconds"],
            "warning_status": "immutable initial baseline",
        }
    )
    for index, label in enumerate(LABELS):
        per_class.append(
            {
                "project": "MYLYN",
                "severity": label,
                "precision": mylyn_metrics["per_class_precision"][index],
                "recall": mylyn_metrics["per_class_recall"][index],
                "f1": mylyn_metrics["per_class_f1"][index],
                "support": mylyn_metrics["per_class_support"][label]["support"],
            }
        )
    intervals.append({"project": "MYLYN", **mylyn_metrics["bootstrap_macro_f1_ci"]})
    project_frame = pd.DataFrame(projects)
    class_frame = pd.DataFrame(per_class)
    model_frame = pd.concat(all_models, ignore_index=True)
    write_csv(output_root / "project_results.csv", project_frame)
    write_csv(output_root / "per_class_results.csv", class_frame)
    write_csv(output_root / "confidence_intervals.csv", pd.DataFrame(intervals))
    write_csv(output_root / "runtime_summary.csv", pd.DataFrame(runtimes))
    write_csv(output_root / "class_distribution_summary.csv", pd.DataFrame(distributions))
    write_csv(output_root / "confusion_summary.csv", pd.DataFrame(confusions))
    write_csv(output_root / "model_by_project_results.csv", model_frame)
    pivot = model_frame.pivot(index="project", columns="model", values="macro_f1").loc[
        PROJECT_ORDER
    ]
    statistic, p_value = friedmanchisquare(*(pivot[model] for model in MODELS))
    pairwise = []
    for left_index, left in enumerate(MODELS):
        for right in MODELS[left_index + 1 :]:
            difference = pivot[left] - pivot[right]
            pairwise.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "mean_difference": difference.mean(),
                    "median_difference": difference.median(),
                    "wins_a": int((difference > 0).sum()),
                    "ties": int((difference == 0).sum()),
                    "wins_b": int((difference < 0).sum()),
                }
            )
    write_json(
        output_root / "friedman_test.json",
        {
            "projects": len(pivot),
            "statistic": statistic,
            "p_value": p_value,
            "posthoc_performed": False,
        },
    )
    write_csv(output_root / "pairwise_effect_sizes.csv", pd.DataFrame(pairwise))
    values = project_frame.test_macro_f1
    without_tptp = project_frame[project_frame.project != "TPTP"].test_macro_f1
    correlation = pearsonr(project_frame.development_size, values)
    per_class_summary = class_frame.groupby("severity").f1.agg(["mean", "median"]).reset_index()
    write_csv(output_root / "per_class_summary.csv", per_class_summary)
    lines = [
        "# Frozen within-project Eclipse results",
        "",
        f"Nine-project macro-F1 mean: {values.mean():.4f}; median: {values.median():.4f}; "
        f"SD: {values.std(ddof=0):.4f}; range: {values.min():.4f}-{values.max():.4f}.",
        f"Sensitivity excluding TPTP: mean {without_tptp.mean():.4f}; "
        f"median {without_tptp.median():.4f}.",
        f"Exploratory Pearson correlation between development size and macro F1: "
        f"r={correlation.statistic:.4f}, p={correlation.pvalue:.4g}.",
        f"Friedman test across eight new projects and four models: chi-square={statistic:.4f}, "
        f"p={p_value:.4g}. No post-hoc significance tests were performed.",
        "",
        "MYLYN is the immutable initial result and is excluded from the four-model Friedman block.",
        "Five MYLYN conflicting exact-text groups and minority-label review remain pending.",
        "Automated modeling does not establish label correctness.",
    ]
    (output_root / "within_project_summary.md").write_text("\n".join(lines), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(project_frame.project, project_frame.test_macro_f1)
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("Held-out macro F1")
    figure.tight_layout()
    figure.savefig(output_root / "macro_f1_by_project.png", dpi=200)
    plt.close(figure)
    pivot_class = class_frame.pivot(index="project", columns="severity", values="f1")
    axis = pivot_class.loc[project_frame.project].plot(kind="bar", figsize=(12, 6))
    axis.set_ylabel("Held-out F1")
    axis.figure.tight_layout()
    axis.figure.savefig(output_root / "per_class_f1_by_project.png", dpi=200)
    plt.close(axis.figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "run", "aggregate"])
    parser.add_argument("--project", choices=PROJECT_ORDER)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("reports/within_project"))
    parser.add_argument("--mylyn-experiment", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        if not args.project:
            parser.error("--project is required")
        _, preparation, split, (_, resources) = preflight_project(args.project, args.root)
        print(
            json.dumps(
                {
                    "project": args.project,
                    "filtering": preparation.filtering_summary,
                    "development": len(split.development),
                    "test": len(split.test),
                    "duplicate_overlap_removed": split.duplicate_overlap_rows_removed,
                    "resources": {k: v for k, v in resources.items() if k != "fold_distribution"},
                },
                indent=2,
            )
        )
    elif args.command == "run":
        if not args.project:
            parser.error("--project is required")
        print(run_project(args.project, args.root, args.output))
    else:
        if not args.mylyn_experiment:
            parser.error("--mylyn-experiment is required")
        aggregate_results(args.root, args.output, args.mylyn_experiment)
