"""Configuration loading and validation."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from .utils import read_yaml, recursive_merge

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {"path": "data/raw/sample_data.csv", "chunksize": 50000},
    "output": {"root_dir": "reports", "experiments_dir": "reports/experiments"},
    "source_columns": {
        "id": ["id", "bug_id", "issue_id", "report_id"],
        "summary": ["summary", "short_desc", "title"],
        "description": ["description", "details", "long_desc", "body"],
        "severity": ["severity", "sev"],
        "creation_time": ["creation_time", "creation time", "created_at", "creation date"],
        "product": ["product"],
        "component": ["component"],
        "dupe_of": ["dupe_of", "dupe of", "duplicate_of"],
    },
    "target_column": "severity",
    "label_exclusions": ["enhancement", "task", "feature", "unknown", "n/a", "none", ""],
    "label_grouping": {
        "enabled": False,
        "groups": {
            "high": ["blocker", "critical"],
            "medium": ["major", "normal"],
            "low": ["minor", "trivial"],
        },
    },
    "text_columns": {"summary": "summary", "description": "description"},
    "cleaning": {
        "lowercase": True,
        "remove_html": True,
        "replace_urls": True,
        "replace_emails": True,
        "remove_code_blocks": False,
        "remove_stack_traces": False,
        "normalize_unicode": True,
        "normalize_whitespace": True,
    },
    "split": {
        "strategy": "chronological",
        "train_fraction": 0.8,
        "random_state": 42,
        "allow_missing_dates_fallback": False,
    },
    "features": {
        "representation": "summary_description",
        "analyzer": "word",
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_df": 1.0,
        "max_features": 50000,
        "sublinear_tf": True,
        "lowercase": False,
    },
    "models": {
        "enabled": ["DummyClassifier", "MultinomialNB", "LogisticRegression", "LinearSVC"],
        "search_method": "grid",
        "search_iterations": 6,
        "class_weight": "balanced",
        "random_state": 42,
        "param_grids": {
            "DummyClassifier": {},
            "MultinomialNB": {
                "clf__alpha": [0.1, 0.5, 1.0],
                "tfidf__ngram_range": [[1, 1], [1, 2]],
            },
            "LogisticRegression": {
                "clf__C": [0.5, 1.0, 2.0],
                "tfidf__ngram_range": [[1, 1], [1, 2]],
            },
            "LinearSVC": {"clf__C": [0.5, 1.0, 2.0], "tfidf__ngram_range": [[1, 1], [1, 2]]},
        },
    },
    "cv": {"strategy": "auto", "n_splits": 3, "shuffle": True, "random_state": 42},
    "primary_metric": "macro_f1",
    "bootstrap": {"n_resamples": 1000, "confidence_level": 0.95, "random_state": 42},
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    loaded = read_yaml(config_path)
    merged = recursive_merge(DEFAULT_CONFIG, loaded)
    loaded_dataset = loaded.get("dataset", {})
    if (
        isinstance(loaded_dataset, dict)
        and any(key in loaded_dataset for key in ("paths", "glob"))
        and "path" not in loaded_dataset
    ):
        merged["dataset"].pop("path", None)
    merged = _resolve_relative_paths(merged, config_path.parent)
    validate_config(merged)
    return merged


def _resolve_relative_paths(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    resolved = dict(config)
    project_root = config_dir.parent if config_dir.name in {"configs", "config"} else config_dir
    dataset = dict(resolved["dataset"])
    if "path" in dataset:
        dataset["path"] = _rooted_path(dataset["path"], project_root)
    if "paths" in dataset:
        dataset["paths"] = [_rooted_path(value, project_root) for value in dataset["paths"]]
    if "glob" in dataset:
        dataset["glob"] = _rooted_path(dataset["glob"], project_root)
    dataset["resolved_paths"] = resolve_dataset_files(dataset)
    resolved["dataset"] = dataset
    output_section = dict(resolved["output"])
    for key in ["root_dir", "experiments_dir"]:
        candidate = Path(output_section[key])
        if not candidate.is_absolute():
            output_section[key] = str((project_root / candidate).resolve())
    resolved["output"] = output_section
    return resolved


def _rooted_path(value: str, project_root: Path) -> str:
    candidate = Path(value)
    return str(candidate if candidate.is_absolute() else (project_root / candidate).resolve())


def resolve_dataset_files(dataset: dict[str, Any]) -> list[str]:
    """Resolve a single path, explicit path list, or glob into deterministic files."""

    specifications = sum(key in dataset for key in ("path", "paths", "glob"))
    if specifications != 1:
        raise ValueError("dataset must define exactly one of path, paths, or glob")
    if "path" in dataset:
        return [str(Path(dataset["path"]).resolve())]
    if "paths" in dataset:
        if not isinstance(dataset["paths"], list) or not dataset["paths"]:
            raise ValueError("dataset.paths must be a non-empty list")
        return [str(Path(value).resolve()) for value in dataset["paths"]]
    matches = sorted(glob.glob(str(dataset["glob"])))
    if not matches:
        raise FileNotFoundError(f"Dataset glob matched no files: {dataset['glob']}")
    return [str(Path(value).resolve()) for value in matches]


def validate_config(config: dict[str, Any]) -> None:
    """Raise ValueError when required configuration values are invalid."""

    required_top_level = [
        "dataset",
        "output",
        "source_columns",
        "target_column",
        "split",
        "features",
        "models",
        "cv",
        "bootstrap",
    ]
    for key in required_top_level:
        if key not in config:
            raise ValueError(f"Missing configuration section: {key}")

    resolved_paths = config["dataset"].get("resolved_paths")
    if not isinstance(resolved_paths, list) or not resolved_paths:
        raise ValueError("dataset specification must resolve to at least one CSV")
    chunksize = config["dataset"].get("chunksize", 50000)
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("dataset.chunksize must be an integer >= 1")

    train_fraction = config["split"].get("train_fraction")
    if not isinstance(train_fraction, (int, float)) or not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("split.train_fraction must be between 0 and 1")

    n_splits = config["cv"].get("n_splits")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("cv.n_splits must be an integer >= 2")

    cv_strategy = config["cv"].get("strategy", "auto")
    if cv_strategy not in {"auto", "time_series", "stratified"}:
        raise ValueError("cv.strategy must be auto, time_series, or stratified")

    split_strategy = config["split"].get("strategy")
    if split_strategy not in {"chronological", "stratified_random"}:
        raise ValueError("split.strategy must be chronological or stratified_random")

    confidence_level = config["bootstrap"].get("confidence_level")
    if not isinstance(confidence_level, (int, float)) or not 0 < confidence_level < 1:
        raise ValueError("bootstrap.confidence_level must be between 0 and 1")

    n_resamples = config["bootstrap"].get("n_resamples")
    if not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("bootstrap.n_resamples must be an integer >= 1")

    enabled_models = config["models"].get("enabled")
    if not isinstance(enabled_models, list) or not enabled_models:
        raise ValueError("models.enabled must be a non-empty list")

    if config["features"].get("representation") not in {
        "summary",
        "description",
        "summary_description",
    }:
        raise ValueError(
            "features.representation must be summary, description, or summary_description"
        )

    if config["features"].get("analyzer") not in {"word", "char", "char_wb"}:
        raise ValueError("features.analyzer must be one of word, char, char_wb")
