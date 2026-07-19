"""Model factories and search-space helpers."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_tfidf_vectorizer
from .preprocessing import TextCombiner
from .splitting import PurgedTimeSeriesSplit


def build_pipeline(model_name: str, config: dict[str, object]) -> Pipeline:
    """Build a leakage-safe sklearn pipeline for a supported model."""

    model = model_name.strip()
    class_weight = config["models"].get("class_weight")
    random_state = int(config["models"].get("random_state", 42))
    tfidf = build_tfidf_vectorizer(config["features"])
    text = TextCombiner(
        summary_column=config["text_columns"]["summary"],
        description_column=config["text_columns"]["description"],
        representation=config["features"]["representation"],
        cleaning_options=config["cleaning"],
    )

    if model == "DummyClassifier":
        classifier = DummyClassifier(strategy="most_frequent")
    elif model == "MultinomialNB":
        classifier = MultinomialNB()
    elif model == "LogisticRegression":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=None,
        )
    elif model == "LinearSVC":
        classifier = LinearSVC(class_weight=class_weight, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("text", text), ("tfidf", tfidf), ("clf", classifier)])


def build_search(
    config: dict[str, object], model_name: str, pipeline: Pipeline, scoring: Any
) -> GridSearchCV | RandomizedSearchCV:
    """Build the configured hyperparameter search object."""

    param_grid = _normalize_param_grid(config["models"].get("param_grids", {}).get(model_name, {}))
    search_method = str(config["models"].get("search_method", "grid")).lower()
    n_jobs = int(config["models"].get("n_jobs", 1))
    cv_splitter = _build_cv(config)
    if search_method == "random" and param_grid:
        return RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=int(config["models"].get("search_iterations", 6)),
            scoring=scoring,
            cv=cv_splitter,
            n_jobs=n_jobs,
            random_state=int(config["models"].get("random_state", 42)),
            refit=True,
        )
    return GridSearchCV(
        pipeline,
        param_grid=param_grid or [{}],
        scoring=scoring,
        cv=cv_splitter,
        n_jobs=n_jobs,
        refit=True,
    )


def supported_models(config: dict[str, object]) -> list[str]:
    """Return the configured enabled models in evaluation order."""

    return list(config["models"].get("enabled", []))


def _build_cv(config: dict[str, object]):
    from sklearn.model_selection import StratifiedKFold

    cv_config = config["cv"]
    strategy = str(cv_config.get("strategy", "auto"))
    if strategy == "auto":
        strategy = (
            "time_series" if config["split"].get("strategy") == "chronological" else "stratified"
        )
    if strategy == "time_series":
        return PurgedTimeSeriesSplit(n_splits=int(cv_config["n_splits"]))
    shuffle = bool(cv_config.get("shuffle", True))
    return StratifiedKFold(
        n_splits=int(cv_config["n_splits"]),
        shuffle=shuffle,
        random_state=int(cv_config.get("random_state", 42)) if shuffle else None,
    )


def _normalize_param_grid(param_grid: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in param_grid.items():
        if key == "tfidf__ngram_range" and isinstance(value, list):
            normalized[key] = [tuple(item) if isinstance(item, list) else item for item in value]
        else:
            normalized[key] = value
    return normalized
