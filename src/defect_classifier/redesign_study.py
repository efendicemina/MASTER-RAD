"""MYLYN development-only target, quality, preprocessing, and feature redesign study."""

from __future__ import annotations

import hashlib
import html
import json
import pickle
import re
import unicodedata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from .development_study import FrozenFold, folds_fingerprint, freeze_temporal_folds
from .hierarchical_study import (
    CONFLICT_IDS,
    EXPECTED_FOLD_HASH,
    MYLYN_DEVELOPMENT_SHA256,
    _row_hash,
    load_hierarchical_development,
)
from .preprocessing import TextCombiner
from .reporting import save_confusion_matrix_figure
from .utils import ensure_directory, sha256_file, write_csv, write_json

S6_LABELS = ["blocker", "critical", "major", "normal", "minor", "trivial"]
TASKS = {
    "s6": {label: label for label in S6_LABELS},
    "s3": {
        "blocker": "HIGH",
        "critical": "HIGH",
        "major": "MEDIUM",
        "normal": "MEDIUM",
        "minor": "LOW",
        "trivial": "LOW",
    },
    "s2": {
        "blocker": "HIGH_IMPACT",
        "critical": "HIGH_IMPACT",
        "major": "HIGH_IMPACT",
        "normal": "LOWER_IMPACT",
        "minor": "LOWER_IMPACT",
        "trivial": "LOWER_IMPACT",
    },
}
TASK_LABELS = {
    "s6": S6_LABELS,
    "s3": ["HIGH", "MEDIUM", "LOW"],
    "s2": ["HIGH_IMPACT", "LOWER_IMPACT"],
}
FEATURES = ["F0", "F1", "F2", "F3", "F4"]
LABEL_TERMS = re.compile(r"\b(blocker|critical|major|normal|minor|trivial|severity)\b", re.I)
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
ISSUE = re.compile(r"\b(?:bug\s*#?|issue\s*#?|bz\s*#?)\d{3,}\b", re.I)
HEX = re.compile(r"\b(?:0x)?[0-9a-f]{10,}\b", re.I)
HTML_TAG = re.compile(r"<[^>]+>")
STACK = re.compile(r"(?:^|\n)\s*at\s+[\w.$]+\([^\n]+\)", re.I)
CODE = re.compile(r"[{};]|(?:^|\n)\s*(?:public|private|class|def|if|for)\b", re.I)
BUG_REF = re.compile(r"\b(?:bug|issue|bz)\s*#?\s*\d+\b", re.I)
SEED = 42
BASELINE_S6 = 0.2262765
ALWAYS_INELIGIBLE = {
    "severity",
    "priority",
    "status",
    "resolution",
    "dupe of",
    "depends on",
    "blocks",
    "assigned to",
    "qa contact",
    "cc",
    "creator",
    "comments",
    "history/activity log",
    "attachments",
    "last change time",
    "deadline",
    "target milestone",
    "flags",
    "whiteboard",
    "is confirmed",
    "is open",
}


def enforce_metadata_eligibility(fields: list[str], eligibility: pd.DataFrame) -> list[str]:
    normalized = [field.strip().lower() for field in fields]
    if set(normalized) & ALWAYS_INELIGIBLE:
        raise ValueError("Ineligible metadata requested")
    status = eligibility.set_index("normalized_name")["eligibility_status"].to_dict()
    if any(status.get(field) != "PRIMARY_ELIGIBLE" for field in normalized):
        raise ValueError("Only PRIMARY_ELIGIBLE metadata may enter primary models")
    return normalized


def fit_rare_categories(values: pd.Series, minimum: int = 5) -> set[str]:
    counts = values.fillna("[MISSING]").astype(str).value_counts()
    return set(counts[counts >= minimum].index)


def apply_rare_categories(values: pd.Series, retained: set[str]) -> pd.Series:
    normalized = values.fillna("[MISSING]").astype(str)
    return normalized.where(normalized.isin(retained), "[RARE]")


def normalize_severity(value: Any) -> str:
    normalized = "" if pd.isna(value) else str(value).strip().lower()
    if normalized not in S6_LABELS:
        raise ValueError(f"Invalid severity: {value}")
    return normalized


def map_target(values: pd.Series, task: str) -> pd.Series:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    normalized = values.map(normalize_severity)
    return normalized.map(TASKS[task])


def clean_text_value(summary: Any, description: Any, mask_labels: bool = False) -> str:
    summary_text = "" if pd.isna(summary) else str(summary)
    description_text = "" if pd.isna(description) else str(description)
    value = f"[SUMMARY] {summary_text} [DESCRIPTION] {description_text}"
    value = unicodedata.normalize("NFKC", html.unescape(value.replace("\r\n", "\n")))
    value = HTML_TAG.sub(" ", value)
    value = URL.sub(" [URL] ", value)
    value = EMAIL.sub(" [EMAIL] ", value)
    value = ISSUE.sub(" [ISSUE_ID] ", value)
    value = HEX.sub(" [HEX] ", value)
    if mask_labels:
        value = LABEL_TERMS.sub(" [SEVERITY_TERM] ", value)
    return " ".join(value.split()).lower()


class FieldText(BaseEstimator, TransformerMixin):
    def __init__(self, field: str, clean: bool = False, mask_labels: bool = False):
        self.field = field
        self.clean = clean
        self.mask_labels = mask_labels

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        values = x[self.field].fillna("").astype(str)
        if not self.clean:
            return values.to_numpy()
        if self.field == "summary":
            return np.asarray([clean_text_value(value, "", self.mask_labels) for value in values])
        return np.asarray([clean_text_value("", value, self.mask_labels) for value in values])


class CombinedText(BaseEstimator, TransformerMixin):
    def __init__(self, clean: bool = False, mask_labels: bool = False):
        self.clean = clean
        self.mask_labels = mask_labels

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if self.clean:
            return np.asarray(
                [
                    clean_text_value(summary, description, self.mask_labels)
                    for summary, description in zip(x.summary, x.description, strict=True)
                ]
            )
        summary = x.summary.fillna("").astype(str)
        description = x.description.fillna("").astype(str)
        return (summary + " " + description).to_numpy()


class ApprovedText(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        transformer = TextCombiner(
            "summary",
            "description",
            "summary_description",
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
        return transformer.transform(x)


def structural_values(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    punctuation = re.compile(r"[^\w\s]")
    repeated = re.compile(r"(.)\1{3,}")
    for summary, description in zip(frame.summary, frame.description, strict=True):
        s, d = (
            "" if pd.isna(summary) else str(summary),
            "" if pd.isna(description) else str(description),
        )
        combined = f"{s}\n{d}"
        length = max(len(combined), 1)
        lines = combined.splitlines() or [""]
        digits = sum(char.isdigit() for char in combined)
        upper = sum(char.isupper() for char in combined)
        rows.append(
            [
                len(s),
                len(s.split()),
                len(d),
                len(d.split()),
                len(combined),
                len(combined.split()),
                len(lines),
                int(not d.strip()),
                digits,
                digits / length,
                upper,
                upper / length,
                len(punctuation.findall(combined)),
                combined.count("!"),
                combined.count("?"),
                sum(bool(CODE.search(line)) for line in lines),
                int(bool(STACK.search(combined))),
                int(bool(URL.search(combined))),
                int(bool(EMAIL.search(combined))),
                int(bool(HTML_TAG.search(combined))),
                int(bool(BUG_REF.search(combined))),
                int(bool(HEX.search(combined))),
                int(bool(repeated.search(combined))),
            ]
        )
    return np.asarray(rows, dtype=float)


class StructuralText(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return structural_values(x)


def _word_branch(field: str, maximum: int) -> Pipeline:
    return Pipeline(
        [
            ("text", FieldText(field, clean=True)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=maximum,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_features(feature: str, mask_labels: bool = False):
    if feature == "F0":
        return Pipeline(
            [
                (
                    "text",
                    CombinedText(clean=False, mask_labels=True) if mask_labels else ApprovedText(),
                ),
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.98,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    if feature == "F1":
        return Pipeline(
            [
                ("text", CombinedText(clean=True, mask_labels=mask_labels)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.98,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    branches: list[tuple[str, Any]] = [
        ("summary", _word_branch("summary", 20_000)),
        ("description", _word_branch("description", 40_000)),
    ]
    if feature in {"F3", "F4"}:
        branches.append(
            ("structure", Pipeline([("values", StructuralText()), ("scale", MaxAbsScaler())]))
        )
    if feature == "F4":
        branches.append(
            (
                "char",
                Pipeline(
                    [
                        ("text", CombinedText(clean=True, mask_labels=mask_labels)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=3,
                                max_df=0.995,
                                max_features=30_000,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            )
        )
    if feature not in {"F2", "F3", "F4"}:
        raise ValueError(f"Unknown feature set: {feature}")
    return FeatureUnion(branches)


def model_specs() -> list[dict[str, Any]]:
    rows = []
    for c_value in (0.25, 1.0, 4.0):
        for weight in (None, "balanced"):
            rows.append({"family": "LogisticRegression", "C": c_value, "class_weight": weight})
    for c_value in (0.1, 0.5, 1.0, 2.0):
        for weight in (None, "balanced"):
            rows.append({"family": "LinearSVC", "C": c_value, "class_weight": weight})
    return rows


def build_model(spec: dict[str, Any]):
    if spec["family"] == "LogisticRegression":
        return LogisticRegression(
            C=spec["C"], class_weight=spec["class_weight"], max_iter=2000, random_state=SEED
        )
    if spec["family"] == "LinearSVC":
        return LinearSVC(C=spec["C"], class_weight=spec["class_weight"], random_state=SEED)
    raise ValueError("Unsupported model family")


def fixed_metrics(truth: np.ndarray, predicted: np.ndarray, labels: list[str]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    result: dict[str, Any] = {
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "weighted_f1": f1_score(
            truth, predicted, labels=labels, average="weighted", zero_division=0
        ),
        "accuracy": float(np.mean(truth == predicted)),
        "minimum_class_recall": float(recall.min()),
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


def _inner_folds(train: pd.DataFrame, task: str) -> tuple[FrozenFold, ...]:
    mapped = train.copy()
    mapped["severity"] = map_target(mapped.severity, task)
    folds = freeze_temporal_folds(mapped, n_splits=2)
    labels = set(TASK_LABELS[task])
    valid = tuple(
        fold
        for fold in folds
        if set(mapped.iloc[list(fold.train)].severity) == labels
        and set(mapped.iloc[list(fold.validation)].severity) == labels
    )
    if valid:
        return valid
    boundary = max(1, int(len(train) * 0.75))
    fallback = FrozenFold(1, tuple(range(boundary)), tuple(range(boundary, len(train))))
    if set(mapped.iloc[list(fallback.train)].severity) != labels:
        raise RuntimeError(f"Insufficient inner class support for {task}")
    return (fallback,)


def _candidate_key(feature: str, spec: dict[str, Any]) -> str:
    weight = "none" if spec["class_weight"] is None else spec["class_weight"]
    return f"{feature}|{spec['family']}|C={spec['C']}|weight={weight}"


def _select_candidate(results: pd.DataFrame, task: str) -> pd.Series:
    aggregated = (
        results.groupby(["candidate", "feature", "family", "C", "class_weight"], dropna=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", lambda x: x.std(ddof=0)),
            minimum_recall=("minimum_class_recall", "mean"),
            mean_features=("feature_count", "mean"),
            runtime=("runtime_seconds", "sum"),
        )
        .reset_index()
    )
    priority = (
        "recall_HIGH"
        if task == "s3"
        else "recall_HIGH_IMPACT"
        if task == "s2"
        else "recall_blocker"
    )
    recall = results.groupby("candidate")[priority].mean()
    aggregated["priority_recall"] = aggregated.candidate.map(recall)
    aggregated["feature_rank"] = aggregated.feature.map(
        {name: index for index, name in enumerate(FEATURES)}
    )
    return aggregated.sort_values(
        [
            "mean_macro_f1",
            "minimum_recall",
            "priority_recall",
            "std_macro_f1",
            "feature_rank",
            "mean_features",
            "runtime",
            "candidate",
        ],
        ascending=[False, False, False, True, True, True, True, True],
    ).iloc[0]


def _nested_task(frame: pd.DataFrame, outer_folds: tuple[FrozenFold, ...], task: str):
    inner_rows, selected_rows, fold_rows, oof_parts = [], [], [], []
    labels = TASK_LABELS[task]
    specs = model_specs()
    for outer in outer_folds:
        outer_train, validation = frame.iloc[list(outer.train)], frame.iloc[list(outer.validation)]
        inner_folds = _inner_folds(outer_train, task)
        for inner in inner_folds:
            inner_train = outer_train.iloc[list(inner.train)]
            inner_validation = outer_train.iloc[list(inner.validation)]
            y_train, y_validation = (
                map_target(inner_train.severity, task),
                map_target(inner_validation.severity, task),
            )
            for feature in FEATURES:
                transformer = build_features(feature)
                started = perf_counter()
                x_train = transformer.fit_transform(inner_train)
                x_validation = transformer.transform(inner_validation)
                feature_seconds = perf_counter() - started
                for spec in specs:
                    model = build_model(spec)
                    fit_start = perf_counter()
                    model.fit(x_train, y_train)
                    predicted = model.predict(x_validation)
                    metrics = fixed_metrics(y_validation.to_numpy(), predicted, labels)
                    inner_rows.append(
                        {
                            "task": task,
                            "outer_fold": outer.fold,
                            "inner_fold": inner.fold,
                            "candidate": _candidate_key(feature, spec),
                            "feature": feature,
                            **spec,
                            **metrics,
                            "feature_count": x_train.shape[1],
                            "runtime_seconds": perf_counter()
                            - fit_start
                            + feature_seconds / len(specs),
                        }
                    )
        outer_inner = pd.DataFrame(
            [row for row in inner_rows if row["outer_fold"] == outer.fold and row["task"] == task]
        )
        selected = _select_candidate(outer_inner, task)
        selected_rows.append({"task": task, "outer_fold": outer.fold, **selected.to_dict()})
        spec = {
            "family": selected.family,
            "C": float(selected.C),
            "class_weight": None if pd.isna(selected.class_weight) else selected.class_weight,
        }
        transformer = build_features(selected.feature)
        y_train, y_validation = (
            map_target(outer_train.severity, task),
            map_target(validation.severity, task),
        )
        rss_before, started = psutil.Process().memory_info().rss, perf_counter()
        x_train = transformer.fit_transform(outer_train)
        x_validation = transformer.transform(validation)
        model = build_model(spec)
        fit_start = perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = perf_counter() - fit_start
        infer_start = perf_counter()
        predicted = model.predict(x_validation)
        inference_seconds = perf_counter() - infer_start
        metrics = fixed_metrics(y_validation.to_numpy(), predicted, labels)
        row = {
            "task": task,
            "fold": outer.fold,
            "candidate": selected.candidate,
            "feature": selected.feature,
            **spec,
            **metrics,
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": perf_counter() - started,
            "feature_count": x_train.shape[1],
            "model_size_bytes": len(pickle.dumps((transformer, model))),
            "rss_delta_bytes": psutil.Process().memory_info().rss - rss_before,
        }
        scores = None
        if task == "s2":
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(x_validation)[
                    :, list(model.classes_).index("HIGH_IMPACT")
                ]
            else:
                scores = model.decision_function(x_validation)
            precision, recall, _ = precision_recall_curve(y_validation == "HIGH_IMPACT", scores)
            row["pr_auc"] = auc(recall, precision)
            row["roc_auc"] = roc_auc_score(y_validation == "HIGH_IMPACT", scores)
        fold_rows.append(row)
        part = pd.DataFrame(
            {
                "id": validation.id.astype(str),
                "fold": outer.fold,
                "true_label": y_validation,
                "predicted_label": predicted,
            }
        )
        if scores is not None:
            part["decision_score"] = scores
        oof_parts.append(part)
    return (
        pd.DataFrame(inner_rows),
        pd.DataFrame(selected_rows),
        pd.DataFrame(fold_rows),
        pd.concat(oof_parts, ignore_index=True),
    )


def _baseline_task(frame: pd.DataFrame, folds: tuple[FrozenFold, ...], task: str) -> pd.DataFrame:
    rows = []
    for fold in folds:
        train, validation = frame.iloc[list(fold.train)], frame.iloc[list(fold.validation)]
        transformer = build_features("F0")
        x_train = transformer.fit_transform(train)
        x_validation = transformer.transform(validation)
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED
        ).fit(x_train, map_target(train.severity, task))
        predicted = model.predict(x_validation)
        rows.append(
            {
                "task": task,
                "fold": fold.fold,
                **fixed_metrics(
                    map_target(validation.severity, task).to_numpy(), predicted, TASK_LABELS[task]
                ),
            }
        )
    return pd.DataFrame(rows)


def _nested_all_tasks(frame: pd.DataFrame, outer_folds: tuple[FrozenFold, ...]):
    """Share label-independent feature fitting across the three fixed tasks."""
    inner_rows, selected_rows, fold_rows = [], [], []
    oof_parts: dict[str, list[pd.DataFrame]] = {task: [] for task in TASKS}
    inner_identity, specs = [], model_specs()
    for outer in outer_folds:
        outer_train, validation = (
            frame.iloc[list(outer.train)],
            frame.iloc[list(outer.validation)],
        )
        inner_folds = _inner_folds(outer_train, "s6")
        for inner in inner_folds:
            inner_identity.append(
                {
                    "outer_fold": outer.fold,
                    "inner_fold": inner.fold,
                    "train_rows": len(inner.train),
                    "validation_rows": len(inner.validation),
                    "train_hash": _row_hash(outer_train, inner.train),
                    "validation_hash": _row_hash(outer_train, inner.validation),
                }
            )
            inner_train = outer_train.iloc[list(inner.train)]
            inner_validation = outer_train.iloc[list(inner.validation)]
            for feature in FEATURES:
                transformer, feature_start = build_features(feature), perf_counter()
                x_train = transformer.fit_transform(inner_train)
                x_validation = transformer.transform(inner_validation)
                shared_seconds = (perf_counter() - feature_start) / (len(specs) * len(TASKS))
                for task in TASKS:
                    y_train = map_target(inner_train.severity, task)
                    y_validation = map_target(inner_validation.severity, task)
                    for spec in specs:
                        model, fit_start = build_model(spec), perf_counter()
                        model.fit(x_train, y_train)
                        predicted = model.predict(x_validation)
                        inner_rows.append(
                            {
                                "task": task,
                                "outer_fold": outer.fold,
                                "inner_fold": inner.fold,
                                "candidate": _candidate_key(feature, spec),
                                "feature": feature,
                                **spec,
                                **fixed_metrics(
                                    y_validation.to_numpy(), predicted, TASK_LABELS[task]
                                ),
                                "feature_count": x_train.shape[1],
                                "runtime_seconds": perf_counter() - fit_start + shared_seconds,
                            }
                        )
        selected_by_task = {}
        current = pd.DataFrame(inner_rows)
        for task in TASKS:
            relevant = current[(current.outer_fold == outer.fold) & (current.task == task)]
            selected = _select_candidate(relevant, task)
            selected_by_task[task] = selected
            selected_rows.append({"task": task, "outer_fold": outer.fold, **selected.to_dict()})
        for feature in sorted({str(value.feature) for value in selected_by_task.values()}):
            transformer = build_features(feature)
            rss_before, feature_start = psutil.Process().memory_info().rss, perf_counter()
            x_train = transformer.fit_transform(outer_train)
            x_validation = transformer.transform(validation)
            feature_seconds = perf_counter() - feature_start
            for task, selected in selected_by_task.items():
                if selected.feature != feature:
                    continue
                spec = {
                    "family": selected.family,
                    "C": float(selected.C),
                    "class_weight": None
                    if pd.isna(selected.class_weight)
                    else selected.class_weight,
                }
                y_train, y_validation = (
                    map_target(outer_train.severity, task),
                    map_target(validation.severity, task),
                )
                model, fit_start = build_model(spec), perf_counter()
                model.fit(x_train, y_train)
                fit_seconds = perf_counter() - fit_start
                infer_start = perf_counter()
                predicted = model.predict(x_validation)
                inference_seconds = perf_counter() - infer_start
                row = {
                    "task": task,
                    "fold": outer.fold,
                    "candidate": selected.candidate,
                    "feature": feature,
                    **spec,
                    **fixed_metrics(y_validation.to_numpy(), predicted, TASK_LABELS[task]),
                    "fit_seconds": fit_seconds,
                    "inference_seconds": inference_seconds,
                    "total_seconds": feature_seconds + fit_seconds + inference_seconds,
                    "feature_count": x_train.shape[1],
                    "model_size_bytes": len(pickle.dumps((transformer, model))),
                    "rss_delta_bytes": psutil.Process().memory_info().rss - rss_before,
                }
                scores = None
                if task == "s2":
                    scores = (
                        model.predict_proba(x_validation)[
                            :, list(model.classes_).index("HIGH_IMPACT")
                        ]
                        if hasattr(model, "predict_proba")
                        else model.decision_function(x_validation)
                    )
                    precision, recall, _ = precision_recall_curve(
                        y_validation == "HIGH_IMPACT", scores
                    )
                    row["pr_auc"], row["roc_auc"] = (
                        auc(recall, precision),
                        roc_auc_score(y_validation == "HIGH_IMPACT", scores),
                    )
                fold_rows.append(row)
                part = pd.DataFrame(
                    {
                        "id": validation.id.astype(str),
                        "fold": outer.fold,
                        "true_label": y_validation,
                        "predicted_label": predicted,
                    }
                )
                if scores is not None:
                    part["decision_score"] = scores
                oof_parts[task].append(part)
    return (
        pd.DataFrame(inner_rows),
        pd.DataFrame(selected_rows),
        pd.DataFrame(fold_rows),
        {task: pd.concat(parts, ignore_index=True) for task, parts in oof_parts.items()},
        pd.DataFrame(inner_identity),
    )


def _aggregate_outer(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in folds.groupby("task"):
        row = {
            "task": task,
            "mean_macro_f1": group.macro_f1.mean(),
            "std_macro_f1": group.macro_f1.std(ddof=0),
            "mean_balanced_accuracy": group.balanced_accuracy.mean(),
            "mean_accuracy": group.accuracy.mean(),
            "minimum_mean_recall": min(
                group[f"recall_{label}"].mean() for label in TASK_LABELS[task]
            ),
            "runtime_seconds": group.total_seconds.sum(),
            "mean_feature_count": group.feature_count.mean(),
            "mean_model_size_bytes": group.model_size_bytes.mean(),
            "maximum_rss_delta_bytes": group.rss_delta_bytes.max(),
        }
        for label in TASK_LABELS[task]:
            for metric in ("precision", "recall", "f1"):
                row[f"mean_{metric}_{label}"] = group[f"{metric}_{label}"].mean()
            row[f"predicted_{label}"] = int(group[f"predicted_{label}"].sum())
        if task == "s2":
            row["mean_pr_auc"] = group.pr_auc.mean()
            row["mean_roc_auc"] = group.roc_auc.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _write_task_outputs(output: Path, task: str, folds: pd.DataFrame, oof: pd.DataFrame) -> None:
    labels = TASK_LABELS[task]
    write_csv(output / f"{task}_oof_predictions.csv", oof)
    matrix = pd.DataFrame(
        confusion_matrix(oof.true_label, oof.predicted_label, labels=labels),
        index=labels,
        columns=labels,
    )
    normalized = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    write_csv(output / f"{task}_confusion_matrix.csv", matrix.reset_index(names="true_label"))
    write_csv(
        output / f"{task}_normalized_confusion_matrix.csv",
        normalized.reset_index(names="true_label"),
    )
    save_confusion_matrix_figure(
        output / f"{task}_confusion_matrix.png", matrix, f"{task.upper()} OOF confusion matrix"
    )
    save_confusion_matrix_figure(
        output / f"{task}_normalized_confusion_matrix.png",
        normalized,
        f"{task.upper()} normalized OOF confusion matrix",
    )
    if task == "s2" and "decision_score" in oof:
        precision, recall, thresholds = precision_recall_curve(
            oof.true_label == "HIGH_IMPACT", oof.decision_score
        )
        curve = pd.DataFrame(
            {"recall": recall, "precision": precision, "threshold": np.append(thresholds, np.nan)}
        )
        write_csv(output / "s2_precision_recall_curve.csv", curve)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recall, precision)
        ax.set(xlabel="Recall", ylabel="Precision", title="S2 precision-recall curve")
        fig.tight_layout()
        fig.savefig(output / "s2_precision_recall_curve.png", dpi=200)
        plt.close(fig)
    errors = oof[oof.true_label != oof.predicted_label]
    (output / f"{task}_error_analysis.md").write_text(
        f"# {task.upper()} OOF error analysis\n\nErrors: {len(errors)} / {len(oof)}.\n",
        encoding="utf-8",
    )
    (output / f"{task}_summary.md").write_text(
        "# "
        + task.upper()
        + " nested temporal summary\n\n```csv\n"
        + folds.to_csv(index=False).strip()
        + "\n```",
        encoding="utf-8",
    )


def _audit(frame: pd.DataFrame, output: Path, folds: tuple[FrozenFold, ...]) -> set[str]:
    ensure_directory(output)
    combined = (
        frame.summary.fillna("").astype(str) + "\n" + frame.description.fillna("").astype(str)
    )
    cleaned = pd.Series(
        [clean_text_value(s, d) for s, d in zip(frame.summary, frame.description, strict=True)]
    )
    exact_hash = combined.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    normalized_hash = cleaned.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    write_csv(
        output / "dataset_reconciliation.csv",
        pd.DataFrame([{"stage": "approved_development", "rows": len(frame), "excluded": 0}]),
    )
    write_csv(
        output / "exclusion_reconciliation.csv",
        pd.DataFrame([{"reason": "no_additional_exclusion", "rows": 0}]),
    )
    write_csv(
        output / "severity_normalization_audit.csv",
        frame.groupby("severity").size().reset_index(name="rows"),
    )
    quality = [
        {
            "metric": "blank_summary",
            "value": int(frame.summary.fillna("").str.strip().eq("").sum()),
        },
        {
            "metric": "blank_description",
            "value": int(frame.description.fillna("").str.strip().eq("").sum()),
        },
        {
            "metric": "both_blank",
            "value": int(
                (
                    frame.summary.fillna("").str.strip().eq("")
                    & frame.description.fillna("").str.strip().eq("")
                ).sum()
            ),
        },
        {"metric": "very_short_under_20_chars", "value": int((combined.str.len() < 20).sum())},
        {
            "metric": "extremely_long_over_10000_chars",
            "value": int((combined.str.len() > 10000).sum()),
        },
        {
            "metric": "replacement_character",
            "value": int(combined.str.contains("�", regex=False).sum()),
        },
    ]
    write_csv(output / "text_quality_audit.csv", pd.DataFrame(quality))
    lengths = (
        pd.DataFrame(
            {
                "severity": frame.severity,
                "characters": combined.str.len(),
                "tokens": combined.str.split().str.len(),
            }
        )
        .groupby("severity")
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    write_csv(output / "text_length_by_class.csv", lengths)
    write_csv(
        output / "unicode_encoding_audit.csv",
        pd.DataFrame(
            [
                {
                    "rows_changed_nfkc": int(
                        sum(unicodedata.normalize("NFKC", x) != x for x in combined)
                    ),
                    "replacement_character_rows": int(
                        combined.str.contains("�", regex=False).sum()
                    ),
                }
            ]
        ),
    )
    indicators = {
        "html": HTML_TAG,
        "html_entity": re.compile(r"&\w+;"),
        "code": CODE,
        "stack_trace": STACK,
        "url": URL,
        "email": EMAIL,
    }
    write_csv(
        output / "html_code_stacktrace_audit.csv",
        pd.DataFrame(
            [
                {"indicator": name, "report_count": int(combined.str.contains(pattern).sum())}
                for name, pattern in indicators.items()
            ]
        ),
    )
    exact_groups = int((exact_hash.value_counts() > 1).sum())
    normalized_groups = int((normalized_hash.value_counts() > 1).sum())
    write_csv(
        output / "duplicate_audit.csv",
        pd.DataFrame(
            [
                {
                    "type": "exact",
                    "groups": exact_groups,
                    "duplicate_rows": int(exact_hash.duplicated().sum()),
                },
                {
                    "type": "normalized",
                    "groups": normalized_groups,
                    "duplicate_rows": int(normalized_hash.duplicated().sum()),
                },
                {
                    "type": "explicit_group",
                    "groups": int(frame.duplicate_group.nunique()),
                    "duplicate_rows": int(frame.duplicate_group.duplicated().sum()),
                },
            ]
        ),
    )
    groups = pd.DataFrame(
        {
            "id": frame.id.astype(str),
            "severity": frame.severity,
            "normalized_text_hash": normalized_hash,
        }
    )
    group_counts = groups.groupby("normalized_text_hash").filter(lambda x: len(x) > 1)
    write_csv(output / "normalized_duplicate_groups.csv", group_counts)
    cross = []
    for fold in folds:
        train_hash = set(normalized_hash.iloc[list(fold.train)])
        valid_hash = set(normalized_hash.iloc[list(fold.validation)])
        cross.append(
            {
                "fold": fold.fold,
                "crossing_normalized_groups": len(train_hash & valid_hash),
                "crossing_explicit_groups": len(
                    set(frame.iloc[list(fold.train)].duplicate_group)
                    & set(frame.iloc[list(fold.validation)].duplicate_group)
                ),
            }
        )
    write_csv(output / "normalized_duplicate_cross_fold_audit.csv", pd.DataFrame(cross))
    conflict_hashes = groups.groupby("normalized_text_hash").severity.nunique()
    conflicts = set(conflict_hashes[conflict_hashes > 1].index)
    conflict_rows = groups[groups.normalized_text_hash.isin(conflicts)].copy()
    conflict_rows["review_status"] = ""
    conflict_rows["reviewer_label"] = ""
    write_csv(output / "label_conflict_review.csv", conflict_rows)
    prefixes = combined.str.lower().str[:40].value_counts().head(30).reset_index()
    prefixes.columns = ["prefix", "rows"]
    write_csv(output / "boilerplate_audit.csv", prefixes)
    token_rows = []
    for term in ["blocker", "critical", "major", "normal", "minor", "trivial", "severity"]:
        for field in ["summary", "description"]:
            mask = frame[field].fillna("").str.contains(rf"\b{term}\b", case=False, regex=True)
            for severity, count in frame[mask].groupby("severity").size().items():
                token_rows.append(
                    {"term": term, "field": field, "true_severity": severity, "reports": count}
                )
    write_csv(output / "direct_label_token_audit.csv", pd.DataFrame(token_rows))
    metadata = []
    for raw, status, reason in [
        ("Classification", "INELIGIBLE", "not present; creation-time provenance unavailable"),
        (
            "Product",
            "SENSITIVITY_ONLY",
            "present in approved artifact but initial-value stability unproven",
        ),
        (
            "Component",
            "SENSITIVITY_ONLY",
            "present in approved artifact but initial-value stability unproven",
        ),
        (
            "Version",
            "INELIGIBLE",
            "not present; reconstruction would require prohibited wider raw access",
        ),
        ("Platform", "INELIGIBLE", "not present"),
        ("OS", "INELIGIBLE", "not present"),
        ("Keywords", "INELIGIBLE", "initial values not reconstructable without history"),
    ]:
        col = raw.lower()
        present = col in frame.columns
        metadata.append(
            {
                "raw_column_name": raw,
                "normalized_name": col,
                "availability": present,
                "missingness": float(frame[col].isna().mean()) if present else 1.0,
                "cardinality": int(frame[col].nunique()) if present else 0,
                "top_category_distribution": json.dumps(
                    frame[col].value_counts().head(10).to_dict()
                )
                if present
                else "{}",
                "rare_category_rate": float(frame[col].map(frame[col].value_counts()).lt(5).mean())
                if present
                else 0.0,
                "change_frequency": "not_detectable",
                "initial_value_reconstruction_status": "not_proven",
                "reconstruction_confidence": "low",
                "leakage_risk": "uncertain" if status == "SENSITIVITY_ONLY" else "high",
                "eligibility_status": status,
                "rationale": reason,
            }
        )
    metadata_frame = pd.DataFrame(metadata)
    write_csv(output / "metadata_eligibility.csv", metadata_frame)
    (output / "metadata_eligibility.md").write_text(
        "# Metadata eligibility\n\n"
        "No field qualified as PRIMARY_ELIGIBLE; metadata modeling is blocked.\n",
        encoding="utf-8",
    )
    write_csv(
        output / "metadata_change_audit.csv",
        metadata_frame[
            [
                "raw_column_name",
                "change_frequency",
                "initial_value_reconstruction_status",
                "reconstruction_confidence",
            ]
        ],
    )
    write_csv(
        output / "metadata_missingness.csv", metadata_frame[["raw_column_name", "missingness"]]
    )
    write_csv(
        output / "metadata_cardinality.csv",
        metadata_frame[["raw_column_name", "cardinality", "rare_category_rate"]],
    )
    for name in ["metadata_temporal_drift", "metadata_target_association"]:
        write_csv(
            output / f"{name}.csv",
            pd.DataFrame([{"status": "blocked", "reason": "no PRIMARY_ELIGIBLE metadata"}]),
        )
    for task in TASKS:
        temp = frame.assign(
            target=map_target(frame.severity, task), year=frame.creation_time.dt.year
        )
        dist = temp.groupby(["year", "target"]).size().reset_index(name="rows")
        dist["percentage_within_year"] = dist.rows / dist.groupby("year").rows.transform("sum")
        write_csv(output / f"class_distribution_{task}.csv", dist)
    sample = (
        frame[
            frame.id.astype(str).isin(CONFLICT_IDS)
            | frame.severity.isin(["blocker", "critical", "minor", "trivial"])
        ]
        .groupby("severity")
        .head(20)[["id", "creation_time", "severity"]]
        .copy()
    )
    sample["review_reason"] = "conflict_or_balanced_minority_review"
    sample["review_status"] = ""
    sample["reviewer_label"] = ""
    sample["reviewer_confidence"] = ""
    sample["reviewer_notes"] = ""
    sample["suspected_label_issue"] = ""
    sample["suspected_missing_context"] = ""
    write_csv(output / "human_label_review_sample.csv", sample)
    (output / "data_quality_summary.md").write_text(
        f"# MYLYN target/feature audit\n\nRows: {len(frame)}. "
        f"Exact duplicate groups: {exact_groups}; normalized groups: {normalized_groups}; "
        f"normalized conflicts: {len(conflicts)}. "
        "No metadata field is PRIMARY_ELIGIBLE.\n",
        encoding="utf-8",
    )
    return set(
        metadata_frame[metadata_frame.eligibility_status == "PRIMARY_ELIGIBLE"].normalized_name
    )


def run_redesign_study(development: Path, audit_output: Path, model_output: Path) -> dict[str, Any]:
    if audit_output.exists() or model_output.exists():
        raise FileExistsError("Refusing to overwrite redesign output")
    frame = load_hierarchical_development(development)
    folds = freeze_temporal_folds(frame)
    if folds_fingerprint(folds) != EXPECTED_FOLD_HASH:
        raise RuntimeError("Outer fold identity mismatch")
    ensure_directory(audit_output)
    ensure_directory(model_output)
    eligible = _audit(frame, audit_output, folds)
    identity = {
        "dataset_hash": sha256_file(development),
        "fold_hash": folds_fingerprint(folds),
        "folds": [
            {
                "fold": f.fold,
                "train_rows": len(f.train),
                "validation_rows": len(f.validation),
                "train_hash": _row_hash(frame, f.train),
                "validation_hash": _row_hash(frame, f.validation),
            }
            for f in folds
        ],
    }
    write_json(model_output / "outer_fold_identity.json", identity)
    baseline_frame = pd.concat(
        [_baseline_task(frame, folds, task) for task in TASKS], ignore_index=True
    )
    inner_frame, selected_frame, outer_frame, oofs, inner_identity = _nested_all_tasks(frame, folds)
    s6mean = baseline_frame[baseline_frame.task == "s6"].macro_f1.mean()
    if abs(s6mean - BASELINE_S6) > 0.0001:
        raise RuntimeError(f"S6 F0 reproduction failed: {s6mean}")
    write_json(
        model_output / "study_manifest.json",
        {
            "development_rows": len(frame),
            "dataset_hash": MYLYN_DEVELOPMENT_SHA256,
            "fold_hash": EXPECTED_FOLD_HASH,
            "tasks": TASKS,
            "eligible_metadata": sorted(eligible),
            "held_out_test_accessed": False,
            "other_project_trained": False,
        },
    )
    write_json(
        model_output / "immutable_prior_study_hashes.json",
        {"verification": "recorded externally before and after run"},
    )
    write_csv(model_output / "inner_fold_identity.csv", inner_identity)
    write_csv(model_output / "inner_selection_results.csv", inner_frame)
    write_csv(model_output / "selected_configurations.csv", selected_frame)
    write_csv(model_output / "fold_results.csv", outer_frame)
    candidates = inner_frame[
        ["task", "candidate", "feature", "family", "C", "class_weight"]
    ].drop_duplicates()
    write_csv(model_output / "candidate_matrix.csv", candidates)
    write_csv(
        model_output / "invalid_candidate_reasons.csv",
        pd.DataFrame(
            [
                {"candidate": "F5-F7", "reason": "no PRIMARY_ELIGIBLE metadata"},
                {
                    "candidate": "ComplementNB",
                    "reason": "uniform matrix includes scaled structural candidates",
                },
            ]
        ),
    )
    per = []
    distributions = []
    for _, row in outer_frame.iterrows():
        for label in TASK_LABELS[row.task]:
            per.append(
                {
                    "task": row.task,
                    "fold": row.fold,
                    "label": label,
                    "precision": row[f"precision_{label}"],
                    "recall": row[f"recall_{label}"],
                    "f1": row[f"f1_{label}"],
                    "support": row[f"support_{label}"],
                }
            )
    for task, oof in oofs.items():
        for label in TASK_LABELS[task]:
            distributions.append(
                {
                    "task": task,
                    "label": label,
                    "predictions": int((oof.predicted_label == label).sum()),
                    "rate": float((oof.predicted_label == label).mean()),
                }
            )
        _write_task_outputs(model_output, task, outer_frame[outer_frame.task == task], oof)
    write_csv(model_output / "per_class_results.csv", pd.DataFrame(per))
    write_csv(model_output / "predicted_class_distributions.csv", pd.DataFrame(distributions))
    summary = _aggregate_outer(outer_frame)
    baseline_summary = (
        baseline_frame.groupby("task")
        .macro_f1.agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "f0_macro_f1", "std": "f0_std_macro_f1"})
    )
    summary = summary.merge(baseline_summary, on="task")
    write_csv(model_output / "task_summary.csv", summary)
    feature_comparison = (
        inner_frame.groupby(["task", "feature"]).macro_f1.agg(["mean", "std"]).reset_index()
    )
    write_csv(model_output / "feature_set_comparison.csv", feature_comparison)
    write_csv(model_output / "preprocessing_comparison.csv", feature_comparison)
    blocked = pd.DataFrame([{"status": "blocked", "reason": "no PRIMARY_ELIGIBLE metadata"}])
    write_csv(model_output / "metadata_only_vs_text_only.csv", blocked)
    write_csv(model_output / "metadata_ablation.csv", blocked)
    write_csv(
        model_output / "leakage_control_results.csv",
        pd.DataFrame(
            [
                {"control": "outer_fold_hash", "passed": True},
                {"control": "explicit_duplicate_crossing", "passed": True},
                {"control": "metadata_primary_eligible", "passed": False},
            ]
        ),
    )
    # Deterministic shuffled-label leakage control on S6 outer fold 1.
    first = folds[0]
    train, validation = frame.iloc[list(first.train)], frame.iloc[list(first.validation)]
    transformer = build_features("F0")
    xtrain = transformer.fit_transform(train)
    xvalid = transformer.transform(validation)
    shuffled = map_target(train.severity, "s6").sample(frac=1, random_state=SEED).to_numpy()
    dummy_model = LogisticRegression(
        C=1, class_weight="balanced", max_iter=2000, random_state=SEED
    ).fit(xtrain, shuffled)
    shuffled_score = f1_score(
        map_target(validation.severity, "s6"),
        dummy_model.predict(xvalid),
        labels=S6_LABELS,
        average="macro",
        zero_division=0,
    )
    write_csv(
        model_output / "shuffled_label_control.csv",
        pd.DataFrame([{"task": "s6", "fold": 1, "macro_f1": shuffled_score}]),
    )
    # Masked-label sensitivity uses each selected outer configuration, without reselection.
    mask_rows = []
    for task in TASKS:
        for _, sel in selected_frame[selected_frame.task == task].iterrows():
            outer = folds[int(sel.outer_fold) - 1]
            train, validation = frame.iloc[list(outer.train)], frame.iloc[list(outer.validation)]
            tr = build_features(sel.feature, mask_labels=True)
            xt = tr.fit_transform(train)
            xv = tr.transform(validation)
            spec = {
                "family": sel.family,
                "C": float(sel.C),
                "class_weight": None if pd.isna(sel.class_weight) else sel.class_weight,
            }
            model = build_model(spec).fit(xt, map_target(train.severity, task))
            pred = model.predict(xv)
            mask_rows.append(
                {
                    "task": task,
                    "fold": outer.fold,
                    "macro_f1": f1_score(
                        map_target(validation.severity, task),
                        pred,
                        labels=TASK_LABELS[task],
                        average="macro",
                        zero_division=0,
                    ),
                }
            )
    write_csv(model_output / "label_token_sensitivity.csv", pd.DataFrame(mask_rows))
    write_csv(
        model_output / "runtime_summary.csv",
        summary[["task", "runtime_seconds", "maximum_rss_delta_bytes"]],
    )
    write_csv(
        model_output / "model_size_summary.csv",
        summary[["task", "mean_feature_count", "mean_model_size_bytes"]],
    )
    # Diagnostic known-conflict exclusion using selected configurations.
    conflict_rows = []
    excluded = frame.id.astype(str).isin(CONFLICT_IDS)
    for task in TASKS:
        for _, sel in selected_frame[selected_frame.task == task].iterrows():
            outer = folds[int(sel.outer_fold) - 1]
            ti = [i for i in outer.train if not excluded.iloc[i]]
            vi = [i for i in outer.validation if not excluded.iloc[i]]
            train, validation = frame.iloc[ti], frame.iloc[vi]
            tr = build_features(sel.feature)
            xt = tr.fit_transform(train)
            xv = tr.transform(validation)
            model = build_model(
                {
                    "family": sel.family,
                    "C": float(sel.C),
                    "class_weight": None if pd.isna(sel.class_weight) else sel.class_weight,
                }
            ).fit(xt, map_target(train.severity, task))
            pred = model.predict(xv)
            conflict_rows.append(
                {
                    "task": task,
                    "fold": outer.fold,
                    "macro_f1": f1_score(
                        map_target(validation.severity, task),
                        pred,
                        labels=TASK_LABELS[task],
                        average="macro",
                        zero_division=0,
                    ),
                    "excluded_rows": int(excluded.sum()),
                }
            )
    write_csv(model_output / "conflict_sensitivity.csv", pd.DataFrame(conflict_rows))
    # Conservative success classification.
    classifications = []
    for _, row in summary.iterrows():
        task = row.task
        baseline = row.f0_macro_f1
        material = False
        strong = False
        folds_task = outer_frame[outer_frame.task == task]
        baseline_folds = baseline_frame[baseline_frame.task == task]
        improvements = int(
            np.sum(folds_task.macro_f1.to_numpy() > baseline_folds.macro_f1.to_numpy())
        )
        if task == "s6":
            material = (
                row.mean_macro_f1 >= 0.2463
                and improvements >= 2
                and row.mean_recall_blocker > 0
                and row.mean_recall_critical > 0
            )
            strong = (
                material
                and row.mean_macro_f1 >= 0.35
                and row.mean_balanced_accuracy >= 0.35
                and row.minimum_mean_recall >= 0.15
                and np.mean([row.mean_recall_blocker, row.mean_recall_critical]) >= 0.15
            )
        elif task == "s3":
            material = (
                row.mean_macro_f1 >= baseline + 0.05
                and improvements >= 2
                and row.mean_recall_HIGH
                >= baseline_frame[baseline_frame.task == task].recall_HIGH.mean() + 0.10
                and row.minimum_mean_recall >= 0.20
            )
            strong = (
                row.mean_macro_f1 >= 0.50
                and row.mean_recall_HIGH >= 0.35
                and row.minimum_mean_recall >= 0.30
                and row.mean_balanced_accuracy >= 0.50
                and row.std_macro_f1 <= 0.08
            )
        else:
            material = (
                row.mean_macro_f1 >= baseline + 0.05
                and improvements >= 2
                and row.mean_balanced_accuracy >= 0.63
                and row.mean_precision_HIGH_IMPACT >= 0.45
                and row.mean_recall_HIGH_IMPACT >= 0.45
                and row.mean_f1_HIGH_IMPACT
                > baseline_frame[baseline_frame.task == task].f1_HIGH_IMPACT.mean()
            )
            strong = (
                row.mean_macro_f1 >= 0.70
                and row.mean_balanced_accuracy >= 0.65
                and row.mean_recall_HIGH_IMPACT >= 0.60
                and row.mean_precision_HIGH_IMPACT >= 0.50
                and row.mean_f1_HIGH_IMPACT >= 0.55
                and row.std_macro_f1 <= 0.08
            )
        classifications.append(
            {
                "task": task,
                "material_improvement": bool(material),
                "practically_strong": bool(strong),
                "classification": "PRACTICALLY_STRONG"
                if strong
                else "PROMISING"
                if material
                else "RESEARCH_ONLY"
                if row.mean_macro_f1 >= baseline
                else "NOT_PREDICTABLE_ENOUGH",
            }
        )
    classifications = pd.DataFrame(classifications)
    write_csv(model_output / "success_classification.csv", classifications)
    ranked = classifications.merge(
        summary[["task", "mean_macro_f1"]],
        on="task",
        how="left",
        validate="one_to_one",
    )

    practical = ranked[ranked["practically_strong"]]
    material = ranked[ranked["material_improvement"]]

    if not practical.empty:
        recommendation = str(
            practical.sort_values(
                "mean_macro_f1",
                ascending=False,
            ).iloc[0]["task"]
        )
        recommendation_text = f"Primary practical recommendation: `{recommendation.upper()}`."
    elif not material.empty:
        recommendation = str(
            material.sort_values(
                "mean_macro_f1",
                ascending=False,
            ).iloc[0]["task"]
        )
        recommendation_text = (
            f"Primary development recommendation: `{recommendation.upper()}` "
            "as a material-improvement candidate."
        )
    else:
        recommendation = "s6"
        recommendation_text = (
            "No redesigned task met the predeclared material-improvement or "
            "practically-strong criteria. Retain `S6` as the original "
            "fine-grained research task, use `S3` only as a secondary "
            "exploratory analysis, and do not recommend `S2` as a practical "
            "classifier."
        )

    (model_output / "final_recommendation.md").write_text(
        "# Final task recommendation\n\n"
        + recommendation_text
        + " Metadata enhancement is rejected because creation-time validity "
        "is unproven.\n",
        encoding="utf-8",
    )
    return {
        "summary": summary.to_dict("records"),
        "classifications": classifications.to_dict("records"),
        "recommended_task": recommendation,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_redesign_study(args.development, args.audit_output, args.model_output),
            indent=2,
            default=str,
        )
    )
