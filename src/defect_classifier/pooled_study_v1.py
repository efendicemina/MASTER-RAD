"""Compute-efficient, preregistered pooled Eclipse severity study v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from scipy import sparse
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
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from .development_study import FrozenFold
from .transfer_study_v2 import (
    S2ThresholdInfeasible,
    exact_identity,
    fit_calibrator,
    select_s2_threshold,
    software_clean,
    software_tokenize,
)
from .utils import package_versions, sha256_file

PROJECTS = ("BIRT", "CDT", "Equinox", "JDT", "MYLYN", "PDE", "Papyrus", "Platform", "TPTP")
TASK_LABELS = {
    "s6": ["blocker", "critical", "major", "minor", "normal", "trivial"],
    "s3": ["HIGH", "LOW", "MEDIUM"],
    "s2": ["HIGH_IMPACT", "LOWER_IMPACT"],
}
TASK_MAPPINGS = {
    "s6": {label: label for label in TASK_LABELS["s6"]},
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
        "minor": "LOWER_IMPACT",
        "normal": "LOWER_IMPACT",
        "trivial": "LOWER_IMPACT",
    },
}
INPUT_HASHES = {
    "BIRT": "a0e2c21c9095bcf250cc5d389177d2e6058b39cd589199316d235b00b532fa76",
    "CDT": "00cbe579b9a1a4606db80933154cb7a2498fc8931310805f033cb4558d725004",
    "Equinox": "f0ee1c9185c6edc119bae133fb64c81c3790559878f4f5279c12c189f337d5ba",
    "JDT": "0c1b597e9be4fce63a22cdac137e2f1f1657f8d6512fec689dc7be2233340c61",
    "PDE": "034ac8abccce6a5fcae49ad604faafd3f2f794c2d4dfd9a993430ef44e1d1808",
    "Papyrus": "b82272d153d1edefa03a82268099f878a4f3e0f9f5a652dbbf259e5f7cc366f4",
    "Platform": "21a82ca083594b286f226debf1bd98ad29063cef267d3809e6d160919d3bfdf1",
    "TPTP": "7aceea71ca29e14e11ce4953203acc5e924a75154715e100566543070b70eca7",
    "MYLYN": "8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5",
}
SEED = 42
SCHEMA_VERSION = "1.0"
STAGE1_KEEP = 6
STAGE1_MARGIN = 0.03
RESUME_LOCK_FIELDS = (
    "schema_version",
    "protocol_sha256",
    "protocol_commit",
    "input_hashes",
    "candidate_checksum",
    "seed",
    "projects",
    "task_mappings",
    "development_fingerprint",
    "locked_test_fingerprint",
    "fold_fingerprint",
    "packages",
    "canonical_command",
)


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    representation: str
    model: str
    c: float
    class_weight: str
    project_weighting: str


@dataclass(slots=True)
class PartitionedData:
    development: pd.DataFrame
    locked_test: pd.DataFrame
    audit: pd.DataFrame
    purged_test_rows: int


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(path, frame.to_csv(index=False))


def validate_resume_manifest(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    for field in RESUME_LOCK_FIELDS:
        if existing.get(field) != expected.get(field):
            raise RuntimeError(f"Resume provenance mismatch: {field}")


def assert_privacy_safe_columns(frame: pd.DataFrame) -> None:
    forbidden = {
        "id",
        "issue_id",
        "global_report_key",
        "summary",
        "description",
        "reporter",
        "assignee",
        "issue_url",
    }
    overlap = forbidden & {str(column).lower() for column in frame.columns}
    if overlap:
        raise ValueError(f"Privacy-unsafe output columns: {sorted(overlap)}")


def map_labels(labels: Iterable[str], task: str) -> np.ndarray:
    mapping = TASK_MAPPINGS[task]
    return np.asarray([mapping[str(value).strip().lower()] for value in labels])


def candidate_grid() -> list[Candidate]:
    rows: list[Candidate] = []
    index = 0
    for representation in ("word", "char", "word_char"):
        for model, values in (("LinearSVC", (0.25, 1.0)), ("LogisticRegression", (0.5,))):
            for c in values:
                for class_weight in ("none", "balanced"):
                    for project_weighting in ("none", "inverse_sqrt"):
                        rows.append(
                            Candidate(
                                f"P-{index:03d}",
                                representation,
                                model,
                                c,
                                class_weight,
                                project_weighting,
                            )
                        )
                        index += 1
    return rows


def candidate_checksum() -> str:
    payload = json.dumps([asdict(row) for row in candidate_grid()], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_parquet(path: Path) -> pd.DataFrame:
    columns = [
        "issue_id",
        "global_report_key",
        "source_project",
        "summary",
        "description",
        "severity",
        "creation_time",
        "dupe_of",
        "duplicate_group_id",
    ]
    table = pq.read_table(path, columns=columns)
    return table.to_pandas()


def _row_key(project: str, identifier: Any) -> str:
    return hashlib.sha256(f"{project}:{identifier}".encode()).hexdigest()[:24]


def load_allowed_universe(processed_root: Path, mylyn_development: Path) -> pd.DataFrame:
    """Load only legacy development; the reserved MYLYN test is never opened."""
    if mylyn_development.name != "development_split.csv":
        raise ValueError("MYLYN input must be the approved development_split.csv")
    frames = []
    valid = set(TASK_LABELS["s6"])
    for project in PROJECTS:
        if project == "MYLYN":
            if sha256_file(mylyn_development) != INPUT_HASHES[project]:
                raise ValueError("MYLYN development SHA-256 mismatch")
            frame = pd.read_csv(mylyn_development)
            frame = frame.rename(
                columns={"id": "issue_id", "duplicate_group": "duplicate_group_id"}
            )
            frame["source_project"] = project
            frame["global_report_key"] = [f"MYLYN:{value}" for value in frame.issue_id]
            frame["dupe_of"] = ""
        else:
            path = processed_root / f"{project}.parquet"
            if sha256_file(path) != INPUT_HASHES[project]:
                raise ValueError(f"{project} SHA-256 mismatch")
            frame = _load_parquet(path)
        frame["severity"] = frame.severity.astype(str).str.strip().str.lower()
        frame["creation_time"] = pd.to_datetime(frame.creation_time, utc=True, errors="coerce")
        frame = frame.loc[frame.severity.isin(valid) & frame.creation_time.notna()].copy()
        frame = frame.sort_values(["creation_time", "global_report_key"], kind="mergesort")
        if project != "MYLYN":
            frame = frame.iloc[: int(np.floor(len(frame) * 0.8))].copy()
        frame["summary"] = frame.summary.fillna("").astype(str)
        frame["description"] = frame.description.fillna("").astype(str)
        frame["exact_text_hash"] = [
            exact_identity(summary, description)
            for summary, description in zip(frame.summary, frame.description, strict=True)
        ]
        frame["row_key"] = [
            _row_key(project, value) for value in frame.global_report_key.astype(str)
        ]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def connected_duplicate_groups(frame: pd.DataFrame) -> pd.Series:
    union = _UnionFind()
    row_tokens = []
    for row in frame.itertuples(index=False):
        row_token = f"row:{row.row_key}"
        exact_token = f"exact:{row.exact_text_hash}"
        union.union(row_token, exact_token)
        linked = str(getattr(row, "duplicate_group_id", "") or "").strip()
        if linked and linked.lower() != "nan":
            union.union(row_token, f"linked:{row.source_project}:{linked}")
        row_tokens.append(row_token)
    roots = [union.find(token) for token in row_tokens]
    return pd.Series(
        [hashlib.sha256(root.encode()).hexdigest()[:24] for root in roots], index=frame.index
    )


def partition_allowed_universe(frame: pd.DataFrame, train_fraction: float = 0.8) -> PartitionedData:
    data = frame.copy()
    data["duplicate_group"] = connected_duplicate_groups(data)
    parts = []
    audit_rows = []
    for project in PROJECTS:
        project_frame = data.loc[data.source_project.eq(project)].sort_values(
            ["creation_time", "row_key"], kind="mergesort"
        )
        cut = int(np.floor(len(project_frame) * train_fraction))
        if cut <= 0 or cut >= len(project_frame):
            raise ValueError(f"{project}: empty pooled development or locked test")
        assigned = project_frame.copy()
        assigned["partition"] = "development"
        assigned.iloc[cut:, assigned.columns.get_loc("partition")] = "locked_test"
        parts.append(assigned)
    combined = pd.concat(parts, ignore_index=True)
    development_groups = set(combined.loc[combined.partition.eq("development"), "duplicate_group"])
    overlap = combined.partition.eq("locked_test") & combined.duplicate_group.isin(
        development_groups
    )
    purged = int(overlap.sum())
    combined.loc[overlap, "partition"] = "purged_from_test"
    development = combined.loc[combined.partition.eq("development")].copy()
    locked = combined.loc[combined.partition.eq("locked_test")].copy()
    if set(development.row_key) & set(locked.row_key):
        raise RuntimeError("Development/locked row overlap")
    if set(development.duplicate_group) & set(locked.duplicate_group):
        raise RuntimeError("Duplicate group crosses development/locked boundary")
    for project in PROJECTS:
        allowed = combined.loc[combined.source_project.eq(project)]
        for partition in ("development", "locked_test", "purged_from_test"):
            part = allowed.loc[allowed.partition.eq(partition)]
            audit_rows.append(
                {
                    "project": project,
                    "partition": partition,
                    "rows": len(part),
                    "minimum_creation_time": part.creation_time.min(),
                    "maximum_creation_time": part.creation_time.max(),
                    "duplicate_groups": part.duplicate_group.nunique(),
                }
            )
    return PartitionedData(development, locked, pd.DataFrame(audit_rows), purged)


def project_class_weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(frame), dtype=float)
    counts = frame.source_project.value_counts()
    weights = frame.source_project.map(lambda value: 1.0 / np.sqrt(counts[value])).to_numpy()
    return weights / weights.mean()


def freeze_development_folds(frame: pd.DataFrame, n_splits: int = 3) -> tuple[FrozenFold, ...]:
    ordered = frame.sort_values(["creation_time", "row_key"], kind="mergesort").reset_index(
        drop=True
    )
    boundaries = np.linspace(0, len(ordered), n_splits + 2, dtype=int)
    folds = []
    for index in range(n_splits):
        train = np.arange(boundaries[0], boundaries[index + 1])
        validation = np.arange(boundaries[index + 1], boundaries[index + 2])
        train_groups = set(ordered.iloc[train].duplicate_group)
        validation = validation[~ordered.iloc[validation].duplicate_group.isin(train_groups)]
        if not len(train) or not len(validation):
            raise ValueError("Empty development fold after duplicate purge")
        folds.append(FrozenFold(index + 1, tuple(map(int, train)), tuple(map(int, validation))))
    return tuple(folds)


def _text(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            software_clean(f"{summary} [SEP] {description}", mask_labels=True)
            for summary, description in zip(frame.summary, frame.description, strict=True)
        ],
        index=frame.index,
    )


def build_vectorizer(representation: str, max_features: int | None = None):
    branch_features = max_features or (50_000 if representation == "word_char" else 75_000)
    word = TfidfVectorizer(
        tokenizer=software_tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=branch_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.98,
        max_features=branch_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    if representation == "word":
        return word
    if representation == "char":
        return char
    return FeatureUnion([("word", word), ("char", char)])


def _model(candidate: Candidate):
    class_weight = None if candidate.class_weight == "none" else "balanced"
    if candidate.model == "LinearSVC":
        return LinearSVC(C=candidate.c, class_weight=class_weight, random_state=SEED)
    return LogisticRegression(
        C=candidate.c,
        class_weight=class_weight,
        max_iter=2000,
        solver="lbfgs",
        random_state=SEED,
    )


def _scores(model, matrix: sparse.spmatrix) -> np.ndarray:
    if hasattr(model, "decision_function"):
        values = np.asarray(model.decision_function(matrix))
        return values if values.ndim == 2 else values.reshape(-1, 1)
    return np.asarray(model.predict_proba(matrix))


def fixed_metrics(truth: np.ndarray, predicted: np.ndarray, labels: list[str]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    return {
        "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(truth, predicted),
        "weighted_f1": f1_score(
            truth, predicted, labels=labels, average="weighted", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "minimum_class_recall": float(recall.min()),
        "per_class": [
            {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        ],
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }


def rank_candidates(rows: list[dict[str, Any]], keep: int = STAGE1_KEEP) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get("status") == "trained"]
    if not eligible:
        raise RuntimeError("All eligible candidates are invalid")
    leaders: dict[tuple[str, str], float] = {}
    for row in eligible:
        key = (row["representation"], row["model"])
        leaders[key] = max(leaders.get(key, -np.inf), row["macro_f1"])
    eligible = [
        row
        for row in eligible
        if row["macro_f1"] >= leaders[(row["representation"], row["model"])] - STAGE1_MARGIN
    ]
    return sorted(
        eligible,
        key=lambda row: (
            -row["macro_f1"],
            -row["project_macro_f1"],
            -row["minimum_class_recall"],
            -row["balanced_accuracy"],
            row.get("fold_macro_f1_sd", 0.0),
            row["feature_count"],
            row["runtime_seconds"],
            row["candidate_id"],
        ),
    )[:keep]


def _project_macro(
    frame: pd.DataFrame, truth: np.ndarray, predicted: np.ndarray, labels: list[str]
) -> float:
    scores = []
    for project in PROJECTS:
        mask = frame.source_project.to_numpy() == project
        if mask.any():
            scores.append(
                f1_score(
                    truth[mask], predicted[mask], labels=labels, average="macro", zero_division=0
                )
            )
    return float(np.mean(scores))


def evaluate_candidate(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    task: str,
    candidate: Candidate,
    max_features: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    model_training = training
    calibration = None
    if task == "s2":
        model_training, calibration = _single_screen_split(training)
    vectorizer = build_vectorizer(candidate.representation, max_features)
    x_train = vectorizer.fit_transform(_text(model_training)).tocsr().astype(np.float32)
    x_validation = vectorizer.transform(_text(validation)).tocsr().astype(np.float32)
    model = _model(candidate)
    truth_train = map_labels(model_training.severity, task)
    truth = map_labels(validation.severity, task)
    model.fit(
        x_train,
        truth_train,
        sample_weight=project_class_weights(model_training, candidate.project_weighting),
    )
    predicted = model.predict(x_validation)
    scores = _scores(model, x_validation)
    threshold = np.nan
    diagnostics: dict[str, Any] = {}
    if task == "s2":
        if calibration is None:
            raise RuntimeError("S2 calibration partition is missing")
        x_calibration = vectorizer.transform(_text(calibration)).tocsr().astype(np.float32)
        calibration_scores = _scores(model, x_calibration)
        if calibration_scores.shape[1] == 1:
            calibration_raw = calibration_scores[:, 0]
            validation_raw = scores[:, 0]
        else:
            high_index = list(model.classes_).index("HIGH_IMPACT")
            calibration_raw = calibration_scores[:, high_index]
            validation_raw = scores[:, high_index]
        calibration_truth = map_labels(calibration.severity, task) == "HIGH_IMPACT"
        calibrator = fit_calibrator(calibration_raw, calibration_truth)
        calibration_probability = calibrator.predict_proba(calibration_raw.reshape(-1, 1))[:, 1]
        selection = select_s2_threshold(calibration_probability, calibration_truth)
        threshold = selection["threshold"]
        probability = calibrator.predict_proba(validation_raw.reshape(-1, 1))[:, 1]
        predicted = np.where(probability >= threshold, "HIGH_IMPACT", "LOWER_IMPACT")
    metrics = fixed_metrics(truth, predicted, TASK_LABELS[task])
    metrics.update(
        {
            **asdict(candidate),
            "task": task,
            "status": "trained",
            "project_macro_f1": _project_macro(validation, truth, predicted, TASK_LABELS[task]),
            "feature_count": x_train.shape[1],
            "runtime_seconds": perf_counter() - started,
            "threshold": threshold,
            **diagnostics,
        }
    )
    predictions = pd.DataFrame(
        {
            "row_key": validation.row_key.to_numpy(),
            "project": validation.source_project.to_numpy(),
            "truth": truth,
            "predicted": predicted,
        }
    )
    return metrics, predictions


def screening_sample(development: pd.DataFrame, rows_per_project: int = 4000) -> pd.DataFrame:
    return (
        pd.concat(
            [
                development.loc[development.source_project.eq(project)]
                .sort_values(["creation_time", "row_key"], kind="mergesort")
                .head(rows_per_project)
                for project in PROJECTS
            ],
            ignore_index=True,
        )
        .sort_values(["creation_time", "row_key"], kind="mergesort")
        .reset_index(drop=True)
    )


def _single_screen_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts_train, parts_validation = [], []
    for project in PROJECTS:
        part = frame.loc[frame.source_project.eq(project)].sort_values(
            ["creation_time", "row_key"], kind="mergesort"
        )
        cut = max(1, int(np.floor(len(part) * 0.75)))
        parts_train.append(part.iloc[:cut])
        parts_validation.append(part.iloc[cut:])
    return pd.concat(parts_train), pd.concat(parts_validation)


def _safe_candidate(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    task: str,
    candidate: Candidate,
    max_features: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    try:
        return evaluate_candidate(training, validation, task, candidate, max_features)
    except S2ThresholdInfeasible as error:
        return (
            {
                **asdict(candidate),
                "task": task,
                "status": "infeasible",
                "failure_type": type(error).__name__,
                **error.diagnostics,
            },
            None,
        )
    except (ValueError, FloatingPointError, MemoryError) as error:
        return (
            {
                **asdict(candidate),
                "task": task,
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            },
            None,
        )


def paired_bootstrap(
    predictions: pd.DataFrame, labels: list[str], n_resamples: int = 2000
) -> dict[str, float]:
    required = {"truth", "baseline", "challenger", "project"}
    if required - set(predictions.columns):
        raise ValueError("Paired bootstrap columns are incomplete")
    rng = np.random.default_rng(SEED)
    row_deltas = []
    truth = predictions.truth.to_numpy()
    baseline = predictions.baseline.to_numpy()
    challenger = predictions.challenger.to_numpy()
    for _ in range(n_resamples):
        index = rng.integers(0, len(predictions), len(predictions))
        row_deltas.append(
            f1_score(
                truth[index], challenger[index], labels=labels, average="macro", zero_division=0
            )
            - f1_score(
                truth[index], baseline[index], labels=labels, average="macro", zero_division=0
            )
        )
    project_deltas = []
    for _ in range(n_resamples):
        sampled = rng.choice(PROJECTS, size=len(PROJECTS), replace=True)
        pieces = [predictions.loc[predictions.project.eq(project)] for project in sampled]
        frame = pd.concat(pieces, ignore_index=True)
        project_deltas.append(
            f1_score(
                frame.truth,
                frame.challenger,
                labels=labels,
                average="macro",
                zero_division=0,
            )
            - f1_score(
                frame.truth,
                frame.baseline,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    return {
        "row_mean_delta": float(np.mean(row_deltas)),
        "row_ci_lower": float(np.percentile(row_deltas, 2.5)),
        "row_ci_upper": float(np.percentile(row_deltas, 97.5)),
        "project_mean_delta": float(np.mean(project_deltas)),
        "project_ci_lower": float(np.percentile(project_deltas, 2.5)),
        "project_ci_upper": float(np.percentile(project_deltas, 97.5)),
    }


def _manifest(
    mode: str,
    protocol: Path,
    protocol_commit: str,
    partitions: PartitionedData,
    command: str,
) -> dict[str, Any]:
    fold_payload = [
        {"fold": fold.fold, "train": fold.train, "validation": fold.validation}
        for fold in freeze_development_folds(partitions.development)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "protocol_sha256": sha256_file(protocol),
        "protocol_commit": protocol_commit,
        "input_hashes": INPUT_HASHES,
        "candidate_checksum": candidate_checksum(),
        "seed": SEED,
        "projects": list(PROJECTS),
        "task_mappings": TASK_MAPPINGS,
        "development_rows": len(partitions.development),
        "locked_test_rows": len(partitions.locked_test),
        "purged_test_rows": partitions.purged_test_rows,
        "development_fingerprint": hashlib.sha256(
            "\n".join(partitions.development.row_key).encode()
        ).hexdigest(),
        "locked_test_fingerprint": hashlib.sha256(
            "\n".join(partitions.locked_test.row_key).encode()
        ).hexdigest(),
        "fold_fingerprint": hashlib.sha256(
            json.dumps(fold_payload, sort_keys=True).encode()
        ).hexdigest(),
        "held_out_test_accessed": False,
        "models_fitted": False,
        "completed_stages": [],
        "fit_budget_per_task": 60,
        "fit_budget_total": 180,
        "packages": package_versions(
            ["numpy", "pandas", "pyarrow", "psutil", "scikit-learn", "scipy"]
        ),
        "canonical_command": command,
    }


def _prepare(
    processed_root: Path,
    mylyn_development: Path,
    audit_output: Path,
    protocol: Path,
    protocol_commit: str,
    mode: str,
    command: str,
    resume: bool = False,
) -> tuple[PartitionedData, dict[str, Any]]:
    if len(protocol_commit) != 40 or any(
        value not in "0123456789abcdef" for value in protocol_commit
    ):
        raise ValueError("--protocol-commit must be a full lowercase Git SHA")
    universe = load_allowed_universe(processed_root, mylyn_development)
    partitions = partition_allowed_universe(universe)
    manifest = _manifest(mode, protocol, protocol_commit, partitions, command)
    manifest_path = audit_output / "study_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Output exists; use a new directory or --resume")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume_manifest(existing, manifest)
        manifest["models_fitted"] = existing.get("models_fitted", False)
        manifest["completed_stages"] = existing.get("completed_stages", [])
    audit_output.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, manifest)
    atomic_csv(audit_output / "partition_audit.csv", partitions.audit)
    manifest_rows = {
        row["source_project"]: row
        for row in json.loads((processed_root / "manifest.json").read_text(encoding="utf-8"))
    }
    dataset_rows = []
    for project in PROJECTS:
        allowed = universe.loc[universe.source_project.eq(project)]
        available_entry = manifest_rows[project]
        dataset_rows.append(
            {
                "project": project,
                "available_rows": int(available_entry["output_row_count"]),
                "eligible_available_rows": int(
                    sum(
                        available_entry["severity_distribution"].get(label, 0)
                        for label in TASK_LABELS["s6"]
                    )
                ),
                "allowed_legacy_development_rows": len(allowed),
                "pooled_development_rows": int(
                    partitions.development.source_project.eq(project).sum()
                ),
                "pooled_locked_test_rows": int(
                    partitions.locked_test.source_project.eq(project).sum()
                ),
                "minimum_creation_time": allowed.creation_time.min(),
                "maximum_creation_time": allowed.creation_time.max(),
                "missing_summary": int(allowed.summary.str.strip().eq("").sum()),
                "missing_description": int(allowed.description.str.strip().eq("").sum()),
                "exact_duplicate_rows": int(allowed.exact_text_hash.duplicated().sum()),
                "known_linked_group_rows": int(
                    allowed.duplicate_group_id.fillna("").astype(str).duplicated(keep=False).sum()
                ),
                "reservation": (
                    "approved MYLYN development only"
                    if project == "MYLYN"
                    else "latest 20% eligible chronological rows excluded"
                ),
            }
        )
    atomic_csv(audit_output / "dataset_audit.csv", pd.DataFrame(dataset_rows))
    distributions = []
    for task in TASK_LABELS:
        mapped = map_labels(partitions.development.severity, task)
        for (project, label), count in (
            pd.DataFrame({"project": partitions.development.source_project, "label": mapped})
            .value_counts()
            .items()
        ):
            distributions.append(
                {
                    "partition": "development",
                    "task": task,
                    "project": project,
                    "label": label,
                    "rows": count,
                }
            )
    atomic_csv(audit_output / "development_class_distributions.csv", pd.DataFrame(distributions))
    atomic_csv(
        audit_output / "candidate_grid.csv", pd.DataFrame([asdict(row) for row in candidate_grid()])
    )
    return partitions, manifest


def plan(
    processed_root: Path,
    mylyn_development: Path,
    audit_output: Path,
    protocol: Path,
    protocol_commit: str,
    command: str,
) -> dict[str, Any]:
    partitions, manifest = _prepare(
        processed_root, mylyn_development, audit_output, protocol, protocol_commit, "plan", command
    )
    validation = {
        "status": "PLAN_VALIDATED",
        "projects": len(PROJECTS),
        "development_rows": len(partitions.development),
        "locked_test_rows": len(partitions.locked_test),
        "duplicate_overlap": 0,
        "planned_fits": 180,
        "held_out_test_accessed": False,
    }
    atomic_json(audit_output / "plan_validation.json", validation)
    return {**validation, "manifest": manifest}


def dry_run(
    processed_root: Path,
    mylyn_development: Path,
    audit_output: Path,
    model_output: Path,
    protocol: Path,
    protocol_commit: str,
    command: str,
    rows_per_project: int = 300,
) -> dict[str, Any]:
    partitions, manifest = _prepare(
        processed_root,
        mylyn_development,
        audit_output,
        protocol,
        protocol_commit,
        "dry-run",
        command,
    )
    sample = screening_sample(partitions.development, rows_per_project)
    training, validation = _single_screen_split(sample)
    candidates = [candidate_grid()[0], candidate_grid()[12], candidate_grid()[24]]
    rows = []
    rss_before = psutil.Process().memory_info().rss
    started = perf_counter()
    for task, candidate in zip(TASK_LABELS, candidates, strict=True):
        row, _ = _safe_candidate(training, validation, task, candidate, max_features=5000)
        rows.append(row)
    elapsed = perf_counter() - started
    rss_delta = max(0, psutil.Process().memory_info().rss - rss_before)
    trained = sum(row["status"] == "trained" for row in rows)
    seconds_per_fit = elapsed / max(trained, 1)
    resource = {
        "pilot_rows": len(sample),
        "pilot_fits": len(rows),
        "pilot_seconds": elapsed,
        "seconds_per_fit": seconds_per_fit,
        "estimated_full_runtime_seconds": seconds_per_fit
        * 180
        * max(len(partitions.development) / max(len(sample), 1), 1) ** 0.65,
        "pilot_rss_delta_bytes": rss_delta,
        "estimated_peak_ram_bytes": max(rss_delta * 4, 8_000_000_000),
        "estimated_disk_bytes": 250_000_000,
    }
    model_output.mkdir(parents=True, exist_ok=True)
    atomic_csv(model_output / "pilot_results.csv", pd.DataFrame(rows))
    atomic_json(audit_output / "resource_estimate.json", resource)
    manifest.update({"models_fitted": trained > 0, "completed_stages": ["TIMING_PILOT"]})
    atomic_json(audit_output / "study_manifest.json", manifest)
    result = {
        "status": "DRY_RUN_COMPLETE",
        "models_fitted": trained,
        "held_out_test_accessed": False,
        **resource,
    }
    atomic_json(audit_output / "dry_run_validation.json", result)
    return result


def real_run(
    processed_root: Path,
    mylyn_development: Path,
    audit_output: Path,
    model_output: Path,
    protocol: Path,
    protocol_commit: str,
    command: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Run development selection only. Locked evaluation is intentionally not implemented here."""
    state_path = audit_output / "engine_state.json"
    partitions, manifest = _prepare(
        processed_root,
        mylyn_development,
        audit_output,
        protocol,
        protocol_commit,
        "real-run",
        command,
        resume,
    )
    state = {"completed": [], "rows": [], "predictions": []}
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    elif state_path.exists():
        raise FileExistsError("Existing engine state; use --resume")
    completed = set(state["completed"])
    result_rows = list(state["rows"])
    prediction_rows = list(state.get("predictions", []))

    def save_state() -> None:
        atomic_json(
            state_path,
            {
                "completed": sorted(completed),
                "rows": result_rows,
                "predictions": prediction_rows,
            },
        )

    selected_rows = []
    bootstrap_rows = []
    project_rows = []
    sample = screening_sample(partitions.development)
    screen_train, screen_validation = _single_screen_split(sample)
    folds = freeze_development_folds(partitions.development)
    ordered = partitions.development.sort_values(
        ["creation_time", "row_key"], kind="mergesort"
    ).reset_index(drop=True)
    for task in TASK_LABELS:
        for fold in folds:
            training = ordered.iloc[list(fold.train)]
            validation = ordered.iloc[list(fold.validation)]
            dummy_key = f"{task}:S0:DUMMY:F{fold.fold}"
            if dummy_key not in completed:
                truth_train = map_labels(training.severity, task)
                truth = map_labels(validation.severity, task)
                dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
                dummy.fit(np.zeros((len(training), 1)), truth_train)
                predicted = dummy.predict(np.zeros((len(validation), 1)))
                row = {
                    **fixed_metrics(truth, predicted, TASK_LABELS[task]),
                    "task": task,
                    "stage": "S0",
                    "candidate_id": "DUMMY_MOST_FREQUENT",
                    "fold": fold.fold,
                    "status": "trained",
                    "checkpoint_key": dummy_key,
                }
                result_rows.append(row)
                prediction_rows.extend(
                    pd.DataFrame(
                        {
                            "row_key": validation.row_key.to_numpy(),
                            "project": validation.source_project.to_numpy(),
                            "truth": truth,
                            "predicted": predicted,
                            "task": task,
                            "fold": fold.fold,
                            "role": "baseline",
                            "candidate_id": "DUMMY_MOST_FREQUENT",
                        }
                    ).to_dict("records")
                )
                completed.add(dummy_key)
                save_state()
                atomic_json(audit_output / f"checkpoint_{task}_S0_dummy_fold_{fold.fold}.json", row)
            baseline_key = f"{task}:S0:POOLED_LINEAR:F{fold.fold}"
            if baseline_key not in completed:
                baseline = candidate_grid()[4]
                row, predictions = _safe_candidate(training, validation, task, baseline)
                row.update(
                    {
                        "stage": "S0",
                        "candidate_id": "POOLED_LINEAR_BASELINE",
                        "fold": fold.fold,
                        "checkpoint_key": baseline_key,
                    }
                )
                result_rows.append(row)
                if predictions is not None:
                    prediction_rows.extend(
                        predictions.assign(
                            task=task,
                            fold=fold.fold,
                            role="baseline",
                            candidate_id="POOLED_LINEAR_BASELINE",
                        ).to_dict("records")
                    )
                completed.add(baseline_key)
                save_state()
                atomic_json(
                    audit_output / f"checkpoint_{task}_S0_pooled_linear_fold_{fold.fold}.json",
                    row,
                )
        stage1 = []
        for candidate in candidate_grid():
            key = f"{task}:S1:{candidate.candidate_id}"
            existing = next((row for row in result_rows if row.get("checkpoint_key") == key), None)
            if key in completed and existing is not None:
                stage1.append(existing)
                continue
            row, _ = _safe_candidate(screen_train, screen_validation, task, candidate)
            row.update({"stage": "S1", "checkpoint_key": key})
            result_rows.append(row)
            completed.add(key)
            save_state()
            atomic_json(audit_output / f"checkpoint_{task}_S1_{candidate.candidate_id}.json", row)
            stage1.append(row)
        top = rank_candidates(stage1)
        for candidate_row in top:
            candidate = Candidate(
                candidate_row["candidate_id"],
                candidate_row["representation"],
                candidate_row["model"],
                float(candidate_row["c"]),
                candidate_row["class_weight"],
                candidate_row["project_weighting"],
            )
            for fold in folds:
                key = f"{task}:S2:{candidate.candidate_id}:F{fold.fold}"
                if key in completed:
                    continue
                row, predictions = _safe_candidate(
                    ordered.iloc[list(fold.train)],
                    ordered.iloc[list(fold.validation)],
                    task,
                    candidate,
                )
                row.update({"stage": "S2", "fold": fold.fold, "checkpoint_key": key})
                result_rows.append(row)
                if predictions is not None:
                    prediction_rows.extend(
                        predictions.assign(
                            task=task,
                            fold=fold.fold,
                            role="challenger",
                            candidate_id=candidate.candidate_id,
                        ).to_dict("records")
                    )
                completed.add(key)
                save_state()
                atomic_json(
                    audit_output
                    / f"checkpoint_{task}_S2_{candidate.candidate_id}_fold_{fold.fold}.json",
                    row,
                )
        stage2_rows = [
            row
            for row in result_rows
            if row.get("task") == task
            and row.get("stage") == "S2"
            and row.get("status") == "trained"
        ]
        aggregates = []
        for candidate_id, rows in pd.DataFrame(stage2_rows).groupby("candidate_id"):
            if len(rows) != len(folds):
                continue
            first = rows.iloc[0]
            aggregates.append(
                {
                    "candidate_id": candidate_id,
                    "task": task,
                    "status": "trained",
                    "representation": first.representation,
                    "model": first.model,
                    "c": first.c,
                    "class_weight": first.class_weight,
                    "project_weighting": first.project_weighting,
                    "macro_f1": float(rows.macro_f1.mean()),
                    "project_macro_f1": float(rows.project_macro_f1.mean()),
                    "minimum_class_recall": float(rows.minimum_class_recall.mean()),
                    "balanced_accuracy": float(rows.balanced_accuracy.mean()),
                    "fold_macro_f1_sd": float(rows.macro_f1.std(ddof=0)),
                    "feature_count": int(rows.feature_count.max()),
                    "runtime_seconds": float(rows.runtime_seconds.sum()),
                }
            )
        selected = rank_candidates(aggregates, keep=1)[0]
        selected_rows.append(selected)
        stage3_key = f"{task}:S3:{selected['candidate_id']}"
        completed.add(stage3_key)
        atomic_json(audit_output / f"checkpoint_{task}_S3_selection.json", selected)
        predictions = pd.DataFrame(prediction_rows)
        baseline_candidates = pd.DataFrame(
            [
                row
                for row in result_rows
                if row.get("task") == task
                and row.get("stage") == "S0"
                and row.get("status") == "trained"
            ]
        )
        baseline_id = baseline_candidates.groupby("candidate_id").macro_f1.mean().idxmax()
        baseline = predictions.loc[
            predictions.task.eq(task)
            & predictions.role.eq("baseline")
            & predictions.candidate_id.eq(baseline_id),
            ["row_key", "fold", "project", "truth", "predicted"],
        ].rename(columns={"predicted": "baseline"})
        challenger = predictions.loc[
            predictions.task.eq(task)
            & predictions.role.eq("challenger")
            & predictions.candidate_id.eq(selected["candidate_id"]),
            ["row_key", "fold", "project", "truth", "predicted"],
        ].rename(columns={"predicted": "challenger"})
        paired = baseline.merge(
            challenger,
            on=["row_key", "fold", "project", "truth"],
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(baseline) or len(paired) != len(challenger):
            raise RuntimeError(f"{task}: selected/baseline OOF predictions are not aligned")
        bootstrap_rows.append({"task": task, **paired_bootstrap(paired, TASK_LABELS[task])})
        for project in PROJECTS:
            part = paired.loc[paired.project.eq(project)]
            metrics = fixed_metrics(
                part.truth.to_numpy(), part.challenger.to_numpy(), TASK_LABELS[task]
            )
            project_rows.append(
                {
                    "task": task,
                    "project": project,
                    "rows": len(part),
                    **{
                        key: metrics[key]
                        for key in ("macro_f1", "accuracy", "weighted_f1", "balanced_accuracy")
                    },
                }
            )
        save_state()
    atomic_csv(model_output / "candidate_results.csv", pd.DataFrame(result_rows))
    atomic_csv(model_output / "selected_configurations.csv", pd.DataFrame(selected_rows))
    atomic_csv(model_output / "bootstrap_results.csv", pd.DataFrame(bootstrap_rows))
    atomic_csv(model_output / "per_project_results.csv", pd.DataFrame(project_rows))
    manifest.update(
        {
            "models_fitted": True,
            "completed_stages": ["S0", "S1", "S2", "S3"],
            "held_out_test_accessed": False,
        }
    )
    atomic_json(audit_output / "study_manifest.json", manifest)
    return {
        "status": "DEVELOPMENT_SELECTION_COMPLETE",
        "held_out_test_accessed": False,
        "locked_evaluation_authorized": False,
    }


def validate(
    audit_output: Path, model_output: Path, protocol: Path, protocol_commit: str
) -> dict[str, Any]:
    manifest = json.loads((audit_output / "study_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "protocol_hash": manifest["protocol_sha256"] == sha256_file(protocol),
        "protocol_commit": manifest["protocol_commit"] == protocol_commit,
        "test_not_accessed": manifest["held_out_test_accessed"] is False,
        "projects_9": manifest["projects"] == list(PROJECTS),
        "candidate_checksum": manifest["candidate_checksum"] == candidate_checksum(),
        "partition_audit": (audit_output / "partition_audit.csv").is_file(),
        "privacy_safe_outputs": not any(
            forbidden in path.name.lower()
            for path in model_output.glob("*")
            for forbidden in ("prediction", "identifier", "locked_test")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Validation failed: {checks}")
    result = {"status": "VALIDATION_PASS", "checks": checks}
    atomic_json(audit_output / "validation.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "dry-run", "real-run", "validate"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/eclipse_core"))
    parser.add_argument("--mylyn-development", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rows-per-project", type=int, default=300)
    return parser


def main() -> None:
    args = _parser().parse_args()
    canonical = "python -m defect_classifier.pooled_study_v1 " + " ".join(
        token for token in sys.argv[1:] if token != "--resume"
    )
    if args.mode == "validate":
        if args.model_output is None:
            raise ValueError("--model-output is required")
        result = validate(args.audit_output, args.model_output, args.protocol, args.protocol_commit)
    else:
        if args.mylyn_development is None:
            raise ValueError("--mylyn-development is required")
        common = (
            args.processed_root,
            args.mylyn_development,
            args.audit_output,
        )
        if args.mode == "plan":
            result = plan(*common, args.protocol, args.protocol_commit, canonical)
        elif args.mode == "dry-run":
            if args.model_output is None:
                raise ValueError("--model-output is required")
            result = dry_run(
                *common,
                args.model_output,
                args.protocol,
                args.protocol_commit,
                canonical,
                args.rows_per_project,
            )
        else:
            if args.model_output is None:
                raise ValueError("--model-output is required")
            result = real_run(
                *common,
                args.model_output,
                args.protocol,
                args.protocol_commit,
                canonical,
                args.resume,
            )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
