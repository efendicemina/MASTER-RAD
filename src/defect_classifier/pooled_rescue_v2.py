"""Bounded, development-only rescue pilot for pooled Eclipse severity v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from .pooled_study_v1 import (
    INPUT_HASHES,
    PROJECTS,
    SEED,
    TASK_LABELS,
    _row_key,
    atomic_csv,
    atomic_json,
    fixed_metrics,
    freeze_development_folds,
    map_labels,
    partition_allowed_universe,
)
from .transfer_study_v2 import (
    S2ThresholdInfeasible,
    exact_identity,
    fit_calibrator,
    select_s2_threshold,
    software_clean,
    software_tokenize,
)
from .utils import package_versions, sha256_file

V1_WINNERS = {"s6": "P-002", "s3": "P-011", "s2": "P-002"}
SCHEMA_VERSION = "2.0-exploratory"
SCREEN_FOLD = 2
STAGE_B_KEEP = 2
RESUME_LOCKS = (
    "schema_version",
    "v1_results_commit",
    "v1_manifest_sha256",
    "candidate_checksum",
    "development_fingerprint",
    "fold_fingerprint",
    "seed",
    "canonical_command",
)


@dataclass(frozen=True, slots=True)
class RescueCandidate:
    candidate_id: str
    representation: str
    alpha: float
    summary_weight: float = 2.0
    project_weighting: bool = False
    duplicate_weighting: bool = False
    metadata: bool = False
    decision: str = "flat"
    nbsvm: bool = False


def rescue_candidates() -> tuple[RescueCandidate, ...]:
    """Eight orthogonal candidates, not an expandable hyperparameter grid."""
    return (
        RescueCandidate("R-01", "field_word", 0.50),
        RescueCandidate("R-02", "field_word", 0.75),
        RescueCandidate("R-03", "field_word_char", 0.50),
        RescueCandidate("R-04", "field_word", 0.50, duplicate_weighting=True),
        RescueCandidate("R-05", "field_word", 0.50, project_weighting=True, metadata=True),
        RescueCandidate("R-06", "field_word", 0.50, duplicate_weighting=True, nbsvm=True),
        RescueCandidate("R-07", "field_word", 0.50, decision="hierarchical"),
        RescueCandidate(
            "R-08",
            "field_word",
            0.75,
            project_weighting=True,
            duplicate_weighting=True,
            metadata=True,
        ),
    )


def candidate_checksum() -> str:
    payload = json.dumps([asdict(row) for row in rescue_candidates()], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _clean(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).map(lambda value: software_clean(value, mask_labels=True))


class RareMetadataEncoder:
    """Deterministic report-time metadata encoder with an explicit unknown bucket."""

    fields = ("source_project", "product", "component")

    def __init__(self, minimum_count: int = 20) -> None:
        self.minimum_count = minimum_count
        self.categories_: dict[str, tuple[str, ...]] = {}
        self.offsets_: dict[str, int] = {}
        self.n_features_ = 0

    def fit(self, frame: pd.DataFrame) -> RareMetadataEncoder:
        offset = 0
        for field in self.fields:
            values = frame[field].fillna("__MISSING__").astype(str)
            counts = values.value_counts()
            kept = tuple(sorted(counts[counts >= self.minimum_count].index))
            self.categories_[field] = kept
            self.offsets_[field] = offset
            offset += len(kept) + 1
        self.n_features_ = offset
        return self

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        rows: list[int] = []
        columns: list[int] = []
        for field in self.fields:
            categories = {value: index for index, value in enumerate(self.categories_[field])}
            unknown = len(categories)
            offset = self.offsets_[field]
            values = frame[field].fillna("__MISSING__").astype(str)
            for row, value in enumerate(values):
                rows.append(row)
                columns.append(offset + categories.get(value, unknown))
        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix(
            (data, (rows, columns)), shape=(len(frame), self.n_features_), dtype=np.float32
        )


class FieldFeatureBuilder:
    """Separately fit Summary/Description blocks, then globally L2-normalize."""

    def __init__(self, candidate: RescueCandidate, max_features: int = 50_000) -> None:
        self.candidate = candidate
        branch = max_features // (4 if candidate.representation == "field_word_char" else 2)
        kwargs = dict(
            tokenizer=software_tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.98,
            max_features=branch,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.summary_word = TfidfVectorizer(**kwargs)
        self.description_word = TfidfVectorizer(**kwargs)
        self.summary_char = None
        self.description_char = None
        if candidate.representation == "field_word_char":
            char_kwargs = dict(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=3,
                max_df=0.98,
                max_features=branch,
                sublinear_tf=True,
                dtype=np.float32,
            )
            self.summary_char = TfidfVectorizer(**char_kwargs)
            self.description_char = TfidfVectorizer(**char_kwargs)
        self.metadata = RareMetadataEncoder() if candidate.metadata else None

    def fit_transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        summary, description = _clean(frame.summary), _clean(frame.description)
        blocks = [
            self.summary_word.fit_transform(summary) * self.candidate.summary_weight,
            self.description_word.fit_transform(description),
        ]
        if self.summary_char is not None and self.description_char is not None:
            blocks.extend(
                [
                    self.summary_char.fit_transform(summary) * self.candidate.summary_weight,
                    self.description_char.fit_transform(description),
                ]
            )
        if self.metadata is not None:
            blocks.append(self.metadata.fit(frame).transform(frame))
        return normalize(sparse.hstack(blocks, format="csr"), norm="l2", copy=False)

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        summary, description = _clean(frame.summary), _clean(frame.description)
        blocks = [
            self.summary_word.transform(summary) * self.candidate.summary_weight,
            self.description_word.transform(description),
        ]
        if self.summary_char is not None and self.description_char is not None:
            blocks.extend(
                [
                    self.summary_char.transform(summary) * self.candidate.summary_weight,
                    self.description_char.transform(description),
                ]
            )
        if self.metadata is not None:
            blocks.append(self.metadata.transform(frame))
        return normalize(sparse.hstack(blocks, format="csr"), norm="l2", copy=False)


def smoothed_class_weights(labels: np.ndarray, alpha: float) -> np.ndarray:
    counts = Counter(labels)
    balanced = {label: len(labels) / (len(counts) * count) for label, count in counts.items()}
    values = np.asarray([balanced[label] ** alpha for label in labels], dtype=np.float64)
    return values / values.mean()


def composed_sample_weights(
    frame: pd.DataFrame, labels: np.ndarray, candidate: RescueCandidate
) -> np.ndarray:
    weights = smoothed_class_weights(labels, candidate.alpha)
    if candidate.project_weighting:
        counts = frame.source_project.value_counts()
        weights *= frame.source_project.map(lambda value: counts[value] ** -0.5).to_numpy()
    if candidate.duplicate_weighting:
        counts = frame.duplicate_group.value_counts()
        weights *= frame.duplicate_group.map(lambda value: counts[value] ** -0.5).to_numpy()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid composed sample weights")
    return weights / weights.mean()


class NBSVMClassifier(ClassifierMixin, BaseEstimator):
    """Small deterministic multiclass one-vs-rest NB-SVM implementation."""

    def __init__(self, c: float = 0.25) -> None:
        self.c = c

    def fit(
        self, matrix: sparse.csr_matrix, labels: np.ndarray, sample_weight: np.ndarray
    ) -> NBSVMClassifier:
        self.classes_ = np.unique(labels)
        self.models_: list[LinearSVC] = []
        self.ratios_: list[np.ndarray] = []
        binary = matrix.copy()
        binary.data[:] = 1.0
        for label in self.classes_:
            positive = labels == label
            p = (np.asarray(binary[positive].sum(axis=0)).ravel() + 1.0) / (positive.sum() + 1.0)
            q = (np.asarray(binary[~positive].sum(axis=0)).ravel() + 1.0) / (
                (~positive).sum() + 1.0
            )
            ratio = np.log(p / q).astype(np.float32)
            model = LinearSVC(C=self.c, random_state=SEED)
            model.fit(matrix.multiply(ratio), positive, sample_weight=sample_weight)
            self.ratios_.append(ratio)
            self.models_.append(model)
        return self

    def decision_function(self, matrix: sparse.csr_matrix) -> np.ndarray:
        return np.column_stack(
            [
                model.decision_function(matrix.multiply(ratio))
                for model, ratio in zip(self.models_, self.ratios_, strict=True)
            ]
        )

    def predict(self, matrix: sparse.csr_matrix) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(matrix), axis=1)]


class HierarchicalClassifier:
    def __init__(self, task: str) -> None:
        self.task = task

    def fit(self, matrix: sparse.csr_matrix, labels: np.ndarray, weights: np.ndarray) -> None:
        majority = "normal" if self.task == "s6" else "MEDIUM"
        gate_labels = np.where(labels == majority, majority, "__MINORITY__")
        self.gate = LinearSVC(C=0.25, random_state=SEED).fit(
            matrix, gate_labels, sample_weight=weights
        )
        minority = labels != majority
        self.minority = LinearSVC(C=0.25, random_state=SEED).fit(
            matrix[minority], labels[minority], sample_weight=weights[minority]
        )
        self.majority = majority

    def predict(self, matrix: sparse.csr_matrix) -> np.ndarray:
        gate = self.gate.predict(matrix)
        result = np.full(len(gate), self.majority, dtype=object)
        minority = gate == "__MINORITY__"
        if minority.any():
            result[minority] = self.minority.predict(matrix[minority])
        return result.astype(str)


def _split_inner(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    training, calibration = [], []
    for project in PROJECTS:
        part = frame.loc[frame.source_project.eq(project)].sort_values(
            ["creation_time", "row_key"], kind="mergesort"
        )
        cut = max(1, int(np.floor(len(part) * 0.8)))
        training.append(part.iloc[:cut])
        calibration.append(part.iloc[cut:])
    return pd.concat(training), pd.concat(calibration)


def evaluate_candidate(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    task: str,
    candidate: RescueCandidate,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    model_training = training
    calibration = None
    if task == "s2":
        model_training, calibration = _split_inner(training)
    features = FieldFeatureBuilder(candidate)
    x_train = features.fit_transform(model_training)
    x_validation = features.transform(validation)
    truth_train = map_labels(model_training.severity, task)
    truth = map_labels(validation.severity, task)
    weights = composed_sample_weights(model_training, truth_train, candidate)
    threshold = np.nan
    if candidate.nbsvm:
        model: Any = NBSVMClassifier().fit(x_train, truth_train, weights)
    elif candidate.decision == "hierarchical" and task in {"s6", "s3"}:
        model = HierarchicalClassifier(task)
        model.fit(x_train, truth_train, weights)
    else:
        model = LinearSVC(C=0.25, random_state=SEED).fit(
            x_train, truth_train, sample_weight=weights
        )
    predicted = model.predict(x_validation)
    if task == "s2":
        if calibration is None:
            raise RuntimeError("Missing inner calibration partition")
        x_calibration = features.transform(calibration)
        cal_scores = np.asarray(model.decision_function(x_calibration))
        val_scores = np.asarray(model.decision_function(x_validation))
        high_index = list(model.classes_).index("HIGH_IMPACT")
        cal_raw = cal_scores[:, high_index] if cal_scores.ndim == 2 else cal_scores
        val_raw = val_scores[:, high_index] if val_scores.ndim == 2 else val_scores
        calibrator = fit_calibrator(
            cal_raw, map_labels(calibration.severity, task) == "HIGH_IMPACT"
        )
        cal_probability = calibrator.predict_proba(cal_raw.reshape(-1, 1))[:, 1]
        selection = select_s2_threshold(
            cal_probability, map_labels(calibration.severity, task) == "HIGH_IMPACT"
        )
        threshold = selection["threshold"]
        val_probability = calibrator.predict_proba(val_raw.reshape(-1, 1))[:, 1]
        predicted = np.where(val_probability >= threshold, "HIGH_IMPACT", "LOWER_IMPACT")
    metrics = fixed_metrics(truth, predicted, TASK_LABELS[task])
    metrics.update(
        {
            **asdict(candidate),
            "task": task,
            "status": "trained",
            "project_macro_f1": float(
                np.mean(
                    [
                        f1_score(
                            truth[validation.source_project.to_numpy() == project],
                            predicted[validation.source_project.to_numpy() == project],
                            labels=TASK_LABELS[task],
                            average="macro",
                            zero_division=0,
                        )
                        for project in PROJECTS
                        if np.any(validation.source_project.to_numpy() == project)
                    ]
                )
            ),
            "feature_count": int(x_train.shape[1]),
            "runtime_seconds": perf_counter() - started,
            "peak_rss_bytes": int(psutil.Process().memory_info().rss),
            "threshold": threshold,
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


def load_partitions(processed_root: Path, mylyn_development: Path):
    """Reconstruct the frozen v1 partition and return development only to callers."""
    valid = set(TASK_LABELS["s6"])
    frames = []
    for project in PROJECTS:
        if project == "MYLYN":
            if sha256_file(mylyn_development) != INPUT_HASHES[project]:
                raise ValueError("MYLYN development SHA mismatch")
            frame = pd.read_csv(mylyn_development).rename(
                columns={"id": "issue_id", "duplicate_group": "duplicate_group_id"}
            )
            frame["source_project"] = project
            frame["global_report_key"] = "MYLYN:" + frame.issue_id.astype(str)
            frame["dupe_of"] = ""
        else:
            path = processed_root / f"{project}.parquet"
            if sha256_file(path) != INPUT_HASHES[project]:
                raise ValueError(f"{project} input SHA mismatch")
            columns = [
                "issue_id",
                "global_report_key",
                "source_project",
                "product",
                "component",
                "summary",
                "description",
                "severity",
                "creation_time",
                "dupe_of",
                "duplicate_group_id",
            ]
            frame = pq.read_table(path, columns=columns).to_pandas()
        for field in ("product", "component"):
            if field not in frame:
                frame[field] = "__UNAVAILABLE__"
        frame["severity"] = frame.severity.astype(str).str.strip().str.lower()
        frame["creation_time"] = pd.to_datetime(frame.creation_time, utc=True, errors="coerce")
        frame = frame.loc[frame.severity.isin(valid) & frame.creation_time.notna()].copy()
        frame = frame.sort_values(["creation_time", "global_report_key"], kind="mergesort")
        if project != "MYLYN":
            frame = frame.iloc[: int(np.floor(len(frame) * 0.8))].copy()
        frame["summary"] = frame.summary.fillna("").astype(str)
        frame["description"] = frame.description.fillna("").astype(str)
        frame["exact_text_hash"] = [
            exact_identity(a, b) for a, b in zip(frame.summary, frame.description, strict=True)
        ]
        frame["row_key"] = [_row_key(project, value) for value in frame.global_report_key]
        frames.append(frame)
    universe = pd.concat(frames, ignore_index=True)
    partitions = partition_allowed_universe(universe)
    development = partitions.development.copy()
    # Drop the locked frame immediately: no vocabulary, fit, calibration or metric can receive it.
    del partitions.locked_test
    return development, partitions.audit, partitions.purged_test_rows


def _read_v1_predictions(engine_state: Path) -> pd.DataFrame:
    selected = set(V1_WINNERS.items())
    rows = []
    in_predictions = False
    collecting = False
    buffer: list[str] = []
    with engine_state.open(encoding="utf-8") as stream:
        for line in stream:
            if not in_predictions:
                if line.startswith('  "predictions": ['):
                    in_predictions = True
                continue
            if line.startswith("  ],") and not collecting:
                break
            if line.startswith("    {") and not collecting:
                collecting, buffer = True, [line]
                continue
            if collecting:
                buffer.append(line)
                if line.rstrip() in ("    },", "    }"):
                    row = json.loads("".join(buffer).rstrip().rstrip(","))
                    collecting, buffer = False, []
                    if (
                        row["role"] == "challenger"
                        and (row["task"], row["candidate_id"]) in selected
                    ):
                        rows.append(row)
    return pd.DataFrame(rows).rename(columns={"predicted": "v1_predicted"})


def rank_stage_a(rows: list[dict[str, Any]], keep: int = STAGE_B_KEEP) -> list[dict[str, Any]]:
    trained = [row for row in rows if row.get("status") == "trained"]
    return sorted(
        trained,
        key=lambda row: (
            -row["macro_f1"],
            -row["project_macro_f1"],
            -row["minimum_class_recall"],
            -row["balanced_accuracy"],
            row["candidate_id"],
        ),
    )[:keep]


def _macro_f1_from_confusion(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    actual = matrix.sum(axis=1).astype(float)
    denominator = predicted + actual
    scores = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator != 0,
    )
    return float(scores.mean())


def fast_paired_bootstrap(
    predictions: pd.DataFrame, labels: list[str], n_resamples: int = 2000
) -> dict[str, float]:
    """Exact paired bootstrap using confusion counts instead of repeated DataFrames."""
    required = {"truth", "baseline", "challenger", "project"}
    if required - set(predictions):
        raise ValueError("Paired bootstrap columns are incomplete")
    label_index = {label: index for index, label in enumerate(labels)}
    truth = predictions.truth.map(label_index).to_numpy(dtype=np.int64)
    baseline = predictions.baseline.map(label_index).to_numpy(dtype=np.int64)
    challenger = predictions.challenger.map(label_index).to_numpy(dtype=np.int64)
    if np.any(pd.isna(truth)) or np.any(pd.isna(baseline)) or np.any(pd.isna(challenger)):
        raise ValueError("Unknown bootstrap label")
    size = len(labels)
    baseline_codes = truth * size + baseline
    challenger_codes = truth * size + challenger
    rng = np.random.default_rng(SEED)
    row_deltas = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        sample = rng.integers(0, len(predictions), len(predictions))
        baseline_matrix = np.bincount(baseline_codes[sample], minlength=size * size).reshape(
            size, size
        )
        challenger_matrix = np.bincount(challenger_codes[sample], minlength=size * size).reshape(
            size, size
        )
        row_deltas[iteration] = _macro_f1_from_confusion(
            challenger_matrix
        ) - _macro_f1_from_confusion(baseline_matrix)
    project_baseline = []
    project_challenger = []
    projects = predictions.project.to_numpy()
    for project in PROJECTS:
        mask = projects == project
        project_baseline.append(
            np.bincount(baseline_codes[mask], minlength=size * size).reshape(size, size)
        )
        project_challenger.append(
            np.bincount(challenger_codes[mask], minlength=size * size).reshape(size, size)
        )
    project_deltas = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        sample = rng.integers(0, len(PROJECTS), len(PROJECTS))
        baseline_matrix = np.sum([project_baseline[index] for index in sample], axis=0)
        challenger_matrix = np.sum([project_challenger[index] for index in sample], axis=0)
        project_deltas[iteration] = _macro_f1_from_confusion(
            challenger_matrix
        ) - _macro_f1_from_confusion(baseline_matrix)
    return {
        "row_mean_delta": float(row_deltas.mean()),
        "row_ci_lower": float(np.percentile(row_deltas, 2.5)),
        "row_ci_upper": float(np.percentile(row_deltas, 97.5)),
        "project_mean_delta": float(project_deltas.mean()),
        "project_ci_lower": float(np.percentile(project_deltas, 2.5)),
        "project_ci_upper": float(np.percentile(project_deltas, 97.5)),
    }


def _safe_evaluate(training, validation, task, candidate):
    try:
        return evaluate_candidate(training, validation, task, candidate)
    except S2ThresholdInfeasible as error:
        return {
            **asdict(candidate),
            "task": task,
            "status": "infeasible",
            **error.diagnostics,
        }, None
    except (ValueError, FloatingPointError, MemoryError) as error:
        return {
            **asdict(candidate),
            "task": task,
            "status": "failed",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
        }, None


def _atomic_checkpoint(path: Path, row: dict[str, Any], predictions: pd.DataFrame | None) -> None:
    payload = {
        "result": row,
        "predictions": None if predictions is None else predictions.to_dict("records"),
    }
    atomic_json(path, payload)


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], pd.DataFrame | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = None if payload["predictions"] is None else pd.DataFrame(payload["predictions"])
    return payload["result"], predictions


def run(
    processed_root: Path,
    mylyn_development: Path,
    v1_audit: Path,
    v1_model: Path,
    audit_output: Path,
    model_output: Path,
    v1_results_commit: str,
    canonical_command: str,
    resume: bool,
) -> dict[str, Any]:
    audit_output.mkdir(parents=True, exist_ok=True)
    model_output.mkdir(parents=True, exist_ok=True)
    development, partition_audit, purged = load_partitions(processed_root, mylyn_development)
    folds = freeze_development_folds(development)
    v1_manifest_path = v1_audit / "study_manifest.json"
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    if v1_manifest["held_out_test_accessed"] is not False:
        raise RuntimeError("v1 manifest does not prove locked test protection")
    development_fingerprint = hashlib.sha256("\n".join(development.row_key).encode()).hexdigest()
    if development_fingerprint != v1_manifest["development_fingerprint"]:
        raise RuntimeError("Rescue development partition differs from v1")
    development = development.sort_values(
        ["creation_time", "row_key"], kind="mergesort"
    ).reset_index(drop=True)
    fold_payload = [{"fold": f.fold, "train": f.train, "validation": f.validation} for f in folds]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "mode": "development-only-rescue",
        "exploratory": True,
        "v1_results_commit": v1_results_commit,
        "v1_manifest_sha256": sha256_file(v1_manifest_path),
        "candidate_checksum": candidate_checksum(),
        "development_fingerprint": development_fingerprint,
        "fold_fingerprint": hashlib.sha256(
            json.dumps(fold_payload, sort_keys=True).encode()
        ).hexdigest(),
        "seed": SEED,
        "canonical_command": canonical_command,
        "held_out_test_accessed": False,
        "locked_evaluation_authorized": False,
        "maximum_candidates_per_task": 8,
        "maximum_new_fit_budget": 36,
        "platform_os_available": False,
        "post_submission_features_used": False,
        "packages": package_versions(
            ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "psutil")
        ),
    }
    manifest_path = audit_output / "study_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Existing rescue run requires --resume")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in RESUME_LOCKS:
            if existing.get(field) != manifest.get(field):
                raise RuntimeError(f"Resume provenance mismatch: {field}")
    elif resume:
        raise FileNotFoundError("Cannot resume without a manifest")
    atomic_json(manifest_path, manifest)
    atomic_csv(audit_output / "partition_audit.csv", partition_audit)
    atomic_csv(
        audit_output / "candidate_definitions.csv", pd.DataFrame(map(asdict, rescue_candidates()))
    )
    v1_predictions = _read_v1_predictions(v1_audit / "engine_state.json")
    results: list[dict[str, Any]] = []
    all_predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    for task in ("s6", "s3", "s2"):
        fold = folds[SCREEN_FOLD - 1]
        train = development.iloc[list(fold.train)]
        validation = development.iloc[list(fold.validation)]
        stage_rows = []
        for candidate in rescue_candidates():
            path = audit_output / f"checkpoint_{task}_A_{candidate.candidate_id}.json"
            if path.exists():
                row, predictions = _load_checkpoint(path)
            else:
                row, predictions = _safe_evaluate(train, validation, task, candidate)
                row.update({"stage": "A", "fold": SCREEN_FOLD})
                _atomic_checkpoint(path, row, predictions)
            results.append(row)
            stage_rows.append(row)
            if predictions is not None:
                all_predictions[(task, candidate.candidate_id, SCREEN_FOLD)] = predictions
        leaders = rank_stage_a(stage_rows)
        for leader in leaders:
            candidate = next(
                c for c in rescue_candidates() if c.candidate_id == leader["candidate_id"]
            )
            for frozen in folds:
                key = (task, candidate.candidate_id, frozen.fold)
                if key in all_predictions:
                    continue
                path = (
                    audit_output
                    / f"checkpoint_{task}_B_{candidate.candidate_id}_F{frozen.fold}.json"
                )
                if path.exists():
                    row, predictions = _load_checkpoint(path)
                else:
                    row, predictions = _safe_evaluate(
                        development.iloc[list(frozen.train)],
                        development.iloc[list(frozen.validation)],
                        task,
                        candidate,
                    )
                    row.update({"stage": "B", "fold": frozen.fold})
                    _atomic_checkpoint(path, row, predictions)
                results.append(row)
                if predictions is not None:
                    all_predictions[key] = predictions
    summaries = []
    projects = []
    classes = []
    bootstraps = []
    for task in ("s6", "s3", "s2"):
        stage_a = [r for r in results if r["task"] == task and r["stage"] == "A"]
        leader_ids = [r["candidate_id"] for r in rank_stage_a(stage_a)]
        for candidate_id in leader_ids:
            frames = []
            for fold in folds:
                challenger = all_predictions[(task, candidate_id, fold.fold)].copy()
                reference = v1_predictions[
                    (v1_predictions.task == task) & (v1_predictions.fold == fold.fold)
                ][["row_key", "v1_predicted"]]
                paired = challenger.merge(reference, on="row_key", validate="one_to_one")
                if len(paired) != len(challenger):
                    raise RuntimeError("v1/challenger rows are not identical")
                frames.append(paired)
            paired = pd.concat(frames, ignore_index=True)
            metrics = fixed_metrics(
                paired.truth.to_numpy(), paired.predicted.to_numpy(), TASK_LABELS[task]
            )
            v1_metrics = fixed_metrics(
                paired.truth.to_numpy(), paired.v1_predicted.to_numpy(), TASK_LABELS[task]
            )
            project_scores = []
            wins = 0
            for project in PROJECTS:
                part = paired[paired.project == project]
                new = f1_score(
                    part.truth,
                    part.predicted,
                    labels=TASK_LABELS[task],
                    average="macro",
                    zero_division=0,
                )
                old = f1_score(
                    part.truth,
                    part.v1_predicted,
                    labels=TASK_LABELS[task],
                    average="macro",
                    zero_division=0,
                )
                wins += new > old
                project_scores.append(new)
                projects.append(
                    {
                        "task": task,
                        "candidate_id": candidate_id,
                        "project": project,
                        "v1_macro_f1": old,
                        "rescue_macro_f1": new,
                        "delta": new - old,
                    }
                )
            fold_rows = [
                r
                for r in results
                if r["task"] == task
                and r["candidate_id"] == candidate_id
                and r.get("fold") in (1, 2, 3)
            ]
            unique = {int(r["fold"]): r for r in fold_rows}
            fold_wins = sum(
                unique[f]["macro_f1"]
                > f1_score(
                    v1_predictions[
                        (v1_predictions.task == task) & (v1_predictions.fold == f)
                    ].truth,
                    v1_predictions[
                        (v1_predictions.task == task) & (v1_predictions.fold == f)
                    ].v1_predicted,
                    labels=TASK_LABELS[task],
                    average="macro",
                    zero_division=0,
                )
                for f in (1, 2, 3)
            )
            summary = {
                "task": task,
                "candidate_id": candidate_id,
                "v1_macro_f1": v1_metrics["macro_f1"],
                **{
                    k: metrics[k]
                    for k in (
                        "macro_f1",
                        "accuracy",
                        "weighted_f1",
                        "balanced_accuracy",
                        "minimum_class_recall",
                    )
                },
                "delta": metrics["macro_f1"] - v1_metrics["macro_f1"],
                "project_macro_f1": float(np.mean(project_scores)),
                "fold_wins": fold_wins,
                "project_wins": wins,
                "runtime_seconds": sum(r["runtime_seconds"] for r in unique.values()),
                "peak_rss_bytes": max(r["peak_rss_bytes"] for r in unique.values()),
                "thresholds": json.dumps([unique[f].get("threshold") for f in (1, 2, 3)]),
            }
            summaries.append(summary)
            for row in metrics["per_class"]:
                classes.append({"task": task, "candidate_id": candidate_id, **row})
            bootstrap_input = paired.rename(
                columns={"v1_predicted": "baseline", "predicted": "challenger"}
            )
            bootstraps.append(
                {
                    "task": task,
                    "candidate_id": candidate_id,
                    **fast_paired_bootstrap(bootstrap_input, TASK_LABELS[task]),
                }
            )
    atomic_csv(
        model_output / "candidate_results.csv",
        pd.DataFrame(results).drop(columns=["confusion_matrix", "per_class"], errors="ignore"),
    )
    atomic_csv(model_output / "stage_b_summary.csv", pd.DataFrame(summaries))
    atomic_csv(model_output / "per_project_results.csv", pd.DataFrame(projects))
    atomic_csv(model_output / "per_class_results.csv", pd.DataFrame(classes))
    atomic_csv(model_output / "bootstrap_results.csv", pd.DataFrame(bootstraps))
    manifest["status"] = "RESCUE_DEVELOPMENT_COMPLETE"
    manifest["models_fitted"] = True
    atomic_json(manifest_path, manifest)
    return {
        "status": "RESCUE_DEVELOPMENT_COMPLETE",
        "held_out_test_accessed": False,
        "locked_evaluation_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--mylyn-development", type=Path, required=True)
    parser.add_argument("--v1-audit", type=Path, required=True)
    parser.add_argument("--v1-model", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--v1-results-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    canonical = "python -m defect_classifier.pooled_rescue_v2 " + " ".join(
        x for x in os.sys.argv[1:] if x != "--resume"
    )
    result = run(
        args.processed_root,
        args.mylyn_development,
        args.v1_audit,
        args.v1_model,
        args.audit_output,
        args.model_output,
        args.v1_results_commit,
        canonical,
        args.resume,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
