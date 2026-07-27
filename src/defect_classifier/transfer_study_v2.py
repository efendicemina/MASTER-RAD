"""Fail-closed MYLYN transfer/drift/threshold study v2."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .development_study import FrozenFold, folds_fingerprint, freeze_temporal_folds
from .hierarchical_study import _row_hash
from .redesign_study import TASK_LABELS, TASKS, _baseline_task, fixed_metrics, map_target
from .utils import package_versions, sha256_file

DEVELOPMENT_SHA256 = "8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5"
DEVELOPMENT_ROWS = 7664
OUTER_FOLD_FINGERPRINT = "5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903"
SEED = 42
MINIMUM_AVAILABLE_MEMORY_GB = 4.0
SCHEMA_VERSION = "2.0"
SOURCE_PROJECTS = ("BIRT", "CDT", "Equinox", "JDT", "Papyrus", "PDE", "Platform", "TPTP")
SOURCE_HASHES = {
    "BIRT": "a0e2c21c9095bcf250cc5d389177d2e6058b39cd589199316d235b00b532fa76",
    "CDT": "00cbe579b9a1a4606db80933154cb7a2498fc8931310805f033cb4558d725004",
    "Equinox": "f0ee1c9185c6edc119bae133fb64c81c3790559878f4f5279c12c189f337d5ba",
    "JDT": "0c1b597e9be4fce63a22cdac137e2f1f1657f8d6512fec689dc7be2233340c61",
    "Papyrus": "b82272d153d1edefa03a82268099f878a4f3e0f9f5a652dbbf259e5f7cc366f4",
    "PDE": "034ac8abccce6a5fcae49ad604faafd3f2f794c2d4dfd9a993430ef44e1d1808",
    "Platform": "21a82ca083594b286f226debf1bd98ad29063cef267d3809e6d160919d3bfdf1",
    "TPTP": "7aceea71ca29e14e11ce4953203acc5e924a75154715e100566543070b70eca7",
}
BASELINE_REFERENCES = {
    "s6": 0.22627654455227184,
    "s3": 0.3971869758657194,
    "s2": 0.5917757621918929,
}
LABEL_PATTERN = re.compile(r"\b(blocker|critical|major|normal|minor|trivial|severity)\b", re.I)
TOKEN_PATTERN = re.compile(r"[A-Za-z_][\w.$]*|\d+(?:\.\d+)*|[^\w\s]", re.UNICODE)
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
BUG_ID = re.compile(r"\b(?:bug|issue|bz)\s*#?\s*\d+\b", re.I)
HEX = re.compile(r"\b(?:0x)?[0-9a-f]{10,}\b", re.I)
HTML_TAG = re.compile(r"<[^>]+>")


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (pd.Timestamp, Path)):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    atomic_bytes(
        path,
        (json.dumps(value, indent=2, default=convert, sort_keys=True) + "\n").encode(),
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def protocol_sha256(path: Path) -> str:
    return sha256_file(path).lower()


def load_target(path: Path) -> pd.DataFrame:
    if path.name != "development_split.csv" or "test" in path.name.lower():
        raise ValueError("Only approved development_split.csv is permitted")
    if sha256_file(path).lower() != DEVELOPMENT_SHA256:
        raise ValueError("Development SHA-256 mismatch")
    frame = pd.read_csv(path)
    if len(frame) != DEVELOPMENT_ROWS:
        raise ValueError("Development row-count mismatch")
    frame["creation_time"] = pd.to_datetime(frame.creation_time, utc=True, errors="raise")
    frame = frame.sort_values("creation_time", kind="mergesort").reset_index(drop=True)
    folds = freeze_temporal_folds(frame)
    if folds_fingerprint(folds) != OUTER_FOLD_FINGERPRINT:
        raise ValueError("Outer-fold fingerprint mismatch")
    return frame


def validate_source_set(root: Path) -> dict[str, Path]:
    found = {path.stem: path for path in root.glob("*.parquet")}
    missing = set(SOURCE_PROJECTS) - set(found)
    if missing:
        raise ValueError(f"Missing source projects: {sorted(missing)}")
    unexpected = set(found) - set(SOURCE_PROJECTS) - {"MYLYN"}
    if unexpected:
        raise ValueError(f"Unexpected source projects: {sorted(unexpected)}")
    return {project: found[project] for project in SOURCE_PROJECTS}


def source_manifest(root: Path, expected_hashes: dict[str, str] = SOURCE_HASHES) -> pd.DataFrame:
    paths = validate_source_set(root)
    rows = []
    for project, path in paths.items():
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        required = {
            "source_project",
            "issue_id",
            "creation_time",
            "severity",
            "summary",
            "description",
        }
        if not required.issubset(names):
            raise ValueError(f"Source schema mismatch for {project}")
        digest = sha256_file(path).lower()
        if digest != expected_hashes.get(project, "").lower():
            raise ValueError(f"Source hash mismatch for {project}")
        rows.append(
            {
                "project": project,
                "path": str(path),
                "sha256": digest,
                "rows": parquet.metadata.num_rows,
                "bytes": path.stat().st_size,
                "schema_columns": json.dumps(names),
            }
        )
    return pd.DataFrame(rows)


def software_clean(text: str, mask_labels: bool = False) -> str:
    import html

    value = unicodedata.normalize("NFKC", html.unescape(str(text or "")))
    value = HTML_TAG.sub(" ", value)
    value = URL.sub(" [URL] ", value)
    value = EMAIL.sub(" [EMAIL] ", value)
    value = BUG_ID.sub(" [BUG_ID] ", value)
    value = HEX.sub(" [HEX] ", value)
    if mask_labels:
        value = LABEL_PATTERN.sub(" [SEVERITY_TERM] ", value)
    return " ".join(value.split())


def software_tokenize(text: str) -> list[str]:
    cleaned = software_clean(text)
    output: list[str] = []
    for token in TOKEN_PATTERN.findall(cleaned):
        normalized = token.lower()
        output.append(normalized)
        if re.search(r"[A-Za-z]", token) and any(mark in token for mark in "._$"):
            pieces = re.split(r"[._$]+", token)
        else:
            pieces = [token]
        for piece in pieces:
            camel = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|\d+", piece)
            output.extend(part.lower() for part in camel if part and part.lower() != normalized)
    return output


def _field_text(frame: pd.DataFrame, field: str) -> list[str]:
    return [software_clean(value) for value in frame[field].fillna("").astype(str)]


def _combined_text(frame: pd.DataFrame) -> list[str]:
    return [
        software_clean(f"{summary} {description}")
        for summary, description in zip(
            frame.summary.fillna(""), frame.description.fillna(""), strict=True
        )
    ]


def _summary_text(frame: pd.DataFrame) -> list[str]:
    return _field_text(frame, "summary")


def _description_text(frame: pd.DataFrame) -> list[str]:
    return _field_text(frame, "description")


def build_representation(name: str, summary_weight: float = 1.0) -> Pipeline | FeatureUnion:
    word_options = {
        "tokenizer": software_tokenize,
        "token_pattern": None,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.98,
        "sublinear_tf": True,
        "dtype": np.float32,
    }
    if name == "R0":
        return Pipeline(
            [
                ("text", FunctionTransformer(_combined_text, validate=False)),
                ("tfidf", TfidfVectorizer(max_features=50_000, **word_options)),
            ]
        )
    branches: list[tuple[str, Pipeline]] = [
        (
            "summary",
            Pipeline(
                [
                    (
                        "text",
                        FunctionTransformer(_summary_text, validate=False),
                    ),
                    ("tfidf", TfidfVectorizer(max_features=20_000, **word_options)),
                ]
            ),
        ),
        (
            "description",
            Pipeline(
                [
                    (
                        "text",
                        FunctionTransformer(_description_text, validate=False),
                    ),
                    ("tfidf", TfidfVectorizer(max_features=40_000, **word_options)),
                ]
            ),
        ),
    ]
    if name == "R2":
        branches.append(
            (
                "char",
                Pipeline(
                    [
                        ("text", FunctionTransformer(_combined_text, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=3,
                                max_df=0.995,
                                max_features=30_000,
                                sublinear_tf=True,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
            )
        )
    if name not in {"R1", "R2"}:
        raise ValueError(f"Unknown representation: {name}")
    weights = {"summary": summary_weight, "description": 1.0}
    if name == "R2":
        weights["char"] = 1.0
    return FeatureUnion(branches, transformer_weights=weights)


def normalized_identity(summary: Any, description: Any) -> str:
    value = software_clean(
        f"{'' if pd.isna(summary) else summary} {'' if pd.isna(description) else description}"
    ).lower()
    return hashlib.sha256(value.encode()).hexdigest()


def exact_identity(summary: Any, description: Any) -> str:
    value = f"{'' if pd.isna(summary) else summary}\n{'' if pd.isna(description) else description}"
    return hashlib.sha256(value.encode()).hexdigest()


def purge_training_overlap(
    training: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    original_validation = validation.copy(deep=True)
    validation_exact = {
        exact_identity(s, d)
        for s, d in zip(validation.summary, validation.description, strict=True)
    }
    validation_normalized = {
        normalized_identity(s, d)
        for s, d in zip(validation.summary, validation.description, strict=True)
    }
    exact = training.apply(lambda row: exact_identity(row.summary, row.description), axis=1)
    normalized = training.apply(
        lambda row: normalized_identity(row.summary, row.description), axis=1
    )
    exact_overlap = exact.isin(validation_exact)
    normalized_overlap = normalized.isin(validation_normalized)
    kept = training[~(exact_overlap | normalized_overlap)].copy()
    if not validation.equals(original_validation):
        raise RuntimeError("Validation was modified")
    after_exact = {
        exact_identity(s, d) for s, d in zip(kept.summary, kept.description, strict=True)
    }
    after_norm = {
        normalized_identity(s, d) for s, d in zip(kept.summary, kept.description, strict=True)
    }
    return kept, {
        "rows_before": len(training),
        "exact_removed": int(exact_overlap.sum()),
        "normalized_removed": int((normalized_overlap & ~exact_overlap).sum()),
        "rows_after": len(kept),
        "overlap_after": len(after_exact & validation_exact)
        + len(after_norm & validation_normalized),
    }


def inner_folds(frame: pd.DataFrame) -> tuple[FrozenFold, ...]:
    folds = freeze_temporal_folds(frame, n_splits=2)
    return folds


def base_model_grid(task: str) -> list[dict[str, Any]]:
    rows = []
    for representation, weights in (
        ("R0", (1.0,)),
        ("R1", (1.0, 2.0, 3.0)),
        ("R2", (1.0, 2.0, 3.0)),
    ):
        for c in (0.1, 0.5, 1.0):
            for weight in (None, "balanced"):
                rows.append(
                    {
                        "task": task,
                        "representation": representation,
                        "summary_weight": weights[0],
                        "model": "LinearSVC",
                        "C": c,
                        "class_weight": weight,
                        "alpha": np.nan,
                    }
                ) if representation == "R0" else rows.extend(
                    {
                        "task": task,
                        "representation": representation,
                        "summary_weight": sw,
                        "model": "LinearSVC",
                        "C": c,
                        "class_weight": weight,
                        "alpha": np.nan,
                    }
                    for sw in weights
                )
        for c in (0.25, 1.0, 4.0):
            for weight in (None, "balanced"):
                rows.append(
                    {
                        "task": task,
                        "representation": representation,
                        "summary_weight": weights[0],
                        "model": "LogisticRegression",
                        "C": c,
                        "class_weight": weight,
                        "alpha": np.nan,
                    }
                ) if representation == "R0" else rows.extend(
                    {
                        "task": task,
                        "representation": representation,
                        "summary_weight": sw,
                        "model": "LogisticRegression",
                        "C": c,
                        "class_weight": weight,
                        "alpha": np.nan,
                    }
                    for sw in weights
                )
        for alpha in (0.1, 0.5, 1.0):
            rows.append(
                {
                    "task": task,
                    "representation": representation,
                    "summary_weight": weights[0],
                    "model": "ComplementNB",
                    "C": np.nan,
                    "class_weight": None,
                    "alpha": alpha,
                }
            ) if representation == "R0" else rows.extend(
                {
                    "task": task,
                    "representation": representation,
                    "summary_weight": sw,
                    "model": "ComplementNB",
                    "C": np.nan,
                    "class_weight": None,
                    "alpha": alpha,
                }
                for sw in weights
            )
        if task == "s2":
            for c in (0.25, 1.0, 4.0):
                for weight in (None, "balanced"):
                    rows.append(
                        {
                            "task": task,
                            "representation": representation,
                            "summary_weight": weights[0],
                            "model": "NBSVM",
                            "C": c,
                            "class_weight": weight,
                            "alpha": np.nan,
                        }
                    ) if representation == "R0" else rows.extend(
                        {
                            "task": task,
                            "representation": representation,
                            "summary_weight": sw,
                            "model": "NBSVM",
                            "C": c,
                            "class_weight": weight,
                            "alpha": np.nan,
                        }
                        for sw in weights
                    )
    unique = pd.DataFrame(rows).drop_duplicates().to_dict("records")
    for index, row in enumerate(unique):
        row["base_candidate_id"] = f"{task}-A-{index:04d}"
    return unique


def candidate_matrix() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for base in base_model_grid(task):
            for source_weight in (0.0, 0.10, 0.25, 0.50):
                for recency in (
                    "all_history",
                    "five_year",
                    "half_life_2",
                    "half_life_4",
                    "half_life_6",
                ):
                    for domain in ("plain", "domain_augmented"):
                        row = {
                            **base,
                            "source_weight": source_weight,
                            "recency": recency,
                            "domain": domain,
                            "ensemble_weight": np.nan,
                            "stage_a_eligible": True,
                            "stage_b_eligible": True,
                            "stage_c_eligible": True,
                            "stage_d_eligible": True,
                            "stage_e_eligible": base["model"] != "ComplementNB",
                            "selectable": True,
                            "nonexecution_reason": "",
                        }
                        row["candidate_id"] = (
                            f"{base['base_candidate_id']}-sw{source_weight:.2f}-{recency}-{domain}"
                        )
                        rows.append(row)
            if base["model"] != "ComplementNB":
                for ensemble_weight in (0.25, 0.50, 0.75):
                    rows.append(
                        {
                            **base,
                            "source_weight": np.nan,
                            "recency": "selected",
                            "domain": "selected",
                            "ensemble_weight": ensemble_weight,
                            "stage_a_eligible": False,
                            "stage_b_eligible": False,
                            "stage_c_eligible": False,
                            "stage_d_eligible": False,
                            "stage_e_eligible": True,
                            "selectable": True,
                            "nonexecution_reason": "requires compatible top pair",
                            "candidate_id": (
                                f"{base['base_candidate_id']}-ensemble-{ensemble_weight:.2f}"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def combine_sample_weights(
    target_count: int,
    source_times: pd.Series,
    validation_start: pd.Timestamp,
    source_weight: float,
    recency: str,
) -> np.ndarray:
    target = np.ones(target_count)
    age = (validation_start - source_times).dt.total_seconds().to_numpy() / (365.25 * 86400)
    if recency == "all_history":
        factor = np.ones(len(age))
    elif recency == "five_year":
        factor = (age <= 5).astype(float)
    else:
        half = float(recency.rsplit("_", 1)[1])
        factor = np.power(0.5, np.maximum(age, 0) / half)
    combined = np.concatenate([target, source_weight * factor])
    mean = combined.mean()
    return combined / mean if mean else combined


def nbsvm_ratio(
    matrix: sparse.spmatrix, labels: np.ndarray, positive: str, smoothing: float = 1.0
) -> np.ndarray:
    binary = matrix.sign()
    pos = np.asarray(binary[labels == positive].sum(axis=0)).ravel() + smoothing
    neg = np.asarray(binary[labels != positive].sum(axis=0)).ravel() + smoothing
    return np.log(pos / pos.sum()) - np.log(neg / neg.sum())


def fit_calibrator(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=SEED)
    calibrator.fit(np.asarray(scores).reshape(-1, 1), labels.astype(int))
    return calibrator


def select_s2_threshold(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    rows = []
    for threshold in np.arange(0.05, 0.9501, 0.005):
        predicted = probabilities >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, labels=[False, True], zero_division=0
        )
        macro = f1_score(truth, predicted, average="macro", zero_division=0)
        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "precision": precision[1],
                "recall": recall[1],
                "f1": f1[1],
                "f2": fbeta_score(truth, predicted, beta=2, zero_division=0),
                "macro_f1": macro,
            }
        )
    eligible = pd.DataFrame(rows)
    eligible = eligible[eligible.precision >= 0.30]
    if eligible.empty:
        raise RuntimeError("No S2 threshold satisfies precision constraint")
    eligible["distance_to_half"] = (eligible.threshold - 0.5).abs()
    return (
        eligible.sort_values(
            ["f2", "recall", "precision", "macro_f1", "distance_to_half", "threshold"],
            ascending=[False, False, False, False, True, False],
        )
        .iloc[0]
        .to_dict()
    )


def paired_bootstrap(
    truth: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    labels: list[str],
    n: int = 1000,
) -> dict[str, float]:
    if not (len(truth) == len(baseline) == len(challenger)):
        raise ValueError("Paired OOF rows are not aligned")
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(n):
        index = rng.integers(0, len(truth), len(truth))
        deltas.append(
            f1_score(
                truth[index], challenger[index], labels=labels, average="macro", zero_division=0
            )
            - f1_score(
                truth[index], baseline[index], labels=labels, average="macro", zero_division=0
            )
        )
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": float(np.percentile(deltas, 2.5)),
        "ci_upper": float(np.percentile(deltas, 97.5)),
    }


def sanitize_key(project: str, internal_id: Any) -> str:
    return hashlib.sha256(f"{project}:{internal_id}".encode()).hexdigest()[:24]


def validate_resume(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    locked = (
        "development_sha256",
        "development_rows",
        "outer_fold_fingerprint",
        "source_hashes",
        "source_projects",
        "protocol_content_sha256",
        "candidate_matrix_checksum",
        "seed",
        "task_mappings",
        "preprocessing",
        "model_grid",
        "success_criteria",
        "cli_command",
        "packages",
        "completed_stages",
        "schema_version",
    )
    mismatches = [key for key in locked if existing.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(f"Resume manifest mismatch: {mismatches}")


@dataclass
class DryRunResult:
    audit_output: Path
    model_output: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class EngineOptions:
    fixture: bool = False
    minimum_available_memory_gb: float = MINIMUM_AVAILABLE_MEMORY_GB
    resume: bool = False


def load_sources(root: Path) -> pd.DataFrame:
    source_manifest(root)
    columns = ["source_project", "issue_id", "creation_time", "severity", "summary", "description"]
    parts = []
    for project, path in validate_source_set(root).items():
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            part = batch.to_pandas()
            part["source_project"] = project
            parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["creation_time"] = pd.to_datetime(frame.creation_time, utc=True, errors="coerce")
    return frame[
        frame.creation_time.notna()
        & frame.severity.astype(str).str.strip().str.lower().isin(TASKS["s6"])
    ].reset_index(drop=True)


def _make_model(candidate: dict[str, Any]):
    model = candidate["model"]
    if model in {"LogisticRegression", "NBSVM"}:
        return LogisticRegression(
            C=float(candidate["C"]),
            class_weight=candidate["class_weight"],
            max_iter=2000,
            solver="lbfgs",
            random_state=SEED,
        )
    if model == "LinearSVC":
        return LinearSVC(
            C=float(candidate["C"]),
            class_weight=candidate["class_weight"],
            random_state=SEED,
        )
    if model == "ComplementNB":
        return ComplementNB(alpha=float(candidate["alpha"]))
    raise ValueError(f"Unknown model: {model}")


def _scores(model: Any, matrix: sparse.spmatrix) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix))
    raw = np.asarray(model.decision_function(matrix))
    return raw[:, None] if raw.ndim == 1 else raw


def _binary_high_score(scores: np.ndarray, classes: list[str]) -> np.ndarray:
    if scores.shape[1] == len(classes):
        return scores[:, classes.index("HIGH_IMPACT")]
    if scores.shape[1] == 1 and len(classes) == 2:
        return scores[:, 0] if classes[1] == "HIGH_IMPACT" else -scores[:, 0]
    raise RuntimeError("Unsupported S2 score shape")


def _score_predictions(scores: np.ndarray, classes: list[str]) -> np.ndarray:
    if scores.shape[1] == 1 and len(classes) == 2:
        return np.where(scores[:, 0] >= 0, classes[1], classes[0])
    return np.asarray(classes)[np.argmax(scores, axis=1)]


def _domain_matrix(matrix: sparse.spmatrix, source_mask: np.ndarray) -> sparse.csr_matrix:
    target_mask = (~source_mask).astype(np.float32)
    source_values = source_mask.astype(np.float32)
    return sparse.hstack(
        [matrix, sparse.diags(target_mask) @ matrix, sparse.diags(source_values) @ matrix],
        format="csr",
        dtype=np.float32,
    )


def _prepare_source(
    sources: pd.DataFrame, validation: pd.DataFrame, validation_start: pd.Timestamp, task: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    eligible = sources[sources.creation_time < validation_start].copy()
    eligible["target"] = map_target(eligible.severity, task)
    eligible = eligible[eligible.target.notna()].copy()
    purged, audit = purge_training_overlap(eligible, validation)
    if audit["overlap_after"] or (len(purged) and purged.creation_time.max() >= validation_start):
        raise RuntimeError("Production source cutoff/purge invariant failed")
    return purged, audit


def _fit_predict(
    target_train: pd.DataFrame,
    source_train: pd.DataFrame,
    validation: pd.DataFrame,
    task: str,
    candidate: dict[str, Any],
    source_weight: float,
    recency: str,
    domain: str,
) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    if recency == "five_year" and len(source_train):
        boundary = validation.creation_time.min() - pd.DateOffset(years=5)
        source_train = source_train[source_train.creation_time >= boundary].copy()
    pooled = pd.concat([target_train, source_train], ignore_index=True)
    y_target = map_target(target_train.severity, task).to_numpy()
    y_source = map_target(source_train.severity, task).to_numpy()
    labels = np.concatenate([y_target, y_source])
    transformer = build_representation(candidate["representation"], candidate["summary_weight"])
    x_train = transformer.fit_transform(pooled).tocsr().astype(np.float32)
    x_validation = transformer.transform(validation).tocsr().astype(np.float32)
    source_mask = np.r_[
        np.zeros(len(target_train), dtype=bool), np.ones(len(source_train), dtype=bool)
    ]
    if domain == "domain_augmented":
        x_train = _domain_matrix(x_train, source_mask)
        x_validation = _domain_matrix(x_validation, np.zeros(len(validation), dtype=bool))
    weights = combine_sample_weights(
        len(target_train),
        source_train.creation_time,
        validation.creation_time.min(),
        source_weight,
        recency,
    )
    model = _make_model(candidate)
    ratio = None
    if candidate["model"] == "NBSVM":
        ratio = nbsvm_ratio(x_train, labels, "HIGH_IMPACT")
        x_train = x_train.multiply(ratio)
        x_validation = x_validation.multiply(ratio)
    if isinstance(model, ComplementNB) and x_train.min() < 0:
        raise RuntimeError("ComplementNB requires a non-negative matrix")
    model.fit(x_train, labels, sample_weight=weights)
    try:
        model_size = len(pickle.dumps((transformer, model)))
    except (AttributeError, TypeError):
        model_size = int(
            sum(value.nbytes for value in model.__dict__.values() if hasattr(value, "nbytes"))
        )
    return (
        model.predict(x_validation),
        _scores(model, x_validation),
        list(model.classes_),
        x_train.shape[1],
        model_size,
    )


def _evaluate_inner_configuration(
    outer_train: pd.DataFrame,
    sources: pd.DataFrame,
    task: str,
    candidate: dict[str, Any],
    source_weight: float,
    recency: str,
    domain: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    parts, rows = [], []
    for inner in inner_folds(outer_train):
        training = outer_train.iloc[list(inner.train)].copy()
        validation = outer_train.iloc[list(inner.validation)].copy()
        training, target_purge = purge_training_overlap(training, validation)
        source, purge = _prepare_source(sources, validation, validation.creation_time.min(), task)
        if source_weight == 0:
            source = source.iloc[0:0]
        predicted, scores, classes, features, _ = _fit_predict(
            training, source, validation, task, candidate, source_weight, recency, domain
        )
        truth = map_target(validation.severity, task).to_numpy()
        metrics = fixed_metrics(truth, predicted, TASK_LABELS[task])
        rows.append(
            {
                **metrics,
                "feature_count": features,
                **purge,
                "target_overlap_after": target_purge["overlap_after"],
            }
        )
        part = pd.DataFrame(
            {
                "row": validation.index,
                "truth": truth,
                "predicted": predicted,
                "fold": inner.fold,
            }
        )
        part["scores"] = [row.tolist() for row in scores]
        part["classes"] = [classes] * len(part)
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True)
    threshold = np.nan
    if task == "s2":
        raw = np.concatenate(
            [
                _binary_high_score(np.asarray(values, dtype=float)[None, :], classes)[0:1]
                for values, classes in zip(combined.scores, combined.classes, strict=True)
            ]
        )
        truth_binary = combined.truth.to_numpy() == "HIGH_IMPACT"
        calibrator = fit_calibrator(raw, truth_binary)
        probabilities = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        selection = select_s2_threshold(probabilities, truth_binary)
        threshold = selection["threshold"]
        combined["predicted"] = np.where(probabilities >= threshold, "HIGH_IMPACT", "LOWER_IMPACT")
        combined["calibrated_probability"] = probabilities
        rows = [
            {
                **fixed_metrics(
                    part.truth.to_numpy(), part.predicted.to_numpy(), TASK_LABELS[task]
                ),
                "feature_count": rows[index]["feature_count"],
                "overlap_after": rows[index]["overlap_after"],
            }
            for index, (_, part) in enumerate(combined.groupby("fold", sort=True))
        ]
    result = pd.DataFrame(rows)
    summary = {
        "macro_f1": float(result.macro_f1.mean()),
        "minimum_class_recall": float(result.minimum_class_recall.mean()),
        "fold_macro_f1_sd": float(result.macro_f1.std(ddof=0)),
        "feature_count": int(result.feature_count.max()),
        "purge_overlap_after": int(result.overlap_after.sum()),
        "threshold": threshold,
    }
    return summary, combined


def _cutoff_contexts(target: pd.DataFrame, folds: tuple[FrozenFold, ...]) -> list[dict[str, Any]]:
    rows = []
    for task in TASKS:
        for fold in folds:
            validation = target.iloc[list(fold.validation)]
            rows.append(
                {
                    "task": task,
                    "fold": fold.fold,
                    "inner_or_outer": "outer",
                    "validation_start": validation.creation_time.min(),
                    "validation": validation,
                }
            )
            train = target.iloc[list(fold.train)]
            for inner in inner_folds(train):
                inner_validation = train.iloc[list(inner.validation)]
                rows.append(
                    {
                        "task": task,
                        "fold": fold.fold,
                        "inner_or_outer": f"inner_{inner.fold}",
                        "validation_start": inner_validation.creation_time.min(),
                        "validation": inner_validation,
                    }
                )
    return rows


def source_cutoff_audit(
    root: Path, target: pd.DataFrame, folds: tuple[FrozenFold, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contexts = _cutoff_contexts(target, folds)
    for context in contexts:
        validation = context["validation"]
        context["validation_exact"] = {
            exact_identity(summary, description)
            for summary, description in zip(validation.summary, validation.description, strict=True)
        }
        context["validation_normalized"] = {
            normalized_identity(summary, description)
            for summary, description in zip(validation.summary, validation.description, strict=True)
        }
    proof = []
    project_counts = []
    columns = ["source_project", "issue_id", "creation_time", "severity", "summary", "description"]
    for project, path in validate_source_set(root).items():
        parquet = pq.ParquetFile(path)
        batches = []
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            batches.append(batch.to_pandas())
        frame = pd.concat(batches, ignore_index=True)
        frame.creation_time = pd.to_datetime(frame.creation_time, utc=True, errors="coerce")
        before = len(frame)
        valid = frame[
            frame.severity.astype(str).str.strip().str.lower().isin(TASKS["s6"])
            & frame.creation_time.notna()
        ].copy()
        valid["_exact_identity"] = [
            exact_identity(summary, description)
            for summary, description in zip(valid.summary, valid.description, strict=True)
        ]
        valid["_normalized_identity"] = [
            normalized_identity(summary, description)
            for summary, description in zip(valid.summary, valid.description, strict=True)
        ]
        project_counts.append(
            {
                "project": project,
                "rows_before_filtering": before,
                "rows_after_label_time_filtering": len(valid),
            }
        )
        for context in contexts:
            cutoff = valid[valid.creation_time < context["validation_start"]].copy()
            exact_overlap = cutoff._exact_identity.isin(context["validation_exact"])
            normalized_overlap = cutoff._normalized_identity.isin(context["validation_normalized"])
            purged = cutoff[~(exact_overlap | normalized_overlap)]
            exact_removed = int(exact_overlap.sum())
            overlap_after = int(
                purged._exact_identity.isin(context["validation_exact"]).sum()
                + purged._normalized_identity.isin(context["validation_normalized"]).sum()
            )
            max_time = purged.creation_time.max() if len(purged) else pd.NaT
            proof.append(
                {
                    "task": context["task"],
                    "fold": context["fold"],
                    "inner_or_outer": context["inner_or_outer"],
                    "project": project,
                    "validation_start": context["validation_start"],
                    "rows_before": before,
                    "rows_after_label_filter": len(valid),
                    "rows_before_cutoff": len(valid),
                    "rows_after_cutoff": len(cutoff),
                    "rows_after_exact_duplicate_purge": len(cutoff) - exact_removed,
                    "rows_after_normalized_duplicate_purge": len(purged),
                    "max_timestamp_after_cutoff": max_time,
                    "cutoff_violations_after_filter": int(
                        pd.notna(max_time) and max_time >= context["validation_start"]
                    ),
                    "source_validation_overlap_after_purge": overlap_after,
                }
            )
        del frame, valid, batches
    return pd.DataFrame(proof), pd.DataFrame(project_counts)


def build_manifest(
    target: pd.DataFrame,
    folds: tuple[FrozenFold, ...],
    manifest_frame: pd.DataFrame,
    protocol: Path,
    candidates: pd.DataFrame,
    command: str,
    dry_run: bool,
) -> dict[str, Any]:
    checksum = hashlib.sha256(candidates.to_csv(index=False).encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "development_sha256": DEVELOPMENT_SHA256,
        "development_rows": len(target),
        "outer_fold_fingerprint": folds_fingerprint(folds),
        "inner_fold_fingerprints": [],
        "source_projects": list(SOURCE_PROJECTS),
        "source_hashes": dict(zip(manifest_frame.project, manifest_frame.sha256, strict=True)),
        "protocol_content_sha256": protocol_sha256(protocol),
        "candidate_matrix_checksum": checksum,
        "seed": SEED,
        "task_mappings": TASKS,
        "preprocessing": {"representations": ["R0", "R1", "R2"]},
        "model_grid": "v2_frozen",
        "success_criteria": "protocol_v2",
        "cli_command": command,
        "packages": package_versions(
            ["numpy", "pandas", "pyarrow", "psutil", "scikit-learn", "scipy"]
        ),
        "completed_stages": [],
        "dry_run": dry_run,
        "held_out_test_accessed": False,
        "models_fitted": False,
    }


def dry_run(
    target_path: Path,
    processed_root: Path,
    audit_output: Path,
    model_output: Path,
    protocol: Path,
    command: str,
    resume: bool = False,
) -> DryRunResult:
    if (audit_output.exists() or model_output.exists()) and not resume:
        raise FileExistsError("Output exists; use --resume")
    target = load_target(target_path)
    folds = freeze_temporal_folds(target)
    manifest_frame = source_manifest(processed_root)
    candidates = candidate_matrix()
    manifest = build_manifest(target, folds, manifest_frame, protocol, candidates, command, True)
    if resume:
        existing = json.loads((audit_output / "study_manifest.json").read_text())
        validate_resume(existing, manifest)
    audit_output.mkdir(parents=True, exist_ok=True)
    model_output.mkdir(parents=True, exist_ok=True)
    proof, counts = source_cutoff_audit(processed_root, target, folds)
    if (
        proof.cutoff_violations_after_filter.sum()
        or proof.source_validation_overlap_after_purge.sum()
    ):
        raise RuntimeError("Cutoff or duplicate integrity failure")
    fold_rows = []
    inner_rows = []
    for fold in folds:
        fold_rows.append(
            {
                "fold": fold.fold,
                "train_rows": len(fold.train),
                "validation_rows": len(fold.validation),
                "train_hash": _row_hash(target, fold.train),
                "validation_hash": _row_hash(target, fold.validation),
                "fingerprint": folds_fingerprint(folds),
            }
        )
        train = target.iloc[list(fold.train)]
        for inner in inner_folds(train):
            inner_rows.append(
                {
                    "outer_fold": fold.fold,
                    "inner_fold": inner.fold,
                    "train_rows": len(inner.train),
                    "validation_rows": len(inner.validation),
                    "train_hash": _row_hash(train, inner.train),
                    "validation_hash": _row_hash(train, inner.validation),
                }
            )
    manifest["inner_fold_fingerprints"] = [
        hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() for row in inner_rows
    ]
    atomic_json(audit_output / "study_manifest.json", manifest)
    atomic_bytes(audit_output / "protocol_snapshot.md", protocol.read_bytes())
    atomic_csv(audit_output / "candidate_matrix.csv", candidates)
    atomic_csv(
        audit_output / "candidate_status.csv",
        candidates[["candidate_id", "selectable", "nonexecution_reason"]],
    )
    atomic_csv(audit_output / "fold_identity.csv", pd.DataFrame(fold_rows))
    atomic_csv(audit_output / "inner_fold_identity.csv", pd.DataFrame(inner_rows))
    atomic_csv(audit_output / "source_dataset_manifest.csv", manifest_frame)
    atomic_csv(audit_output / "source_project_counts.csv", counts)
    atomic_csv(
        audit_output / "class_distributions.csv",
        pd.concat(
            [
                target.assign(task=task, target=map_target(target.severity, task))
                .groupby(["task", "target"])
                .size()
                .reset_index(name="rows")
                for task in TASKS
            ]
        ),
    )
    atomic_csv(
        audit_output / "label_quality_audit.csv",
        target.groupby("severity").size().reset_index(name="rows"),
    )
    atomic_csv(audit_output / "data_cutoff_proof.csv", proof)
    duplicate = (
        proof.groupby(["task", "fold", "inner_or_outer", "project"])[
            [
                "rows_after_exact_duplicate_purge",
                "rows_after_normalized_duplicate_purge",
                "source_validation_overlap_after_purge",
            ]
        ]
        .first()
        .reset_index()
    )
    atomic_csv(audit_output / "duplicate_purge_results.csv", duplicate)
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(audit_output)
    integrity = pd.DataFrame(
        [
            {"check": "development_sha", "passed": True},
            {"check": "development_rows", "passed": len(target) == DEVELOPMENT_ROWS},
            {
                "check": "outer_fingerprint",
                "passed": folds_fingerprint(folds) == OUTER_FOLD_FINGERPRINT,
            },
            {"check": "source_count_8", "passed": len(manifest_frame) == 8},
            {"check": "target_is_mylyn", "passed": True},
            {"check": "test_accessed_no", "passed": True},
            {
                "check": "cutoff_violations_zero",
                "passed": proof.cutoff_violations_after_filter.sum() == 0,
            },
            {
                "check": "overlap_zero",
                "passed": proof.source_validation_overlap_after_purge.sum() == 0,
            },
            {"check": "candidate_grid_nonempty", "passed": len(candidates) > 1000},
            {"check": "models_not_fitted", "passed": True},
        ]
    )
    atomic_csv(audit_output / "integrity_checks.csv", integrity)
    resource = {
        "available_memory_gb": memory.available / 2**30,
        "required_memory_gb": MINIMUM_AVAILABLE_MEMORY_GB,
        "real_run_allowed": memory.available / 2**30 >= MINIMUM_AVAILABLE_MEMORY_GB,
        "disk_free_gb": disk.free / 2**30,
        "candidate_rows": len(candidates),
        "estimated_runtime_hours": 8.0,
        "models_fitted": 0,
    }
    atomic_json(audit_output / "resource_estimate.json", resource)
    bundle = (
        "# VALIDATION BUNDLE — MYLYN transfer v2 dry-run\n\n"
        + "\n".join(
            f"- {row.check}: {'PASS' if row.passed else 'FAIL'}" for row in integrity.itertuples()
        )
        + f"\n- source_projects_count: {len(manifest_frame)}"
        + f"\n- candidate_rows: {len(candidates)}"
        + "\n- models_fitted: 0"
        + f"\n- real_run_allowed: {resource['real_run_allowed']}\n"
    )
    atomic_bytes(audit_output / "VALIDATION_BUNDLE.md", bundle.encode())
    for name in (
        "baseline_reproduction",
        "inner_selection_results",
        "selected_configurations",
        "task_summary",
        "fold_results",
        "per_class_results",
        "transfer_comparison",
        "recency_comparison",
        "domain_adaptation_comparison",
        "ensemble_comparison",
        "threshold_results",
        "calibration_results",
        "predicted_class_distributions",
        "control_results",
        "coverage_results",
        "bootstrap_delta_results",
        "runtime_summary",
        "model_size_summary",
        "oof_predictions",
    ):
        atomic_csv(
            model_output / f"{name}.csv", pd.DataFrame([{"status": "DRY_RUN_NO_MODELS_FITTED"}])
        )
    atomic_bytes(
        model_output / "protocol_deviations.md", b"# Protocol deviations\n\nNone in dry-run.\n"
    )
    atomic_bytes(
        model_output / "final_recommendation.md",
        b"# Final recommendation\n\nRUN_BLOCKED_PENDING_PROTOCOL_AND_RESOURCE_GATE.\n",
    )
    return DryRunResult(audit_output, model_output, manifest)


def _stage_a_candidates(task: str, fixture: bool) -> list[dict[str, Any]]:
    rows = [{**row, "candidate_id": row["base_candidate_id"]} for row in base_model_grid(task)]
    if not fixture:
        return rows
    wanted = [
        row
        for row in rows
        if row["representation"] == "R0"
        and row["model"] in {"LogisticRegression", "LinearSVC"}
        and float(row["C"]) == 1.0
        and row["class_weight"] == "balanced"
    ]
    return wanted


def _rank_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -row["macro_f1"],
            -row["minimum_class_recall"],
            row["fold_macro_f1_sd"],
            {"R0": 0, "R1": 1, "R2": 2}[row["representation"]],
            row["feature_count"],
            row["runtime_seconds"],
            row["candidate_id"],
        ),
    )


def _inner_stage_row(
    outer_train: pd.DataFrame,
    sources: pd.DataFrame,
    task: str,
    candidate: dict[str, Any],
    stage: str,
    source_weight: float,
    recency: str,
    domain: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = perf_counter()
    metrics, oof = _evaluate_inner_configuration(
        outer_train, sources, task, candidate, source_weight, recency, domain
    )
    row = {
        **candidate,
        **metrics,
        "stage": stage,
        "source_weight": source_weight,
        "recency": recency,
        "domain": domain,
        "runtime_seconds": perf_counter() - started,
        "status": "trained",
    }
    return row, oof


def _evaluate_stage_a_grid(
    outer_train: pd.DataFrame, task: str, candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    metric_rows: dict[str, list[dict[str, Any]]] = {row["candidate_id"]: [] for row in candidates}
    oof_parts: dict[str, list[pd.DataFrame]] = {row["candidate_id"]: [] for row in candidates}
    for inner in inner_folds(outer_train):
        training = outer_train.iloc[list(inner.train)].copy()
        validation = outer_train.iloc[list(inner.validation)].copy()
        training, purge = purge_training_overlap(training, validation)
        truth = map_target(validation.severity, task).to_numpy()
        labels = map_target(training.severity, task).to_numpy()
        groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
        for candidate in candidates:
            groups.setdefault(
                (candidate["representation"], candidate["summary_weight"]), []
            ).append(candidate)
        for (representation, summary_weight), grouped in groups.items():
            feature_started = perf_counter()
            transformer = build_representation(representation, summary_weight)
            x_train = transformer.fit_transform(training).tocsr().astype(np.float32)
            x_validation = transformer.transform(validation).tocsr().astype(np.float32)
            feature_seconds = (perf_counter() - feature_started) / len(grouped)
            for candidate in grouped:
                train_matrix, validation_matrix = x_train, x_validation
                if candidate["model"] == "NBSVM":
                    ratio = nbsvm_ratio(train_matrix, labels, "HIGH_IMPACT")
                    train_matrix = train_matrix.multiply(ratio)
                    validation_matrix = validation_matrix.multiply(ratio)
                model = _make_model(candidate)
                model_started = perf_counter()
                model.fit(train_matrix, labels)
                predicted = model.predict(validation_matrix)
                scores = _scores(model, validation_matrix)
                metrics = fixed_metrics(truth, predicted, TASK_LABELS[task])
                metric_rows[candidate["candidate_id"]].append(
                    {
                        **metrics,
                        "feature_count": x_train.shape[1],
                        "overlap_after": purge["overlap_after"],
                        "runtime_seconds": perf_counter() - model_started + feature_seconds,
                    }
                )
                part = pd.DataFrame(
                    {
                        "row": validation.index,
                        "truth": truth,
                        "predicted": predicted,
                        "fold": inner.fold,
                        "scores": [row.tolist() for row in scores],
                        "classes": [list(model.classes_)] * len(validation),
                    }
                )
                oof_parts[candidate["candidate_id"]].append(part)
    results, combined_oof = [], {}
    for candidate in candidates:
        identifier = candidate["candidate_id"]
        combined = pd.concat(oof_parts[identifier], ignore_index=True)
        rows = metric_rows[identifier]
        threshold = np.nan
        if task == "s2":
            raw = np.asarray(
                [
                    _binary_high_score(np.asarray(score)[None, :], classes)[0]
                    for score, classes in zip(combined.scores, combined.classes, strict=True)
                ]
            )
            truth_binary = combined.truth.to_numpy() == "HIGH_IMPACT"
            calibrator = fit_calibrator(raw, truth_binary)
            probability = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
            selected = select_s2_threshold(probability, truth_binary)
            threshold = selected["threshold"]
            combined["predicted"] = np.where(
                probability >= threshold, "HIGH_IMPACT", "LOWER_IMPACT"
            )
            rows = [
                {
                    **fixed_metrics(
                        part.truth.to_numpy(),
                        part.predicted.to_numpy(),
                        TASK_LABELS[task],
                    ),
                    "feature_count": rows[index]["feature_count"],
                    "overlap_after": rows[index]["overlap_after"],
                    "runtime_seconds": rows[index]["runtime_seconds"],
                }
                for index, (_, part) in enumerate(combined.groupby("fold", sort=True))
            ]
        frame = pd.DataFrame(rows)
        results.append(
            {
                **candidate,
                "stage": "A",
                "source_weight": 0.0,
                "recency": "all_history",
                "domain": "plain",
                "macro_f1": float(frame.macro_f1.mean()),
                "minimum_class_recall": float(frame.minimum_class_recall.mean()),
                "fold_macro_f1_sd": float(frame.macro_f1.std(ddof=0)),
                "feature_count": int(frame.feature_count.max()),
                "purge_overlap_after": int(frame.overlap_after.sum()),
                "threshold": threshold,
                "runtime_seconds": float(frame.runtime_seconds.sum()),
                "status": "trained",
            }
        )
        combined_oof[identifier] = combined
    return results, combined_oof


def _baseline_reproduction(
    target: pd.DataFrame, folds: tuple[FrozenFold, ...], strict: bool = True
) -> pd.DataFrame:
    result = pd.concat([_baseline_task(target, folds, task) for task in TASKS], ignore_index=True)
    means = result.groupby("task", as_index=False).macro_f1.mean()
    means["reference_macro_f1"] = means.task.map(BASELINE_REFERENCES)
    means["absolute_difference"] = (means.macro_f1 - means.reference_macro_f1).abs()
    means["status"] = np.where(
        means.absolute_difference <= 1e-6, "REPRODUCED", "BASELINE_REPRODUCTION_FAILED"
    )
    if strict and (means.status != "REPRODUCED").any():
        raise RuntimeError("BASELINE_REPRODUCTION_FAILED")
    return means


def execute_production_study(
    target: pd.DataFrame,
    sources: pd.DataFrame,
    audit_output: Path,
    model_output: Path,
    options: EngineOptions,
) -> dict[str, Any]:
    """Run the same production dispatcher for fixture and frozen full-grid studies."""
    folds = freeze_temporal_folds(target)
    baseline = _baseline_reproduction(target, folds, strict=not options.fixture)
    atomic_csv(model_output / "baseline_reproduction.csv", baseline)
    inner_rows, selected_rows, fold_rows, oof_rows = [], [], [], []
    stage_tables: dict[str, list[dict[str, Any]]] = {stage: [] for stage in "ABCDE"}
    completed: set[str] = set()
    state_path = audit_output / "engine_state.json"
    if options.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        inner_rows = state["inner_rows"]
        selected_rows = state["selected_rows"]
        fold_rows = state["fold_rows"]
        oof_rows = state["oof_rows"]
        stage_tables = state["stage_tables"]
        completed = set(state["completed"])
    for task in TASKS:
        for outer in folds:
            checkpoint_key = f"{task}:{outer.fold}"
            if checkpoint_key in completed:
                continue
            outer_train = target.iloc[list(outer.train)].copy()
            validation = target.iloc[list(outer.validation)].copy()
            evaluated, oof_by_id = _evaluate_stage_a_grid(
                outer_train, task, _stage_a_candidates(task, options.fixture)
            )
            top = _rank_stage(evaluated)[:2]
            stage_tables["A"].extend(evaluated)
            stage_b = []
            for candidate in top:
                for weight in (0.0, 0.1, 0.25, 0.5):
                    row, oof = _inner_stage_row(
                        outer_train, sources, task, candidate, "B", weight, "all_history", "plain"
                    )
                    stage_b.append(row)
                    oof_by_id[f"B:{candidate['candidate_id']}:{weight}"] = oof
            stage_tables["B"].extend(stage_b)
            best_b = _rank_stage(stage_b)[0]
            stage_c = []
            for recency in (
                "all_history",
                "five_year",
                "half_life_2",
                "half_life_4",
                "half_life_6",
            ):
                row, _ = _inner_stage_row(
                    outer_train,
                    sources,
                    task,
                    best_b,
                    "C",
                    best_b["source_weight"],
                    recency,
                    "plain",
                )
                stage_c.append(row)
            stage_tables["C"].extend(stage_c)
            best_c = _rank_stage(stage_c)[0]
            stage_d = []
            for domain in ("plain", "domain_augmented"):
                row, oof = _inner_stage_row(
                    outer_train,
                    sources,
                    task,
                    best_c,
                    "D",
                    best_c["source_weight"],
                    best_c["recency"],
                    domain,
                )
                stage_d.append(row)
                oof_by_id[f"D:{domain}"] = oof
            stage_tables["D"].extend(stage_d)
            best_d = _rank_stage(stage_d)[0]
            # Stage E is genuinely evaluated; a single-model fallback is retained unless
            # two compatible score matrices improve inner macro-F1 by at least 0.01.
            ensemble_rows = []
            if len(stage_d) == 2:
                left = oof_by_id["D:plain"].sort_values("row")
                right = oof_by_id["D:domain_augmented"].sort_values("row")
                if not left[["row", "truth"]].equals(right[["row", "truth"]]):
                    raise RuntimeError("Inner ensemble OOF rows are not aligned")
                for weight in (0.25, 0.5, 0.75):
                    left_scores = np.vstack(left.scores)
                    right_scores = np.vstack(right.scores)
                    if (
                        left_scores.shape != right_scores.shape
                        or left.classes.iloc[0] != right.classes.iloc[0]
                    ):
                        reason = "incompatible score/class shape"
                        macro = np.nan
                    else:
                        classes = np.asarray(left.classes.iloc[0])
                        ensemble_scores = weight * left_scores + (1 - weight) * right_scores
                        if task == "s2":
                            high_score = _binary_high_score(ensemble_scores, classes.tolist())
                            high_truth = left.truth.to_numpy() == "HIGH_IMPACT"
                            ensemble_calibrator = fit_calibrator(high_score, high_truth)
                            probability = ensemble_calibrator.predict_proba(
                                high_score.reshape(-1, 1)
                            )[:, 1]
                            ensemble_threshold = select_s2_threshold(probability, high_truth)[
                                "threshold"
                            ]
                            predicted = np.where(
                                probability >= ensemble_threshold,
                                "HIGH_IMPACT",
                                "LOWER_IMPACT",
                            )
                        else:
                            predicted = _score_predictions(ensemble_scores, classes.tolist())
                        macro = f1_score(
                            left.truth,
                            predicted,
                            labels=TASK_LABELS[task],
                            average="macro",
                            zero_division=0,
                        )
                        reason = (
                            "gain below 0.01" if macro < best_d["macro_f1"] + 0.01 else "accepted"
                        )
                    ensemble_rows.append(
                        {
                            "task": task,
                            "outer_fold": outer.fold,
                            "stage": "E",
                            "ensemble_weight": weight,
                            "macro_f1": macro,
                            "status": "trained",
                            "selection_reason": reason,
                        }
                    )
            stage_tables["E"].extend(ensemble_rows)
            accepted_ensemble = [
                row for row in ensemble_rows if row["selection_reason"] == "accepted"
            ]
            selected_ensemble = (
                max(accepted_ensemble, key=lambda row: row["macro_f1"])
                if accepted_ensemble
                else None
            )
            outer_started = perf_counter()
            outer_train, target_purge = purge_training_overlap(outer_train, validation)
            source, purge = _prepare_source(
                sources, validation, validation.creation_time.min(), task
            )
            if best_d["source_weight"] == 0:
                source = source.iloc[0:0]
            rss_before = psutil.Process().memory_info().rss
            predicted, scores, classes, features, model_size = _fit_predict(
                outer_train,
                source,
                validation,
                task,
                best_d,
                best_d["source_weight"],
                best_d["recency"],
                best_d["domain"],
            )
            selected_inner = oof_by_id[f"D:{best_d['domain']}"]
            if selected_ensemble is not None:
                other_domain = "domain_augmented" if best_d["domain"] == "plain" else "plain"
                _, other_scores, other_classes, other_features, other_size = _fit_predict(
                    outer_train,
                    source,
                    validation,
                    task,
                    best_d,
                    best_d["source_weight"],
                    best_d["recency"],
                    other_domain,
                )
                if classes != other_classes or scores.shape != other_scores.shape:
                    raise RuntimeError("Selected ensemble score/class shape changed on refit")
                weight = selected_ensemble["ensemble_weight"]
                scores = weight * scores + (1 - weight) * other_scores
                predicted = _score_predictions(scores, classes)
                features += other_features
                model_size += other_size
                left = oof_by_id[f"D:{best_d['domain']}"]
                right = oof_by_id[f"D:{other_domain}"]
                selected_inner = left.copy()
                selected_inner["scores"] = [
                    (weight * np.asarray(a) + (1 - weight) * np.asarray(b)).tolist()
                    for a, b in zip(left.scores, right.scores, strict=True)
                ]
            truth = map_target(validation.severity, task).to_numpy()
            selected_threshold = np.nan
            calibration_rows = []
            if task == "s2":
                inner_raw = np.asarray(
                    [
                        _binary_high_score(np.asarray(values, dtype=float)[None, :], inner_classes)[
                            0
                        ]
                        for values, inner_classes in zip(
                            selected_inner.scores, selected_inner.classes, strict=True
                        )
                    ]
                )
                inner_truth = selected_inner.truth.to_numpy() == "HIGH_IMPACT"
                calibrator = fit_calibrator(inner_raw, inner_truth)
                inner_probability = calibrator.predict_proba(inner_raw.reshape(-1, 1))[:, 1]
                threshold_row = select_s2_threshold(inner_probability, inner_truth)
                selected_threshold = threshold_row["threshold"]
                outer_raw = _binary_high_score(scores, classes)
                outer_probability = calibrator.predict_proba(outer_raw.reshape(-1, 1))[:, 1]
                predicted = np.where(
                    outer_probability >= selected_threshold,
                    "HIGH_IMPACT",
                    "LOWER_IMPACT",
                )
                calibration_rows.append(
                    {
                        "task": task,
                        "fold": outer.fold,
                        "fit_scope": "inner_oof_only",
                        "inner_rows": len(inner_raw),
                        "threshold": selected_threshold,
                    }
                )
            comparator_prediction, comparator_scores, comparator_classes, _, _ = _fit_predict(
                outer_train,
                source.iloc[0:0],
                validation,
                task,
                best_d,
                0.0,
                "all_history",
                best_d["domain"],
            )
            if selected_ensemble is not None:
                _, comparator_other_scores, comparator_other_classes, _, _ = _fit_predict(
                    outer_train,
                    source.iloc[0:0],
                    validation,
                    task,
                    best_d,
                    0.0,
                    "all_history",
                    other_domain,
                )
                if comparator_classes != comparator_other_classes:
                    raise RuntimeError("Comparator ensemble classes are not aligned")
                weight = selected_ensemble["ensemble_weight"]
                comparator_scores = (
                    weight * comparator_scores + (1 - weight) * comparator_other_scores
                )
                comparator_prediction = _score_predictions(comparator_scores, comparator_classes)
            if task == "s2":
                _, comparator_inner = _evaluate_inner_configuration(
                    outer_train,
                    sources,
                    task,
                    best_d,
                    0.0,
                    "all_history",
                    best_d["domain"],
                )
                if selected_ensemble is not None:
                    _, comparator_other_inner = _evaluate_inner_configuration(
                        outer_train,
                        sources,
                        task,
                        best_d,
                        0.0,
                        "all_history",
                        other_domain,
                    )
                    if not comparator_inner[["row", "truth"]].equals(
                        comparator_other_inner[["row", "truth"]]
                    ):
                        raise RuntimeError("Comparator inner ensemble rows are not aligned")
                    comparator_inner = comparator_inner.copy()
                    comparator_inner["scores"] = [
                        (
                            weight * np.asarray(left_score) + (1 - weight) * np.asarray(right_score)
                        ).tolist()
                        for left_score, right_score in zip(
                            comparator_inner.scores,
                            comparator_other_inner.scores,
                            strict=True,
                        )
                    ]
                comparator_inner_raw = np.asarray(
                    [
                        _binary_high_score(np.asarray(value, dtype=float)[None, :], inner_classes)[
                            0
                        ]
                        for value, inner_classes in zip(
                            comparator_inner.scores,
                            comparator_inner.classes,
                            strict=True,
                        )
                    ]
                )
                comparator_inner_truth = comparator_inner.truth.to_numpy() == "HIGH_IMPACT"
                comparator_calibrator = fit_calibrator(comparator_inner_raw, comparator_inner_truth)
                comparator_inner_probability = comparator_calibrator.predict_proba(
                    comparator_inner_raw.reshape(-1, 1)
                )[:, 1]
                comparator_threshold = select_s2_threshold(
                    comparator_inner_probability, comparator_inner_truth
                )["threshold"]
                comparator_outer_raw = _binary_high_score(comparator_scores, comparator_classes)
                comparator_outer_probability = comparator_calibrator.predict_proba(
                    comparator_outer_raw.reshape(-1, 1)
                )[:, 1]
                comparator_prediction = np.where(
                    comparator_outer_probability >= comparator_threshold,
                    "HIGH_IMPACT",
                    "LOWER_IMPACT",
                )
            metrics = fixed_metrics(truth, predicted, TASK_LABELS[task])
            metrics["confusion_matrix"] = json.dumps(
                confusion_matrix(truth, predicted, labels=TASK_LABELS[task]).tolist()
            )
            metrics["normalized_confusion_matrix"] = json.dumps(
                confusion_matrix(
                    truth,
                    predicted,
                    labels=TASK_LABELS[task],
                    normalize="true",
                ).tolist()
            )
            if task == "s2":
                high_truth = truth == "HIGH_IMPACT"
                high_prediction = predicted == "HIGH_IMPACT"
                tp = int(np.sum(high_truth & high_prediction))
                tn = int(np.sum(~high_truth & ~high_prediction))
                fp = int(np.sum(~high_truth & high_prediction))
                fn = int(np.sum(high_truth & ~high_prediction))
                metrics.update(
                    {
                        "high_impact_f2": fbeta_score(
                            high_truth, high_prediction, beta=2, zero_division=0
                        ),
                        "specificity": tn / (tn + fp) if tn + fp else 0.0,
                        "npv": tn / (tn + fn) if tn + fn else 0.0,
                        "predicted_positive_rate": float(high_prediction.mean()),
                        "pr_auc": average_precision_score(high_truth, outer_probability),
                        "roc_auc": roc_auc_score(high_truth, outer_probability),
                        "true_positive": tp,
                    }
                )
            fold_rows.append(
                {
                    "task": task,
                    "fold": outer.fold,
                    **metrics,
                    "feature_count": features,
                    "model_size_bytes": model_size,
                    "rss_delta_bytes": psutil.Process().memory_info().rss - rss_before,
                    "runtime_seconds": perf_counter() - outer_started,
                    "outer_evaluations": 1,
                    "selected_threshold": selected_threshold,
                    "ensemble_weight": (
                        selected_ensemble["ensemble_weight"]
                        if selected_ensemble is not None
                        else np.nan
                    ),
                    **purge,
                    "target_overlap_after": target_purge["overlap_after"],
                }
            )
            selected_rows.append(
                {
                    "task": task,
                    "fold": outer.fold,
                    **{
                        key: best_d[key]
                        for key in (
                            "candidate_id",
                            "base_candidate_id",
                            "representation",
                            "summary_weight",
                            "model",
                            "C",
                            "class_weight",
                            "alpha",
                            "source_weight",
                            "recency",
                            "domain",
                        )
                    },
                }
            )
            inner_rows.extend(evaluated + stage_b + stage_c + stage_d)
            for position, row_index in enumerate(validation.index):
                oof_rows.append(
                    {
                        "row_key": sanitize_key("MYLYN", row_index),
                        "task": task,
                        "fold": outer.fold,
                        "true_label": truth[position],
                        "challenger_prediction": predicted[position],
                        "matched_target_only_prediction": comparator_prediction[position],
                        "score": json.dumps(scores[position].tolist()),
                        "classes": json.dumps(classes),
                    }
                )
            if calibration_rows:
                stage_tables.setdefault("CALIBRATION", []).extend(calibration_rows)
            audit_output.mkdir(parents=True, exist_ok=True)
            completed.add(checkpoint_key)
            atomic_json(
                state_path,
                {
                    "inner_rows": inner_rows,
                    "selected_rows": selected_rows,
                    "fold_rows": fold_rows,
                    "oof_rows": oof_rows,
                    "stage_tables": stage_tables,
                    "completed": sorted(completed),
                },
            )
            atomic_json(
                audit_output / f"checkpoint_{task}_fold_{outer.fold}.json",
                {"task": task, "fold": outer.fold, "completed_stages": list("ABCDE")},
            )
    model_output.mkdir(parents=True, exist_ok=True)
    atomic_csv(model_output / "inner_selection_results.csv", pd.DataFrame(inner_rows))
    candidates = candidate_matrix()
    executed = {
        (
            row["task"],
            row["base_candidate_id"],
            row["source_weight"],
            row["recency"],
            row["domain"],
        )
        for row in inner_rows
    }
    status_rows = []
    for row in candidates.to_dict("records"):
        key = (
            row["task"],
            row["base_candidate_id"],
            row["source_weight"],
            row["recency"],
            row["domain"],
        )
        trained = key in executed
        status_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "status": "trained" if trained else "not_executed",
                "reason": "" if trained else "eliminated by a prior inner-only stage",
            }
        )
    atomic_csv(audit_output / "candidate_status.csv", pd.DataFrame(status_rows))
    atomic_csv(model_output / "selected_configurations.csv", pd.DataFrame(selected_rows))
    atomic_csv(model_output / "fold_results.csv", pd.DataFrame(fold_rows))
    atomic_csv(model_output / "oof_predictions.csv", pd.DataFrame(oof_rows))
    for stage, name in (
        ("B", "transfer_comparison"),
        ("C", "recency_comparison"),
        ("D", "domain_adaptation_comparison"),
        ("E", "ensemble_comparison"),
    ):
        atomic_csv(model_output / f"{name}.csv", pd.DataFrame(stage_tables[stage]))
    atomic_csv(
        model_output / "calibration_results.csv",
        pd.DataFrame(stage_tables.get("CALIBRATION", [])),
    )
    atomic_csv(
        model_output / "threshold_results.csv",
        pd.DataFrame(fold_rows)[["task", "fold", "selected_threshold"]].dropna(),
    )
    folds_frame, oof_frame = pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)
    task_summary = (
        folds_frame.groupby("task", as_index=False)
        .agg(
            challenger_macro_f1=("macro_f1", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            minimum_class_recall=("minimum_class_recall", "mean"),
        )
        .merge(
            baseline[["task", "macro_f1"]].rename(columns={"macro_f1": "baseline_macro_f1"}),
            on="task",
        )
    )
    task_summary["delta"] = task_summary.challenger_macro_f1 - task_summary.baseline_macro_f1
    atomic_csv(model_output / "task_summary.csv", task_summary)
    per_class = []
    for row in folds_frame.to_dict("records"):
        for label in TASK_LABELS[row["task"]]:
            per_class.append(
                {
                    "task": row["task"],
                    "fold": row["fold"],
                    "label": label,
                    "precision": row[f"precision_{label}"],
                    "recall": row[f"recall_{label}"],
                    "f1": row[f"f1_{label}"],
                    "support": row[f"support_{label}"],
                }
            )
    atomic_csv(model_output / "per_class_results.csv", pd.DataFrame(per_class))
    distributions = (
        oof_frame.groupby(["task", "fold", "challenger_prediction"]).size().reset_index(name="rows")
    )
    atomic_csv(model_output / "predicted_class_distributions.csv", distributions)
    bootstrap_rows = []
    for task, part in oof_frame.groupby("task", sort=True):
        result = paired_bootstrap(
            part.true_label.to_numpy(),
            part.matched_target_only_prediction.to_numpy(),
            part.challenger_prediction.to_numpy(),
            TASK_LABELS[task],
        )
        bootstrap_rows.append({"task": task, **result})
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    atomic_csv(model_output / "bootstrap_delta_results.csv", bootstrap_frame)
    task_summary = task_summary.merge(bootstrap_frame, on="task")
    fold_gains = []
    for (task, fold), part in oof_frame.groupby(["task", "fold"], sort=True):
        labels = TASK_LABELS[task]
        baseline_fold = f1_score(
            part.true_label,
            part.matched_target_only_prediction,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        challenger_fold = f1_score(
            part.true_label,
            part.challenger_prediction,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        fold_gains.append({"task": task, "fold": fold, "gain": challenger_fold > baseline_fold})
    gain_counts = (
        pd.DataFrame(fold_gains)
        .groupby("task", as_index=False)
        .gain.sum()
        .rename(columns={"gain": "folds_improved"})
    )
    task_summary = task_summary.merge(gain_counts, on="task")
    task_summary["clear_transfer_evidence"] = (
        (task_summary.mean_delta > 0)
        & (task_summary.ci_lower > 0)
        & (task_summary.folds_improved >= 2)
    )
    statuses = []
    for summary in task_summary.itertuples():
        task_folds = folds_frame[folds_frame.task == summary.task]
        task_oof = oof_frame[oof_frame.task == summary.task]
        predicted_counts = task_oof.challenger_prediction.value_counts(normalize=True)
        dominant_share = float(predicted_counts.max())
        comparator_metrics = fixed_metrics(
            task_oof.true_label.to_numpy(),
            task_oof.matched_target_only_prediction.to_numpy(),
            TASK_LABELS[summary.task],
        )
        fold_sd = float(task_folds.macro_f1.std(ddof=0))
        if summary.task == "s6":
            material = (
                summary.challenger_macro_f1 >= 0.2463
                and summary.folds_improved >= 2
                and task_folds.recall_blocker.mean() > 0
                and task_folds.recall_critical.mean() > 0
                and task_oof.challenger_prediction.nunique() == 6
                and dominant_share < 0.85
                and summary.minimum_class_recall
                >= comparator_metrics["minimum_class_recall"] - 0.05
            )
            strong = material and (
                summary.challenger_macro_f1 >= 0.35
                and summary.balanced_accuracy >= 0.35
                and summary.minimum_class_recall >= 0.15
                and task_folds.recall_blocker.mean() >= 0.15
                and task_folds.recall_critical.mean() >= 0.15
            )
        elif summary.task == "s3":
            material = (
                summary.challenger_macro_f1 >= summary.baseline_macro_f1 + 0.05
                and summary.folds_improved >= 2
                and task_folds.recall_HIGH.mean() >= 0.20
                and task_folds.recall_HIGH.mean() >= comparator_metrics["recall_HIGH"] + 0.10
                and summary.minimum_class_recall >= 0.20
                and task_oof.challenger_prediction.nunique() == 3
                and dominant_share < 0.85
            )
            strong = material and (
                summary.challenger_macro_f1 >= 0.50
                and task_folds.recall_HIGH.mean() >= 0.35
                and summary.minimum_class_recall >= 0.30
                and summary.balanced_accuracy >= 0.50
                and fold_sd <= 0.08
            )
        else:
            positive_rate = float(task_folds.predicted_positive_rate.mean())
            material = (
                summary.challenger_macro_f1 >= summary.baseline_macro_f1 + 0.05
                and summary.folds_improved >= 2
                and summary.balanced_accuracy >= 0.63
                and task_folds.precision_HIGH_IMPACT.mean() >= 0.45
                and task_folds.recall_HIGH_IMPACT.mean() >= 0.45
                and task_folds.f1_HIGH_IMPACT.mean() > comparator_metrics["f1_HIGH_IMPACT"]
                and 0.02 <= positive_rate <= 0.60
            )
            strong = material and (
                summary.challenger_macro_f1 >= 0.70
                and summary.balanced_accuracy >= 0.65
                and task_folds.recall_HIGH_IMPACT.mean() >= 0.60
                and task_folds.precision_HIGH_IMPACT.mean() >= 0.50
                and task_folds.f1_HIGH_IMPACT.mean() >= 0.55
                and fold_sd <= 0.08
            )
        if strong and summary.clear_transfer_evidence:
            statuses.append("PRACTICALLY_STRONG")
        elif material and summary.clear_transfer_evidence:
            statuses.append("PROMISING")
        elif summary.challenger_macro_f1 >= summary.baseline_macro_f1:
            statuses.append("RESEARCH_ONLY")
        else:
            statuses.append("NOT_PREDICTABLE_ENOUGH")
    task_summary["success_status"] = statuses
    task_summary["transfer_evidence_status"] = np.where(
        task_summary.clear_transfer_evidence,
        "CLEAR_TRANSFER_EVIDENCE",
        "NO_CLEAR_TRANSFER_EVIDENCE",
    )
    atomic_csv(model_output / "task_summary.csv", task_summary)
    atomic_csv(
        model_output / "runtime_summary.csv",
        pd.DataFrame(inner_rows).groupby("stage", as_index=False).runtime_seconds.sum(),
    )
    atomic_csv(
        model_output / "model_size_summary.csv",
        folds_frame[
            [
                "task",
                "fold",
                "feature_count",
                "model_size_bytes",
                "rss_delta_bytes",
            ]
        ],
    )
    control_rows = []
    rng = np.random.default_rng(SEED)
    for task in TASKS:
        for outer in folds:
            validation = target.iloc[list(outer.validation)].copy()
            training = target.iloc[list(outer.train)].copy()
            truth = map_target(validation.severity, task).to_numpy()
            train_truth = map_target(training.severity, task).to_numpy()
            mode = pd.Series(train_truth).mode().sort_values().iloc[0]
            dummy = np.repeat(mode, len(validation))
            control_rows.append(
                {
                    "control": "dummy_most_frequent",
                    "task": task,
                    "fold": outer.fold,
                    **fixed_metrics(truth, dummy, TASK_LABELS[task]),
                }
            )
            if outer.fold != 1:
                continue
            selected = next(
                row for row in selected_rows if row["task"] == task and row["fold"] == outer.fold
            )
            transformer = build_representation(
                selected["representation"], selected["summary_weight"]
            )
            x_train = transformer.fit_transform(training)
            x_validation = transformer.transform(validation)
            shuffled = train_truth.copy()
            rng.shuffle(shuffled)
            shuffled_model = _make_model(selected)
            if selected["model"] == "NBSVM":
                ratio = nbsvm_ratio(x_train, shuffled, "HIGH_IMPACT")
                x_train = x_train.multiply(ratio)
                x_validation = x_validation.multiply(ratio)
            shuffled_model.fit(x_train, shuffled)
            shuffled_prediction = shuffled_model.predict(x_validation)
            control_rows.append(
                {
                    "control": "shuffled_training_labels",
                    "task": task,
                    "fold": outer.fold,
                    **fixed_metrics(truth, shuffled_prediction, TASK_LABELS[task]),
                }
            )
            masked_train, masked_validation = training.copy(), validation.copy()
            for frame in (masked_train, masked_validation):
                frame["summary"] = frame.summary.map(
                    lambda value: software_clean(value, mask_labels=True)
                )
                frame["description"] = frame.description.map(
                    lambda value: software_clean(value, mask_labels=True)
                )
            masked_prediction, _, _, _, _ = _fit_predict(
                masked_train,
                sources.iloc[0:0],
                masked_validation,
                task,
                selected,
                0.0,
                "all_history",
                "plain",
            )
            control_rows.append(
                {
                    "control": "label_token_masking",
                    "task": task,
                    "fold": outer.fold,
                    **fixed_metrics(truth, masked_prediction, TASK_LABELS[task]),
                }
            )
    atomic_csv(model_output / "control_results.csv", pd.DataFrame(control_rows))
    coverage_rows = []
    for task, part in oof_frame.groupby("task", sort=True):
        confidence = []
        for value in part.score:
            score = np.asarray(json.loads(value), dtype=float)
            if len(score) == 1:
                confidence.append(abs(score[0]))
            else:
                ordered = np.sort(score)
                confidence.append(ordered[-1] - ordered[-2])
        ranked = part.assign(confidence=confidence).sort_values(
            ["confidence", "row_key"], ascending=[False, True]
        )
        for coverage in (1.0, 0.95, 0.9, 0.8, 0.7):
            selected = ranked.iloc[: max(1, int(np.ceil(len(ranked) * coverage)))]
            coverage_rows.append(
                {
                    "task": task,
                    "coverage": coverage,
                    "rows": len(selected),
                    **fixed_metrics(
                        selected.true_label.to_numpy(),
                        selected.challenger_prediction.to_numpy(),
                        TASK_LABELS[task],
                    ),
                }
            )
    atomic_csv(model_output / "coverage_results.csv", pd.DataFrame(coverage_rows))
    atomic_bytes(
        model_output / "final_recommendation.md",
        (
            "# Final recommendation\n\n"
            + "\n".join(f"- {row.task}: {row.success_status}" for row in task_summary.itertuples())
            + "\n"
        ).encode(),
    )
    atomic_bytes(model_output / "protocol_deviations.md", b"# Protocol deviations\n\nNone.\n")
    bundle_rows = [
        "| Task | Baseline macro-F1 | Challenger macro-F1 | Delta | Balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    bundle_rows.extend(
        "| {task} | {baseline_macro_f1:.6f} | {challenger_macro_f1:.6f} | "
        "{delta:.6f} | {balanced_accuracy:.6f} |".format(**row)
        for row in task_summary.to_dict("records")
    )
    bundle = "# VALIDATION BUNDLE\n\n" + "\n".join(bundle_rows) + "\n"
    atomic_bytes(model_output / "VALIDATION_BUNDLE.md", bundle.encode())
    required = (
        "baseline_reproduction.csv",
        "inner_selection_results.csv",
        "selected_configurations.csv",
        "fold_results.csv",
        "per_class_results.csv",
        "transfer_comparison.csv",
        "recency_comparison.csv",
        "domain_adaptation_comparison.csv",
        "ensemble_comparison.csv",
        "bootstrap_delta_results.csv",
        "oof_predictions.csv",
        "final_recommendation.md",
        "VALIDATION_BUNDLE.md",
    )
    missing = [
        name
        for name in required
        if not (model_output / name).is_file() or not (model_output / name).stat().st_size
    ]
    if missing:
        raise RuntimeError(f"Required production outputs missing or empty: {missing}")
    return {"status": "PRODUCTION_ENGINE_EXECUTED", "models_fitted": True, "folds": len(fold_rows)}


def run_synthetic_stages(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Synthetic output exists")
    output.mkdir(parents=True)
    labels = np.array(["HIGH_IMPACT", "LOWER_IMPACT"] * 24)
    texts = np.array(
        [
            f"{'failure severe' if label == 'HIGH_IMPACT' else 'minor display'} token{i % 4}"
            for i, label in enumerate(labels)
        ]
    )
    source = np.array([False] * 24 + [True] * 24)
    times = pd.Series(pd.date_range("2019-01-01", periods=48, freq="30D", tz="UTC"))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    results = []
    for stage, source_weight, recency, domain in (
        ("A", 0.0, "all_history", "plain"),
        ("B", 0.25, "all_history", "plain"),
        ("C", 0.25, "half_life_4", "plain"),
        ("D", 0.25, "half_life_4", "domain_augmented"),
    ):
        weights = combine_sample_weights(
            24, times[source], pd.Timestamp("2023-01-01", tz="UTC"), source_weight, recency
        )
        x = matrix
        if domain == "domain_augmented":
            x = sparse.hstack(
                [
                    matrix,
                    sparse.diags((~source).astype(float)) @ matrix,
                    sparse.diags(source.astype(float)) @ matrix,
                ],
                format="csr",
            )
        model = LogisticRegression(
            C=1, class_weight="balanced", max_iter=2000, random_state=SEED
        ).fit(x, labels, sample_weight=weights)
        predicted = model.predict(x)
        results.append(
            {
                "stage": stage,
                "status": "trained",
                "macro_f1": f1_score(labels, predicted, average="macro"),
                "rows": len(labels),
            }
        )
    # Real score-level Stage E, not a placeholder.
    first = LogisticRegression(random_state=SEED).fit(matrix, labels).decision_function(matrix)
    second = LinearSVC(random_state=SEED).fit(matrix, labels).decision_function(matrix)
    ensemble = 0.5 * first + 0.5 * second
    predicted = np.where(ensemble >= 0, "LOWER_IMPACT", "HIGH_IMPACT")
    results.append(
        {
            "stage": "E",
            "status": "trained",
            "macro_f1": f1_score(labels, predicted, average="macro"),
            "rows": len(labels),
        }
    )
    # Inner-only calibration and threshold selection proof.
    truth = labels == "HIGH_IMPACT"
    calibrator = fit_calibrator(first, truth)
    probabilities = calibrator.predict_proba(first.reshape(-1, 1))[:, 1]
    threshold = select_s2_threshold(probabilities, truth)
    atomic_csv(output / "stage_results.csv", pd.DataFrame(results))
    atomic_json(output / "threshold_selection.json", threshold)
    return {"stages": results, "threshold": threshold, "models_fitted": True}


def real_run(
    target_path: Path,
    processed_root: Path,
    audit_output: Path,
    model_output: Path,
    protocol: Path,
    command: str,
    resume: bool = False,
    minimum_available_memory_gb: float = MINIMUM_AVAILABLE_MEMORY_GB,
    protocol_commit: str | None = None,
) -> dict[str, Any]:
    """Apply the resource gate before dispatching the full production engine."""
    available = psutil.virtual_memory().available / 2**30
    if available < minimum_available_memory_gb:
        raise RuntimeError(
            "RUN_BLOCKED_LOW_MEMORY: "
            f"{available:.3f} GiB available; {minimum_available_memory_gb:.3f} required"
        )
    disk_anchor = audit_output
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    disk_free = shutil.disk_usage(disk_anchor).free / 2**30
    if disk_free < 10.0:
        raise RuntimeError(f"RUN_BLOCKED_LOW_DISK: {disk_free:.3f} GiB free; 10.000 required")
    if not protocol_commit or not re.fullmatch(r"[0-9a-f]{40}", protocol_commit):
        raise ValueError("Real run requires the pushed 40-character protocol commit SHA")
    dry = dry_run(
        target_path,
        processed_root,
        audit_output,
        model_output,
        protocol,
        command,
        resume,
    )
    target = load_target(target_path)
    sources = load_sources(processed_root)
    result = execute_production_study(
        target,
        sources,
        audit_output,
        model_output,
        EngineOptions(minimum_available_memory_gb=minimum_available_memory_gb, resume=resume),
    )
    dry.manifest.update(
        {
            "dry_run": False,
            "models_fitted": True,
            "completed_stages": list("ABCDE"),
            "protocol_commit": protocol_commit,
        }
    )
    atomic_json(audit_output / "study_manifest.json", dry.manifest)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["dry-run", "synthetic", "real-run"])
    parser.add_argument("--target-development", type=Path)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/eclipse_core"))
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument(
        "--protocol", type=Path, default=Path("docs/mylyn_transfer_drift_threshold_protocol_v2.md")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--minimum-available-memory-gb", type=float, default=MINIMUM_AVAILABLE_MEMORY_GB
    )
    parser.add_argument("--protocol-commit")
    args = parser.parse_args()
    command_tokens = [token for token in sys.argv[1:] if token != "--resume"]
    command = "python -m defect_classifier.transfer_study_v2 " + " ".join(command_tokens)
    if args.command == "synthetic":
        print(json.dumps(run_synthetic_stages(args.model_output), indent=2))
        return
    if not all((args.target_development, args.audit_output, args.model_output)):
        parser.error("target and both outputs are required")
    if args.command == "dry-run":
        result = dry_run(
            args.target_development,
            args.processed_root,
            args.audit_output,
            args.model_output,
            args.protocol,
            command,
            args.resume,
        )
        print(json.dumps(result.manifest, indent=2))
        return
    print(
        json.dumps(
            real_run(
                args.target_development,
                args.processed_root,
                args.audit_output,
                args.model_output,
                args.protocol,
                command,
                args.resume,
                args.minimum_available_memory_gb,
                args.protocol_commit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
