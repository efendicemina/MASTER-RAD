"""Transfer + drift + threshold study (development-only; dry-run capability).

This module implements a safe, read-only dry-run entrypoint for the new
MYLYN transfer/drift/threshold study. It enforces dataset immutability checks,
fold fingerprint verification, and generates audit outputs during --dry-run.

It includes resume/manifest validation and atomic writing for audit artifacts.
Dry-run is real (reads processed parquet metadata) but does not fit models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from . import development_study
from .utils import ensure_directory, sha256_file

# Approved constants (from the research brief)
APPROVED_DEVELOPMENT_SHA256 = "8dab3df33aa8bbc64afa844d5cddabb6fa45899dc7064f38cee292d326aa05e5"
APPROVED_OUTER_FOLD_FINGERPRINT = (
    "5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903"
)
APPROVED_DEVELOPMENT_ROWS = 7664
EXPECTED_SOURCE_PROJECT_COUNT = 8


def _atomic_write_json(path: Path, obj: Any) -> None:
    ensure_directory(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        encoding="utf-8",
        dir=str(path.parent),
    ) as tf:
        json.dump(obj, tf, indent=2)
        tmp = Path(tf.name)
    os.replace(tmp, path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_directory(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        encoding="utf-8",
        dir=str(path.parent),
    ) as tf:
        df.to_csv(tf.name, index=False)
        tmp = Path(tf.name)
    os.replace(tmp, path)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_development_candidates(reports_root: Path) -> list[Path]:
    candidates: list[Path] = []
    if not reports_root.exists():
        return candidates
    for path in reports_root.glob("**/tables/development_split.csv"):
        candidates.append(path)
    return candidates


def select_development_by_sha(candidates: Iterable[Path], expected_sha: str) -> Path | None:
    for candidate in candidates:
        try:
            if sha256_file(candidate).lower() == expected_sha.lower():
                return candidate
        except Exception:
            continue
    return None


def verify_development_file(path: Path, expected_sha: str) -> None:
    # Explicitly reject test_split.csv
    if path.name == "test_split.csv":
        raise ValueError("Input path points to test_split.csv which is forbidden for this study")
    if path.name != "development_split.csv":
        raise ValueError(
            "Transfer study accepts only development_split.csv as "
            "target development input"
        )
    actual = sha256_file(path)
    if actual.lower() != expected_sha.lower():
        raise ValueError(
            f"Development split SHA mismatch: expected {expected_sha} actual {actual}"
        )


def compute_outer_fingerprint(frame: pd.DataFrame) -> str:
    folds = development_study.freeze_temporal_folds(frame, n_splits=3)
    return development_study.folds_fingerprint(folds)


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # collapse whitespace and lowercase
    return re.sub(r"\s+", " ", s).strip().lower()


def _software_tokenize(text: str) -> list[str]:
    """Lightweight software-aware tokenizer:
    - neutralizes URLs, emails, long hex and bug IDs
    - splits camelCase, snake_case and dotted.identifiers
    - preserves technical tokens and short hex-like tokens
    """
    if not isinstance(text, str) or not text:
        return []

    # neutralize URLs and emails and long hex (>8) to placeholders
    text = re.sub(r"https?://\S+", "URL", text)
    text = re.sub(r"\S+@\S+", "EMAIL", text)
    text = re.sub(r"\b0x[0-9a-fA-F]{8,}\b", "HEX", text)
    text = re.sub(r"\b[0-9a-fA-F]{8,}\b", "HEX", text)

    # split on non-word but keep dots for dotted identifiers
    parts = re.split(r"([^\w\.])", text)
    tokens: list[str] = []
    for p in parts:
        if not p or re.match(r"\W+", p):
            continue
        # split dotted
        for sub in p.split('.'):
            # split snake
            for tok in sub.split('_'):
                # split camelCase
                camel_tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", tok)
                if camel_tokens:
                    tokens.extend(camel_tokens)
                else:
                    tokens.append(tok)
    # final clean
    return [t for t in (x for x in tokens if x)][:500]


def _process_processed_root(processed_root: Path) -> pd.DataFrame:
    """Scan processed_root for parquet files and extract per-project metadata.

    Returns a DataFrame with columns:
      - project, rows, min_creation_time, max_creation_time, classes_present
    """
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root {processed_root} does not exist")

    parquet_files = list(processed_root.glob("**/*.parquet"))
    projects = {}

    for pq in parquet_files:
        try:
            # read minimal columns if present
            cols = [c.lower() for c in pd.read_parquet(pq, engine="pyarrow", columns=None).columns]
        except Exception:
            # fallback: skip unreadable
            continue
        # attempt to read only needed columns to compute counts and timestamp bounds
        need_cols = []
        # choose a sensible project column name if available
        project_col = None
        for candidate in ("source_project", "project", "product"):
            if candidate in cols:
                project_col = candidate
                break
        if project_col:
            need_cols.append(project_col)
        if "creation_time" in cols:
            need_cols.append("creation_time")
        if "severity" in cols:
            need_cols.append("severity")
        if not need_cols:
            continue
        try:
            df = pd.read_parquet(pq, engine="pyarrow", columns=need_cols)
        except Exception:
            continue
        # normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        # find which column we have for project
        proj_col = None
        for candidate in ("source_project", "project", "product"):
            if candidate in [c.lower() for c in df.columns]:
                proj_col = candidate if candidate in df.columns else candidate
                # map to actual existing lowercase column name
                break
        if proj_col is None:
            continue
        for proj, g in df.groupby(proj_col):
            if proj not in projects:
                projects[proj] = {"rows": 0, "min_ct": None, "max_ct": None, "classes": set()}
            projects[proj]["rows"] += len(g)
            if "creation_time" in g:
                try:
                    min_ct = pd.to_datetime(g["creation_time"]).min()
                    max_ct = pd.to_datetime(g["creation_time"]).max()
                except Exception:
                    min_ct = None
                    max_ct = None
                if min_ct is not None and (
                    projects[proj]["min_ct"] is None or min_ct < projects[proj]["min_ct"]
                ):
                    projects[proj]["min_ct"] = min_ct
                if max_ct is not None and (
                    projects[proj]["max_ct"] is None or max_ct > projects[proj]["max_ct"]
                ):
                    projects[proj]["max_ct"] = max_ct
            if "severity" in g:
                projects[proj]["classes"].update(set(g["severity"].astype(str).unique()))

    rows = []
    for proj, meta in projects.items():
        rows.append(
            {
                "project": proj,
                "rows": meta["rows"],
                "min_creation_time": str(meta["min_ct"]) if meta["min_ct"] is not None else None,
                "max_creation_time": str(meta["max_ct"]) if meta["max_ct"] is not None else None,
                "classes": ",".join(sorted(meta["classes"])) if meta["classes"] else "",
            }
        )
    return pd.DataFrame(rows)


def _write_minimal_dry_run_outputs(
    frame: pd.DataFrame, dev_path: Path, output: Path, processed_root: Path
) -> None:
    ensure_directory(output)
    dataset_sha = sha256_file(dev_path)
    manifest = {
        "development_path": str(dev_path),
        "dataset_sha256": dataset_sha,
        "development_rows": int(len(frame)),
        "outer_fold_fingerprint": compute_outer_fingerprint(frame),
        "dry_run": True,
    }
    _atomic_write_json(output / "study_manifest.json", manifest)

    # candidate matrix (small sanity grid)
    candidates = [
        {
            "candidate_id": "R0_logreg",
            "input_variant": "summary_description",
            "feature_variant": "word_1_2",
            "model": "LogisticRegression",
        },
        {
            "candidate_id": "R1_logreg",
            "input_variant": "summary_description",
            "feature_variant": "word_char",
            "model": "LogisticRegression",
        },
    ]
    df_cand = pd.DataFrame(candidates)
    _atomic_write_csv(output / "candidate_matrix.csv", df_cand)

    # fold identity
    folds = development_study.freeze_temporal_folds(frame, n_splits=3)
    rows = []
    for fold in folds:
        train_df = frame.iloc[list(fold.train)]
        val_df = frame.iloc[list(fold.validation)]
        rows.append(
            {
                "fold": fold.fold,
                "train_size": len(fold.train),
                "validation_size": len(fold.validation),
                "train_min_creation_time": str(pd.to_datetime(train_df.creation_time).min()),
                "train_max_creation_time": str(pd.to_datetime(train_df.creation_time).max()),
                "validation_min_creation_time": str(pd.to_datetime(val_df.creation_time).min()),
                "validation_max_creation_time": str(pd.to_datetime(val_df.creation_time).max()),
            }
        )
    _atomic_write_csv(output / "fold_identity.csv", pd.DataFrame(rows))

    # source dataset manifest (read processed_root metadata)
    src_df = _process_processed_root(processed_root)
    # ensure MYLYN not present
    non_mylyn = src_df[~src_df["project"].str.lower().eq("mylyn")]
    if len(non_mylyn) < EXPECTED_SOURCE_PROJECT_COUNT:
        raise RuntimeError(
            "Missing required non-MYLYN processed source projects. Found: "
            f"{sorted(list(src_df['project']))}"
        )
    _atomic_write_csv(output / "source_dataset_manifest.csv", src_df)

    # data cutoff proof
    proof = pd.DataFrame(
        [
            {
                "min_creation_time": str(frame.creation_time.min()),
                "max_creation_time": str(frame.creation_time.max()),
                "development_rows": int(len(frame)),
            }
        ]
    )
    _atomic_write_csv(output / "data_cutoff_proof.csv", proof)

    # duplicate purge results (exact and normalized)
    frame = frame.copy()
    frame["concat_text"] = (frame["summary"].fillna("") + " "+ frame["description"].fillna(""))
    frame["exact_dup_count"] = frame.duplicated(subset=["concat_text"]).astype(int)
    frame["normalized_text"] = frame["concat_text"].map(_normalize_text)
    frame["normalized_dup_count"] = frame.duplicated(subset=["normalized_text"]).astype(int)
    dup_stats = {
        "total_rows": len(frame),
        "exact_duplicates": int(frame['exact_dup_count'].sum()),
        "normalized_duplicates": int(frame['normalized_dup_count'].sum()),
    }
    _atomic_write_json(output / "duplicate_purge_results.json", dup_stats)

    # VALIDATION_BUNDLE.md (DRY_RUN)
    lines = [
        "# VALIDATION_BUNDLE (DRY_RUN)",
        "NOTE: This is a dry-run artifact and contains no raw text or private identifiers.",
        f"development_sha256: {dataset_sha}",
        f"development_rows: {len(frame)}",
        f"outer_fold_fingerprint: {manifest['outer_fold_fingerprint']}",
        f"source_projects_count: {len(src_df)}",
    ]
    (output / "VALIDATION_BUNDLE.md").write_text("\n".join(lines), encoding="utf-8")


def _run_dry_run(
    dev_path: Path,
    processed_root: Path,
    audit_output: Path,
    model_output: Path,
    resume: bool = False,
) -> None:
    # fail-closed if audit_output already exists unless resume=True
    if audit_output.exists() and not resume:
        raise RuntimeError(
            "Audit output directory already exists. Use --resume to continue an existing run."
        )

    # locate development file if a reports directory root was supplied
    if dev_path.is_dir():
        candidates = find_development_candidates(dev_path)
        selected = select_development_by_sha(candidates, APPROVED_DEVELOPMENT_SHA256)
        if selected is None:
            raise FileNotFoundError(
                "No development_split.csv with the approved SHA-256 was found "
                "under the provided reports root"
            )
        dev_path = selected

    verify_development_file(dev_path, APPROVED_DEVELOPMENT_SHA256)

    frame = development_study.load_development_only(dev_path)

    if len(frame) != APPROVED_DEVELOPMENT_ROWS:
        raise ValueError(
            f"Development row count mismatch: expected {APPROVED_DEVELOPMENT_ROWS} "
            f"got {len(frame)}"
        )

    fingerprint = compute_outer_fingerprint(frame)
    if fingerprint != APPROVED_OUTER_FOLD_FINGERPRINT:
        raise ValueError(
            "Outer fold fingerprint mismatch: expected "
            f"{APPROVED_OUTER_FOLD_FINGERPRINT} got {fingerprint}"
        )

    # resume manifest validation
    manifest_path = audit_output / "study_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise RuntimeError("Cannot resume: study_manifest.json not found in audit output")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("dataset_sha256") != sha256_file(dev_path):
            raise RuntimeError("Resume manifest mismatch: dataset SHA does not match")
        if existing.get("outer_fold_fingerprint") != fingerprint:
            raise RuntimeError("Resume manifest mismatch: outer fold fingerprint does not match")

    # produce audit outputs (atomic)
    _write_minimal_dry_run_outputs(frame, dev_path, audit_output, processed_root)
    # model_output reserved for artifacts (dry-run should create only directory)
    ensure_directory(model_output)


def _candidate_matrix_checksum(audit_output: Path) -> str | None:
    path = audit_output / "candidate_matrix.csv"
    if not path.exists():
        return None
    return _hash_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="defect_classifier.transfer_study")
    parser.add_argument("--target-development", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    # safety: forbid direct paths that look like test outputs
    if "test_split.csv" in str(args.target_development).lower():
        raise ValueError("test_split.csv is explicitly forbidden as input for transfer_study")

    if args.dry_run:
        _run_dry_run(
            args.target_development,
            args.processed_root,
            args.audit_output,
            args.model_output,
            resume=args.resume,
        )
        print(json.dumps({"status": "dry_run_completed"}, indent=2))
        return 0

    # Full run is not yet implemented; the implementation must follow the protocol and
    # is memory-bounded. For now, prevent accidental runs.
    raise NotImplementedError(
        "Full transfer study run is not implemented in this commit. "
        "Use --dry-run to validate the protocol and manifest."
    )


if __name__ == "__main__":
    raise SystemExit(main())
