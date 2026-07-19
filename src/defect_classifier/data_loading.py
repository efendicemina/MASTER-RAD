"""Dataset loading and column normalization."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .utils import sha256_file


def normalize_column_name(name: str) -> str:
    """Normalize a column name for fuzzy matching."""

    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    return normalized.strip("_")


def normalize_frame_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a copy of *frame* with normalized column names."""

    renamed = {column: normalize_column_name(str(column)) for column in frame.columns}
    return frame.rename(columns=renamed).copy(), renamed


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file using pandas."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def infer_column_map(frame: pd.DataFrame, source_columns: dict[str, list[str]]) -> dict[str, str]:
    """Infer canonical column names from aliases in the configuration."""

    normalized_columns = {normalize_column_name(str(column)): column for column in frame.columns}
    resolved: dict[str, str] = {}
    for canonical_name, aliases in source_columns.items():
        search_terms = [canonical_name, *aliases]
        for alias in search_terms:
            normalized_alias = normalize_column_name(str(alias))
            if normalized_alias in normalized_columns:
                resolved[canonical_name] = normalized_columns[normalized_alias]
                break
    return resolved


def resolve_columns(
    frame: pd.DataFrame, source_columns: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename known columns to canonical names where possible."""

    renamed_frame, _ = normalize_frame_columns(frame)
    resolved = infer_column_map(renamed_frame, source_columns)
    rename_map = {actual_name: canonical_name for canonical_name, actual_name in resolved.items()}
    return renamed_frame.rename(columns=rename_map).copy(), resolved


def file_fingerprint(path: str | Path) -> str:
    """Return a SHA-256 fingerprint for the input file."""

    return sha256_file(path)


def coerce_datetime_column(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Parse a datetime column in-place and return the modified frame."""

    result = frame.copy()
    result[column_name] = pd.to_datetime(result[column_name], errors="coerce", utc=True)
    return result


def ensure_text_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Guarantee that text columns exist and are string-like."""

    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)
    return result
