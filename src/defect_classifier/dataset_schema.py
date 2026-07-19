"""Cheap schema discovery for very large CSV sources."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loading import normalize_column_name
from .utils import ensure_directory, write_json

CORE_ALIASES: dict[str, list[str]] = {
    "issue_url": ["issue url", "issue_url", "bug url", "bug_url"],
    "issue_id": ["id", "issue id", "issue_id", "bug id", "bug_id"],
    "product": ["product"],
    "component": ["component"],
    "summary": ["summary", "short desc", "short_desc", "title"],
    "description": ["description", "details", "body", "long_desc"],
    "severity": ["severity", "bug severity", "bug_severity", "sev"],
    "creation_time": ["creation time", "creation_time", "created at", "created_at"],
    "dupe_of": ["dupe of", "dupe_of", "duplicate of", "duplicate_of"],
}
REQUIRED_FIELDS = {"issue_id", "summary", "description", "severity", "creation_time"}


@dataclass(slots=True)
class SchemaAuditRow:
    source_path: str
    file_name: str
    source_project: str
    file_size: int
    detected_encoding: str
    sample_rows_read: int
    raw_source_columns: list[str]
    canonical_column_mappings: dict[str, str]
    missing_required_columns: list[str]
    ambiguous_mappings: dict[str, list[str]]
    audit_status: str


def derive_source_project(path: str | Path) -> str:
    name = Path(path).stem
    suffix = "_dataset_issues"
    return name[: -len(suffix)] if name.lower().endswith(suffix) else name


def read_csv_header(path: str | Path) -> tuple[list[str], str]:
    """Read only the CSV header, accepting UTF-8 and UTF-8 BOM."""

    source = Path(path)
    with source.open("rb") as binary:
        has_bom = binary.read(3) == b"\xef\xbb\xbf"
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return next(csv.reader(handle)), "utf-8-sig" if has_bom else "utf-8"
    except UnicodeDecodeError as exc:
        raise ValueError(f"Unsupported CSV encoding in {source}; expected UTF-8/BOM") from exc
    except (StopIteration, csv.Error) as exc:
        raise ValueError(f"Cannot read CSV header from {source}: {exc}") from exc


def resolve_core_columns(raw_columns: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    normalized: dict[str, list[str]] = {}
    for column in raw_columns:
        normalized.setdefault(normalize_column_name(column.lstrip("\ufeff")), []).append(column)
    mappings: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for canonical, aliases in CORE_ALIASES.items():
        candidates: list[str] = []
        for alias in [canonical, *aliases]:
            candidates.extend(normalized.get(normalize_column_name(alias), []))
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            mappings[canonical] = candidates[0]
        elif len(candidates) > 1:
            ambiguous[canonical] = candidates
    return mappings, ambiguous


def audit_schema_file(path: str | Path) -> SchemaAuditRow:
    source = Path(path)
    columns, encoding = read_csv_header(source)
    mappings, ambiguous = resolve_core_columns(columns)
    missing = sorted(REQUIRED_FIELDS - set(mappings) - set(ambiguous))
    sample_rows = 0
    if not missing and not ambiguous:
        try:
            sample_rows = len(
                pd.read_csv(
                    source,
                    usecols=list(mappings.values()),
                    nrows=2,
                    encoding="utf-8-sig",
                    on_bad_lines="error",
                )
            )
        except Exception as exc:
            raise ValueError(f"Minimal sample parse failed for {source}: {exc}") from exc
    status = "PASS" if not missing and not ambiguous else "FAIL"
    return SchemaAuditRow(
        source_path=str(source.resolve()),
        file_name=source.name,
        source_project=derive_source_project(source),
        file_size=source.stat().st_size,
        detected_encoding=encoding,
        sample_rows_read=sample_rows,
        raw_source_columns=columns,
        canonical_column_mappings=mappings,
        missing_required_columns=missing,
        ambiguous_mappings=ambiguous,
        audit_status=status,
    )


def audit_schemas(paths: list[str], report_dir: str | Path) -> list[SchemaAuditRow]:
    rows = [audit_schema_file(path) for path in paths]
    output = ensure_directory(report_dir)
    payload = [asdict(row) for row in rows]
    write_json(output / "schema_audit.json", payload)
    flat = pd.DataFrame(payload)
    for column in [
        "raw_source_columns",
        "canonical_column_mappings",
        "missing_required_columns",
        "ambiguous_mappings",
    ]:
        flat[column] = flat[column].map(str)
    flat.to_csv(output / "schema_audit.csv", index=False)
    return rows


def schema_mapping(path: str | Path) -> tuple[SchemaAuditRow, dict[str, str]]:
    audit = audit_schema_file(path)
    if audit.audit_status != "PASS":
        raise ValueError(
            f"Schema audit failed for {audit.file_name}: missing={audit.missing_required_columns}, "
            f"ambiguous={audit.ambiguous_mappings}"
        )
    return audit, audit.canonical_column_mappings


def audit_payload(row: SchemaAuditRow) -> dict[str, Any]:
    return asdict(row)
