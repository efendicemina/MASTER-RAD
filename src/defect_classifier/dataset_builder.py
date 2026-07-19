"""Streaming construction of reduced, leakage-safe project Parquet files."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .data_validation import normalize_label
from .dataset_schema import derive_source_project, schema_mapping
from .utils import ensure_directory, sha256_file, utc_now_iso, write_json

CORE_COLUMNS = [
    "issue_url",
    "issue_id",
    "source_project",
    "source_file",
    "global_report_key",
    "product",
    "component",
    "summary",
    "description",
    "severity",
    "creation_time",
    "dupe_of",
    "has_valid_summary",
    "has_valid_description",
    "has_valid_text",
    "has_valid_severity",
    "is_excluded_label",
    "has_valid_creation_time",
    "exact_text_hash",
    "duplicate_group_id",
]
FORBIDDEN_COLUMNS = [
    "comments",
    "history_activity_log",
    "attachments",
    "creator",
    "creator_detail",
    "assigned_to",
    "assigned_to_detail",
    "cc",
    "cc_detail",
    "qa_contact",
    "priority",
    "status",
    "resolution",
]


def _configuration_fingerprint(config: dict[str, Any]) -> str:
    relevant = {
        "core_columns": CORE_COLUMNS,
        "label_exclusions": config.get("label_exclusions", []),
        "schema_version": 1,
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_hash(summary: str, description: str) -> str:
    normalized = " ".join(f"{summary}\n{description}".lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _prepare_chunk(
    raw: pd.DataFrame,
    mappings: dict[str, str],
    project: str,
    source_file: str,
    exclusions: list[str],
) -> pd.DataFrame:
    reverse = {source: canonical for canonical, source in mappings.items()}
    chunk = raw.rename(columns=reverse).copy()
    for column in ["issue_url", "product", "component", "dupe_of"]:
        if column not in chunk:
            chunk[column] = ""
    for column in [
        "issue_url",
        "issue_id",
        "product",
        "component",
        "summary",
        "description",
        "severity",
        "dupe_of",
    ]:
        chunk[column] = chunk[column].fillna("").astype(str)
    chunk["issue_id"] = chunk["issue_id"].str.strip()
    chunk["source_project"] = project
    chunk["source_file"] = source_file
    chunk["global_report_key"] = project + ":" + chunk["issue_id"]
    chunk["creation_time"] = pd.to_datetime(chunk["creation_time"], errors="coerce", utc=True)
    summary_clean = chunk["summary"].str.strip()
    description_clean = chunk["description"].str.strip()
    severity_normalized = chunk["severity"].map(normalize_label)
    chunk["has_valid_summary"] = summary_clean.ne("")
    chunk["has_valid_description"] = description_clean.ne("")
    chunk["has_valid_text"] = chunk["has_valid_summary"] | chunk["has_valid_description"]
    chunk["has_valid_severity"] = severity_normalized.ne("")
    excluded = {normalize_label(value) for value in exclusions}
    chunk["is_excluded_label"] = severity_normalized.isin(excluded)
    chunk["has_valid_creation_time"] = chunk["creation_time"].notna()
    chunk["exact_text_hash"] = [
        _text_hash(summary, description)
        for summary, description in zip(chunk["summary"], chunk["description"], strict=True)
    ]
    dupe = chunk["dupe_of"].str.strip()
    chunk["duplicate_group_id"] = chunk["global_report_key"]
    chunk.loc[dupe.ne(""), "duplicate_group_id"] = project + ":" + dupe[dupe.ne("")]
    return chunk[CORE_COLUMNS]


def validate_parquet(path: str | Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    forbidden = sorted(set(columns) & set(FORBIDDEN_COLUMNS))
    if forbidden:
        raise ValueError(f"Forbidden columns in processed Parquet: {forbidden}")
    missing = sorted(set(CORE_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Processed Parquet is missing core columns: {missing}")
    return {"rows": parquet.metadata.num_rows, "columns": columns}


def validate_project_output(
    path: str | Path,
    expected_project: str,
    expected_source_file: str,
    expected_rows: int,
) -> dict[str, Any]:
    """Validate row count, identity fields and temporal/string Arrow types."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    basic = validate_parquet(path)
    if basic["rows"] != expected_rows:
        raise ValueError(f"Parquet row mismatch: {basic['rows']} != {expected_rows}")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    issue_id_type = schema.field("issue_id").type
    if not (pa.types.is_string(issue_id_type) or pa.types.is_large_string(issue_id_type)):
        raise ValueError("issue_id must be stored as an Arrow string")
    creation_type = schema.field("creation_time").type
    if not pa.types.is_timestamp(creation_type) or creation_type.tz != "UTC":
        raise ValueError(f"creation_time must be a UTC timestamp, got {creation_type}")
    identity = pq.read_table(path, columns=["source_project", "source_file"]).to_pandas()
    projects = set(identity["source_project"].dropna().astype(str))
    source_files = set(identity["source_file"].dropna().astype(str))
    if projects != {expected_project}:
        raise ValueError(f"Unexpected source_project values: {projects}")
    if source_files != {expected_source_file}:
        raise ValueError(f"Unexpected source_file values: {source_files}")
    return {**basic, "source_project": expected_project, "source_file": expected_source_file}


def build_project(
    source_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    manifest_entry: dict[str, Any] | None = None,
    force: bool = False,
    chunksize: int | None = None,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = Path(source_path)
    source_signature = (source.stat().st_size, source.stat().st_mtime_ns)
    audit, mappings = schema_mapping(source)
    project = derive_source_project(source)
    destination_dir = ensure_directory(output_dir)
    output = destination_dir / f"{project}.parquet"
    temporary = destination_dir / f".{project}.parquet.tmp"
    source_hash = sha256_file(source)
    config_hash = _configuration_fingerprint(config)
    if not force and manifest_entry:
        unchanged = (
            manifest_entry.get("source_sha256") == source_hash
            and manifest_entry.get("configuration_fingerprint") == config_hash
            and manifest_entry.get("completion_status") == "complete"
            and output.exists()
        )
        if unchanged:
            validated = validate_parquet(output)
            if validated["rows"] == manifest_entry.get("output_row_count"):
                return {**manifest_entry, "resume_status": "skipped_unchanged"}

    if temporary.exists():
        temporary.unlink()
    read_columns = list(mappings.values())
    dtype = {mappings["issue_id"]: "string"}
    if "dupe_of" in mappings:
        dtype[mappings["dupe_of"]] = "string"
    started = perf_counter()
    writer = None
    raw_rows = 0
    severity_counts: Counter[str] = Counter()
    missing_text = 0
    invalid_dates = 0
    exact_hashes: Counter[str] = Counter()
    chunk_size = chunksize or int(config["dataset"].get("chunksize", 50000))
    try:
        iterator = pd.read_csv(
            source,
            usecols=read_columns,
            dtype=dtype,
            chunksize=chunk_size,
            encoding="utf-8-sig",
            on_bad_lines="error",
            low_memory=False,
        )
        for chunk_number, raw in enumerate(iterator, start=1):
            print(f"[{project}] chunk {chunk_number}: {len(raw):,} rows")
            processed = _prepare_chunk(
                raw,
                mappings,
                project,
                source.name,
                config.get("label_exclusions", []),
            )
            raw_rows += len(processed)
            severity_counts.update(processed["severity"].map(normalize_label))
            missing_text += int((~processed["has_valid_text"]).sum())
            invalid_dates += int((~processed["has_valid_creation_time"]).sum())
            exact_hashes.update(processed["exact_text_hash"])
            table = pa.Table.from_pandas(processed, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    except Exception as exc:
        if writer is not None:
            writer.close()
        raise ValueError(
            f"Failed processing {source.name} near input row {raw_rows + 2:,}: {exc}"
        ) from exc
    if writer is None:
        raise ValueError(f"Source CSV contains no data rows: {source}")
    writer.close()
    if (source.stat().st_size, source.stat().st_mtime_ns) != source_signature:
        raise ValueError(f"Raw source changed during processing: {source}")
    validated = validate_parquet(temporary)
    os.replace(temporary, output)
    duration = perf_counter() - started
    duplicate_rows = sum(count - 1 for count in exact_hashes.values() if count > 1)
    return {
        "source_filename": source.name,
        "source_project": project,
        "source_path": str(source.resolve()),
        "source_file_size": source.stat().st_size,
        "source_sha256": source_hash,
        "detected_encoding": audit.detected_encoding,
        "configuration_fingerprint": config_hash,
        "processing_timestamp": utc_now_iso(),
        "raw_row_count": raw_rows,
        "output_row_count": validated["rows"],
        "parse_error_count": 0,
        "output_parquet_path": str(output.resolve()),
        "output_file_size": output.stat().st_size,
        "severity_distribution": dict(sorted(severity_counts.items())),
        "missing_text_count": missing_text,
        "invalid_date_count": invalid_dates,
        "exact_duplicate_row_count": duplicate_rows,
        "processing_duration_seconds": duration,
        "completion_status": "complete",
        "resume_status": "rebuilt" if force else "built",
    }


def build_dataset(
    config: dict[str, Any],
    force: bool = False,
    project: str | None = None,
    chunksize: int | None = None,
) -> list[dict[str, Any]]:
    output_dir = Path(config.get("processed", {}).get("core_dir", "data/processed/eclipse_core"))
    if not output_dir.is_absolute():
        output_dir = Path(config["output"]["root_dir"]).parent / output_dir
    ensure_directory(output_dir)
    manifest_path = output_dir / "manifest.json"
    existing: list[dict[str, Any]] = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_project = {entry["source_project"].lower(): entry for entry in existing}
    selected = []
    for path in config["dataset"]["resolved_paths"]:
        source_project = derive_source_project(path)
        if project and source_project.lower() != project.lower():
            continue
        selected.append(path)
    if not selected:
        raise ValueError(f"No configured dataset matched project: {project}")
    results = []
    for path in selected:
        source_project = derive_source_project(path)
        print(f"Processing {source_project}: {Path(path).name}")
        result = build_project(
            path,
            output_dir,
            config,
            manifest_entry=by_project.get(source_project.lower()),
            force=force,
            chunksize=chunksize,
        )
        by_project[source_project.lower()] = result
        results.append(result)
        write_json(
            manifest_path, sorted(by_project.values(), key=lambda row: row["source_project"])
        )
    return results
