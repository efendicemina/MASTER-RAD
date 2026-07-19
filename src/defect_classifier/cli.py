"""Command-line interface for the defect classifier package."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .data_loading import load_raw_csv, resolve_columns
from .data_validation import inspect_dataset
from .dataset_audits import generate_dataset_audits, write_manifest_csv
from .dataset_builder import build_dataset, validate_project_output
from .dataset_schema import audit_schemas
from .final_reports import create_final_reports
from .readiness import validate_training_readiness
from .reporting import save_frame, save_payload
from .training import evaluate_experiment, prepare_dataframe, train_experiment
from .utils import ensure_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="defect_classifier", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_config_command(subparsers, "inspect-data")
    _add_config_command(subparsers, "prepare-data")
    _add_config_command(subparsers, "train")
    _add_config_command(subparsers, "evaluate")
    _add_config_command(subparsers, "run-all")

    audit_schema = subparsers.add_parser("audit-schema", help="Audit CSV headers only")
    audit_schema.add_argument("--config", required=True)
    audit_schema.add_argument("--project")

    build_dataset_parser = subparsers.add_parser(
        "build-dataset", help="Build reduced project Parquet datasets"
    )
    build_dataset_parser.add_argument("--config", required=True)
    build_dataset_parser.add_argument("--force", action="store_true")
    build_dataset_parser.add_argument("--project")
    build_dataset_parser.add_argument("--chunksize", type=int)

    readiness = subparsers.add_parser(
        "validate-training-readiness", help="Audit feasibility without model training"
    )
    readiness.add_argument("--config", required=True)
    readiness.add_argument("--project")

    validate_project = subparsers.add_parser(
        "validate-project-output", help="Validate a processed project Parquet and manifest"
    )
    validate_project.add_argument("--config", required=True)
    validate_project.add_argument("--project", required=True)

    finalize = subparsers.add_parser(
        "finalize-pretraining", help="Create final cross-project reports without training"
    )
    finalize.add_argument("--config", required=True)

    predict = subparsers.add_parser("predict", help="Predict a severity label for one report")
    predict.add_argument("--model", required=True, help="Path to a fitted joblib pipeline")
    predict.add_argument("--summary", required=True, help="Bug report summary")
    predict.add_argument("--description", required=True, help="Bug report description")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-data":
            return _inspect_data(args.config)
        if args.command == "prepare-data":
            return _prepare_data(args.config)
        if args.command == "train":
            return _train(args.config)
        if args.command == "evaluate":
            return _evaluate(args.config)
        if args.command == "run-all":
            outcome = train_experiment(args.config)
            evaluate_experiment(outcome.experiment_dir, args.config)
            return 0
        if args.command == "audit-schema":
            return _audit_schema(args.config, args.project)
        if args.command == "build-dataset":
            return _build_dataset(
                args.config, force=args.force, project=args.project, chunksize=args.chunksize
            )
        if args.command == "validate-training-readiness":
            return _validate_readiness(args.config, args.project)
        if args.command == "validate-project-output":
            return _validate_project_output(args.config, args.project)
        if args.command == "finalize-pretraining":
            return _finalize_pretraining(args.config)
        if args.command == "predict":
            return _predict(args.model, args.summary, args.description)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


def _add_config_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser], name: str
) -> None:
    parser = subparsers.add_parser(name, help=f"Run the {name} workflow")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")


def _inspect_data(config_path: str) -> int:
    config = load_config(config_path)
    dataset_paths = config["dataset"]["resolved_paths"]
    if len(dataset_paths) > 1 or Path(dataset_paths[0]).stat().st_size > 100 * 1024 * 1024:
        outputs = generate_dataset_audits(config)
        print(f"Large-dataset inspection complete: {len(outputs)} audit outputs")
        return 0
    frame = load_raw_csv(config["dataset"]["path"])
    resolved_frame, _ = resolve_columns(frame, config["source_columns"])
    inspection = inspect_dataset(resolved_frame, config)

    output_root = Path(config["output"]["root_dir"])
    metrics_dir = ensure_directory(output_root / "metrics")
    tables_dir = ensure_directory(output_root / "tables")

    save_payload(metrics_dir / "data_inspection.json", inspection.summary)
    save_frame(tables_dir / "severity_distribution.csv", inspection.tables["severity_counts"])
    save_frame(tables_dir / "missing_values.csv", inspection.tables["missing_values"])
    save_frame(tables_dir / "filtering_summary.csv", inspection.tables["filtering_summary"])

    print("Inspection summary")
    print(f"- Resolved dataset path: {inspection.summary['resolved_dataset_path']}")
    print(f"- Total rows: {inspection.summary['total_rows']}")
    columns = ", ".join(inspection.summary["detected_columns"])
    print(f"- Detected columns: {columns}")

    severity_parts = [
        (
            f"{entry[config['target_column']]}={int(entry['count'])} "
            f"({float(entry['percentage']) * 100:.1f}%)"
        )
        for entry in inspection.summary["severity_class_counts"]
    ]
    print(f"- Severity counts: {', '.join(severity_parts)}")
    print(f"- Missing Summary count: {inspection.summary['missing_summary_count']}")
    print(f"- Missing Description count: {inspection.summary['missing_description_count']}")
    print(f"- Missing Severity count: {inspection.summary['missing_severity_count']}")
    print(f"- Duplicate count: {inspection.summary['duplicate_count']}")

    creation_range = inspection.summary.get("creation_date_range") or {}
    if creation_range.get("earliest") and creation_range.get("latest"):
        print(f"- Creation date range: {creation_range['earliest']} -> {creation_range['latest']}")
    else:
        print("- Creation date range: unavailable")

    print(
        "- Proposed retained/excluded rows: "
        f"{inspection.summary['proposed_retained_rows']} retained, "
        f"{inspection.summary['proposed_excluded_rows']} excluded"
    )

    return 0


def _prepare_data(config_path: str) -> int:
    config = load_config(config_path)
    cleaned, resolved_columns, inspection = prepare_dataframe(config)
    output_dir = ensure_directory(Path(config["output"]["root_dir"]) / "prepared")
    save_frame(output_dir / "prepared_data.csv", cleaned)
    save_payload(output_dir / "inspection.json", inspection)
    save_payload(output_dir / "resolved_columns.json", resolved_columns)
    return 0


def _train(config_path: str) -> int:
    train_experiment(config_path)
    return 0


def _evaluate(config_path: str) -> int:
    config = load_config(config_path)
    experiments_root = Path(config["output"]["experiments_dir"])
    if not experiments_root.exists():
        raise FileNotFoundError("No experiment directory exists yet. Run train first.")
    latest_experiment = sorted([path for path in experiments_root.iterdir() if path.is_dir()])[-1]
    evaluate_experiment(latest_experiment, config_path)
    return 0


def _predict(model_path: str, summary: str, description: str) -> int:
    import joblib

    pipeline = joblib.load(model_path)
    frame = pd.DataFrame({"summary": [summary], "description": [description]})
    prediction = pipeline.predict(frame)[0]
    print(prediction)
    return 0


def _audit_schema(config_path: str, project: str | None) -> int:
    config = load_config(config_path)
    paths = config["dataset"]["resolved_paths"]
    if project:
        paths = [path for path in paths if Path(path).stem.lower().startswith(project.lower())]
    if not paths:
        raise ValueError(f"No configured source matched project: {project}")
    report_dir = Path(config["output"]["root_dir"]) / "dataset_audit"
    rows = audit_schemas(paths, report_dir)
    for row in rows:
        print(f"[{row.audit_status}] {row.source_project}: {row.file_name}")
    return 0 if all(row.audit_status == "PASS" for row in rows) else 2


def _build_dataset(
    config_path: str,
    force: bool,
    project: str | None,
    chunksize: int | None,
) -> int:
    if chunksize is not None and chunksize < 1:
        raise ValueError("--chunksize must be >= 1")
    config = load_config(config_path)
    results = build_dataset(config, force=force, project=project, chunksize=chunksize)
    write_manifest_csv(config)
    for result in results:
        print(
            f"[{result['resume_status']}] {result['source_project']}: "
            f"{result['output_row_count']:,} rows, "
            f"{result['output_file_size'] / (1024**2):.1f} MiB"
        )
    return 0


def _validate_readiness(config_path: str, project: str | None) -> int:
    config = load_config(config_path)
    payload = validate_training_readiness(config, project)
    print(f"Training readiness: {payload['overall_status']}")
    for row in payload["projects"]:
        print(f"[{row['status']}] {row['scope']}: {row['usable_rows']:,} usable rows")
    return 0


def _validate_project_output(config_path: str, project: str) -> int:
    import json

    config = load_config(config_path)
    core_dir = Path(config.get("processed", {}).get("core_dir", "data/processed/eclipse_core"))
    if not core_dir.is_absolute():
        core_dir = Path(config["output"]["root_dir"]).parent / core_dir
    manifest = json.loads((core_dir / "manifest.json").read_text(encoding="utf-8"))
    matches = [row for row in manifest if row["source_project"].lower() == project.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest entry for {project}, found {len(matches)}")
    entry = matches[0]
    result = validate_project_output(
        entry["output_parquet_path"],
        entry["source_project"],
        entry["source_filename"],
        int(entry["raw_row_count"]),
    )
    if int(entry["parse_error_count"]) != 0:
        raise ValueError(f"Manifest records parse errors for {project}")
    print(f"[VALID] {project}: {result['rows']:,} rows")
    return 0


def _finalize_pretraining(config_path: str) -> int:
    config = load_config(config_path)
    outputs = create_final_reports(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0
