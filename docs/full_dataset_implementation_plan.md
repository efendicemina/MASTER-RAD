# Full Eclipse Dataset Implementation Plan

## Objective

Create a reproducible, resumable and leakage-aware reduced dataset from the nine Eclipse CSV
exports before any real model training. Raw CSV files are immutable inputs. Initial `Summary`
and `Description` are the only planned model features.

## Phases

1. Resolve one path, an explicit path list, or a repository-root-relative glob.
2. Audit every CSV header and reject missing or ambiguous research-column mappings.
3. Process each project independently in configurable chunks, reading only required columns.
4. Atomically create one reduced Parquet file per project and update the manifest.
5. Produce severity, text, temporal, duplicate, privacy and leakage audits.
6. Validate chronological holdout and expanding-window CV feasibility without training.
7. Pilot MYLYN, then process projects from smallest to largest, leaving JDT and Platform last.

## Disk and memory considerations

- Default ingestion chunks contain 50,000 rows and only research columns.
- Large comments, history, attachments and personal/post-report metadata are never loaded.
- Parquet uses Zstandard compression and is written project-by-project.
- Atomic `.parquet.tmp` files require temporary free space roughly equal to one reduced project.
- Streaming SHA-256 requires a complete sequential read but constant memory.
- Audits iterate Parquet batches. Exact-hash counters grow with the number of distinct reports;
  this must be monitored for the largest projects.
- No combined CSV or monolithic in-memory dataset is created.

## Risks

- malformed CSV quoting or unsupported encodings;
- schema differences or ambiguous aliases between exports;
- truncated source files and interrupted writes;
- incomplete `Dupe of` relationships and unresolved near-duplicates;
- invalid dates that make chronological evaluation infeasible;
- label drift, rare classes and project-specific severity conventions;
- insufficient disk space for Platform and JDT;
- extremely long initial descriptions increasing feature-building cost.

## Dataset-dependent decisions

- which raw severity labels belong to the multiclass target;
- whether `enhancement` is excluded for the primary defect-severity question;
- whether high/medium/low grouping is scientifically defensible;
- minimum class support thresholds;
- pooled, within-project, or cross-project evaluation;
- treatment of exact duplicate narratives with conflicting labels;
- whether a later scalable near-duplicate method is necessary.

## Expected outputs

- `reports/dataset_audit/schema_audit.{csv,json}`;
- one validated Parquet file per processed project;
- `data/processed/eclipse_core/manifest.json`;
- severity, text, temporal, duplicate and privacy audit tables;
- `training_readiness.{json,csv,md}` with PASS, WARNING or FAIL decisions.

## Acceptance criteria

- sample single-file workflow remains operational;
- single path, explicit path list and glob configurations resolve from repository root;
- all nine headers pass or fail with explicit diagnostics;
- ingestion never loads forbidden columns and uses bounded chunks;
- IDs remain strings, timestamps are UTC, and global keys are project-qualified;
- raw and output row counts match unless explicit quarantine behavior is configured;
- completed outputs are atomic, validated and resumable by hash/config fingerprint;
- CI tests use synthetic data only;
- MYLYN pilot produces a validated manifest, audits and readiness report;
- no real model is trained during dataset construction or auditing.
