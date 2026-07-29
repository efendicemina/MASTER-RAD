# Eclipse pooled v1 preparation validation

## Provenance and outcome

- Base commit: `e92eff2747586baee7480a2524e425bca25e0071`
- Frozen protocol commit: `428af8aa61ef490d50c7a5e1eb1e47f38999cf4c`
- Protocol SHA-256: `159f9c7d54a49a9cb354214708e67adbbbd4f4b987a7050c4c263e2fb0ad2400`
- Candidate checksum: `c3cc83a29b481c53e3c330e9f2bdf880fa4d96381ccf30fcba355bdf100ec784`
- Dry-run: `DRY_RUN_COMPLETE`; three development pilot fits; locked test not accessed
- Validation mode: `VALIDATION_PASS` (protocol, commit, projects, grid, partition and privacy)

## Dataset and reservation audit

“Eligible available” is the six valid S6 labels in the immutable processed manifest.
“Allowed” excludes the pre-existing latest-20% reservation for every non-MYLYN project.
MYLYN uses only its approved 7,664-row development split; its reserved test was never opened.

| Project | Available | Eligible available | Allowed | Pooled development | New locked test | Time coverage of allowed rows | Missing summary / description | Exact duplicate rows | Known linked-group rows |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BIRT | 23,308 | 21,132 | 16,905 | 13,524 | 3,312 | 2005-01-07–2009-06-02 | 0 / 63 | 145 | 2,328 |
| CDT | 22,371 | 19,705 | 15,764 | 12,611 | 3,045 | 2002-01-14–2014-06-29 | 0 / 105 | 82 | 2,110 |
| Equinox | 14,559 | 13,123 | 10,498 | 8,398 | 2,031 | 2001-10-11–2013-03-15 | 0 / 40 | 28 | 1,943 |
| JDT | 63,266 | 53,076 | 42,460 | 33,968 | 8,021 | 2001-10-11–2014-04-30 | 2 / 293 | 174 | 9,743 |
| MYLYN | 13,993 | 9,673 | 7,664 | 6,131 | 1,514 | 2005-06-10–2012-08-09 | 0 / 25 | 0 | 1,264 |
| PDE | 17,639 | 15,593 | 12,474 | 9,979 | 2,425 | 2001-10-11–2012-10-23 | 0 / 70 | 65 | 2,268 |
| Papyrus | 13,253 | 10,910 | 8,728 | 6,982 | 1,734 | 2008-10-06–2017-08-31 | 0 / 352 | 20 | 593 |
| Platform | 122,496 | 106,803 | 85,442 | 68,353 | 16,459 | 2001-10-11–2013-05-24 | 1 / 793 | 421 | 19,675 |
| TPTP | 10,579 | 9,458 | 7,566 | 6,052 | 1,458 | 2003-11-05–2008-03-14 | 0 / 108 | 137 | 1,030 |

Totals are 165,998 pooled-development rows and 39,999 newly locked rows. A further 1,504
candidate test rows were conservatively excluded because their connected exact/linked group
occurred in development. Row-key and duplicate-component overlap are both zero. Every project
has non-empty development and locked partitions, and its development maximum timestamp is no
later than its locked-test minimum timestamp.

`development_class_distributions.csv` records complete per-project S6, S3 and S2 development
distributions. It confirms every task is represented overall; project/fold class absence is a
reported limitation rather than silently resampled. The historical dataset audit reports
1,804 excess exact-text rows, 40,946 linked duplicate rows and 148 cross-project exact groups
before the stricter pooled component purge.

## Search and dry-run

The frozen grid has 36 candidates per task. Planned development fits are 60/task and 180
total. The pilot intentionally sampled only 300 early development rows/project and one model
per task; its metrics are smoke diagnostics and are not scientific estimates:

| Task | Candidate | Representation | Macro-F1 | Accuracy | Weighted F1 | Balanced accuracy | Runtime |
|---|---|---|---:|---:|---:|---:|---:|
| S6 | P-000 LinearSVC | word | 0.1494 | 0.8119 | 0.7275 | 0.1667 | 7.23 s |
| S3 | P-012 LinearSVC | char | 0.3134 | 0.8874 | 0.8345 | 0.3333 | 8.92 s |
| S2 | P-024 LinearSVC | word+char | 0.5831 | 0.7659 | 0.7839 | 0.6036 | 11.38 s |

The 27.53-second pilot estimates 6.67 hours for the full 180-fit development search. Reserve
8 GB minimum / 12 GB preferred RAM and 250 MB disk. No transformer, GPU or new dependency is
required.

## Leakage, integrity and privacy gates

- Previously reserved rows are excluded before the pooled split.
- MYLYN full Parquet and reserved test are prohibited by path/hash policy.
- Duplicate connected components never cross development/test or development folds.
- S2 calibration and thresholds use only development calibration/OOF data.
- Manifests lock protocol commit/hash, all input hashes, split/fold fingerprints, candidate
  checksum, packages, seed and canonical command.
- Resume mismatches fail before continuing; checkpoints are atomic and explicit by
  task/stage/candidate/fold.
- Candidate failure is structured and non-terminal unless no eligible candidate remains.
- Published artifacts contain aggregates only; raw text, issue IDs, row predictions, models,
  engine state and locked rows are excluded.
- `held_out_test_accessed=false`; no locked performance was computed.

Preparation is valid. The full pooled real-run and any final-model fit remain unauthorized and
were not started.
