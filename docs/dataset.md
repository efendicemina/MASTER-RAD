# Dataset

The project expects a CSV file placed manually in `data/raw/`.

Expected semantic columns:

- report identifier
- summary
- description
- severity
- creation time
- product
- component
- optional duplicate link column such as `Dupe of`

Column names are normalized to handle case differences, spaces, and minor formatting variations.

The code does not download data automatically and does not write back to the raw dataset.

## Full-dataset readiness checks

Before interpreting model results, verify:

- every retained class has enough development examples for the configured CV strategy
- no class occurs only in the future test period
- class counts are reported separately for development and test
- timestamps are parseable; partial missing dates are not silently placed in the test set
- exact duplicates and linked duplicate groups do not cross the split
- near-duplicate summary/description pairs are manually or algorithmically audited
- the product and component coverage matches the intended population of the thesis

The included `sample_data.csv` is a smoke-test sample only. Its metrics must not be reported as
research findings.

## Large Eclipse exports

Large CSVs may be configured using `dataset.path`, `dataset.paths`, or `dataset.glob`. Paths are
resolved from the repository root for configurations stored in `configs/`. Run `audit-schema`
before `build-dataset`. Construction reads only mapped research columns in 50,000-row chunks and
writes one atomic, compressed Parquet file per project under `data/processed/eclipse_core/`.

Raw-to-core ingestion preserves research rows and adds eligibility flags. Modeling exclusions
remain configuration-driven and are not destructive ingestion decisions.
