# Training Candidate Configuration Rationale

All Eclipse training candidate configurations are deliberately disabled. Setting
`training.enabled: true` requires explicit researcher approval after human label review.

## Confirmed dataset facts

- Nine project-level Parquet datasets preserve the raw row counts.
- Core files contain Summary and Description but exclude comments, history, attachments,
  operational status fields and personal assignment/contact data.
- Severity is imbalanced and project distributions differ.
- The combined eligible collection has 259,473 reports; normal accounts for 202,442.
- TPTP contains only 26 eligible trivial reports.
- The audit found 148 exact-text hash groups crossing project boundaries.
- Exact narrative duplicates and Bugzilla duplicate links exist.
- Creation timestamps are valid in the processed collection.

## Methodological choices

- Only Summary and Description are predictive inputs.
- `enhancement` is explicitly excluded from the primary defect-severity target.
- Chronological 80/20 holdout and expanding-window CV are retained.
- Macro-F1 with fixed labels is the primary metric.
- Duplicate groups must not cross evaluation boundaries.
- Stratified bootstrap uses a deterministic seed.
- Dummy, logistic regression and LinearSVC form a conservative initial model set.
- The grid is intentionally small to limit resource use.

## Configurable alternatives

- word versus character TF-IDF;
- summary-only, description-only and combined ablations;
- minimum document frequency and feature cap;
- within-project versus pooled evaluation;
- bootstrap resample count.

## Unresolved scientific questions

- whether Eclipse severity labels are sufficiently consistent between projects and years;
- whether excluding enhancement matches the final thesis research question;
- whether high/medium/low grouping is conceptually defensible;
- how exact duplicates with conflicting labels should be handled;
- whether leave-one-project-out evaluation represents the intended deployment setting.

The original six severity classes remain intact. No rare class is automatically merged or
discarded. Grouped severity is a secondary candidate, not an assumed superior target.
