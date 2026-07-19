# Data Leakage Risks

The project explicitly avoids the following leakage sources:

- target labels or any derivative of the target label
- priority, status, resolution, assignee, QA contact, comments, history logs, and any post-report fields
- text generated after the initial report submission
- feature preprocessing fitted on the full dataset before splitting

Mitigations implemented in code:

- preprocessing is inside scikit-learn pipelines
- split happens before model fitting
- exact duplicate reports are removed before splitting
- duplicate relationship columns are converted into groups
- test rows whose duplicate group already occurs in development are removed

Residual risk:

- a dataset may contain near-duplicates or duplicated narratives with different metadata; this is documented and should be reviewed for any real experiment.
- duplicate links may be incomplete; text-similarity auditing is therefore required on the full dataset.

## Full Eclipse ingestion boundary

Reduced Parquet files retain identifiers, project/product/component, initial Summary and
Description, severity, creation time and duplicate links plus audit flags. Comments, history,
attachments, creator/assignee/CC/QA data, priority, status and resolution are excluded before
Parquet writing. They are post-report, personal, operational, or target-proxy fields.

Only the initial Summary and Description may enter the model pipeline. Product, component,
timestamps, identifiers, duplicate fields and audit flags exist for splitting, auditing and
error analysis—not as predictive features.
