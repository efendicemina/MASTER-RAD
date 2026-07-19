# Research Protocol

This repository supports a master's thesis study on automatic defect severity classification from the textual content of the initial bug report.

Implemented scope:

- multiclass severity prediction from summary and description text
- grouped severity variant for comparison
- leakage-aware preprocessing and splitting
- reproducible experiment tracking

Planned scope:

- thesis-specific statistical analysis beyond the baseline machine-learning experiments
- manual qualitative interpretation of the final outputs

Assumptions:

- the dataset is an Eclipse Bugzilla-style issue report export
- severity labels are text labels stored in one column
- creation timestamps are available for chronological holdout unless documented fallback is enabled

Actual findings are intentionally absent until a real dataset is processed.
