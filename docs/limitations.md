# Limitations

- Results depend on the quality and completeness of the manually provided dataset.
- Chronological splitting requires a parseable creation-time column.
- Duplicate and near-duplicate handling is conservative and may not eliminate all leakage risk.
- No deep-learning baseline is included by default.
- The included sample CSV is too small and imbalanced for inferential conclusions.
- Linked-duplicate protection cannot identify unrecorded near-duplicates.
- Confidence intervals remain unstable when the held-out test set has very few examples per class.
