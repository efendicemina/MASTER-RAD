# Experiment Design

Primary experiment:

- multiclass severity prediction
- macro F1 as the selection metric
- chronological holdout split
- model selection performed only on the development set
- expanding-window cross-validation inside the development period
- a fixed label set for all folds and final metrics

Secondary experiment:

- grouped severity labels: high, medium, low

Model families:

- DummyClassifier
- MultinomialNB
- LogisticRegression
- LinearSVC

The final held-out test set is evaluated once per experiment.

All candidate models, including the majority-class dummy baseline, use the same
cross-validation folds. Hyperparameter selection never uses the final test period.

For a chronological experiment, a class may legitimately be absent from the future test
period. A class appearing only in the future test period is rejected because the model could
not have learned it. Reported macro metrics retain the development label set for consistency.

Recommended ablations for the full dataset:

- summary only
- description only
- summary and description
- word TF-IDF
- character `char_wb` TF-IDF
- original and grouped severity targets
