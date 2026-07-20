# Frozen Eclipse within-project protocol

This protocol is frozen before viewing any new project held-out result. It applies in
order to TPTP, Papyrus, PDE, Equinox, BIRT, CDT, JDT, and Platform. MYLYN remains the
immutable previously completed result and must not be retrained or reevaluated.

- Labels: blocker, critical, major, normal, minor, trivial. Enhancement, missing,
  blank, unknown, and unmapped labels are excluded without regrouping.
- Input: initial Summary and Description only. Product, Component, identifiers,
  duplicate links, and post-report information are prohibited predictive features.
- Representation: word TF-IDF, n-grams (1, 2), `min_df=2`, `max_df=0.98`, maximum
  50,000 features, sublinear TF, fitted inside each pipeline/fold.
- Models: most-frequent DummyClassifier; MultinomialNB with alpha 0.5;
  LogisticRegression with C=1.0 and balanced class weights; LinearSVC with C=1.0 and
  balanced class weights. Seeds are 42 and no per-project tuning is allowed.
- Evaluation: chronological 80/20 holdout; three-fold expanding-window CV within the
  development period; fixed-six-label macro F1; exact-text earliest-report retention;
  duplicate-link purge across CV and holdout boundaries.
- Test discipline: CV and all full-development fits complete before the test partition
  is exposed to inference. The four frozen models are evaluated together in one
  project-level held-out access event. That event is recorded exactly once and cannot
  be repeated. LogisticRegression is the predeclared primary model; test results cannot
  change the protocol or model choice.
- Uncertainty: stratified bootstrap macro-F1 confidence intervals use 1,000 resamples,
  95% confidence, and seed 42.
- Safety: one project at a time, processed Parquet only, at least 80 GiB free disk,
  complete six-class development/test/fold coverage, immutable experiment directories,
  and immediate stop on provenance change or incomplete artifacts.

Across-project primary summaries use one macro-F1 value per project. Friedman testing
of the four held-out model scores is limited to the eight newly evaluated projects,
because MYLYN has no approved held-out scores for the three baseline models. Pairwise
differences emphasize effect sizes; post-hoc significance tests are allowed only after
a significant omnibus test and remain exploratory with eight blocks.
