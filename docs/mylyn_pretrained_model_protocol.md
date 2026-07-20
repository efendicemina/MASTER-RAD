# Frozen MYLYN pretrained-representation protocol

This protocol is frozen before any pretrained development result is produced. The only
allowed research data are the immutable MYLYN `development_split.csv`, Summary,
Description, and the six labels blocker, critical, major, normal, minor, and trivial.
The API rejects test split, test predictions, test metrics, held-out labels, and every
other Eclipse project. No held-out artifact may be loaded for model development.

## Fixed folds and baseline

All candidates use the exact three duplicate-purged expanding-window folds identified
by SHA-256 `5c484bdee4fb24245c060df2d7fee5091184bd6f6a1594d900ee962b7e908903`.
The approved TF-IDF Logistic Regression development baseline is macro F1 `0.2263`.

A pretrained approach is **materially promising** only if its mean fixed-fold macro F1
is at least `0.2463`, an absolute improvement of `0.0200`. This threshold exceeds the
baseline fold standard deviation (`0.0165`), represents about 8.8% relative improvement,
and prevents a small noisy difference from motivating another one-shot test.

Selection uses highest mean fixed-six-label macro F1. Results within 0.002 are tied and
use, in order: lower fold standard deviation; higher minimum per-class recall; higher
blocker then critical recall; lower computational cost; simpler reproducibility. Accuracy
or weighted F1 cannot select a model.

## Frozen semantic embeddings

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- Revision: `c21050a7ef692090620a6d037dd736908f9c7cf6`.
- License declared by the model repository: Apache-2.0.
- Official tokenizer and encoder through Hugging Face Transformers.
- Maximum length: 256 tokens; Summary tokens are reserved before Description truncation.
- Encoder is always in evaluation mode with gradients disabled. Attention-mask mean
  pooling is L2-normalized per report, without statistics from validation data.
- Embeddings are cached by model/revision, development-file hash, preprocessing
  fingerprint, and frozen-fold identity.
- Frozen classifiers: LogisticRegression and LinearSVC, C=1.0, balanced class weights,
  seed 42. No tuning, resampling, threshold adjustment, or metadata features.

## Compact transformer fine-tuning

The reusable implementation uses the same model/revision, 256 tokens, weighted
cross-entropy or focal loss, fold-training-only class weights, batch size 8, gradient
accumulation 4, learning rate 2e-5, maximum 3 epochs, gradient clipping 1.0, patience 1,
and macro-F1 checkpoint selection. The hardware audit found no CUDA GPU and only about
3.1 GiB available RAM. Full three-fold fine-tuning is therefore blocked before results;
only a tiny synthetic CPU smoke test is permitted and cannot be reported as a scientific
transformer result.

No pretrained challenger is created unless the completed frozen-embedding approach
meets `0.2463`. Any challenger remains disabled and requires explicit new approval for
held-out evaluation.
