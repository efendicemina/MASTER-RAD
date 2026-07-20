# MYLYN pretrained representations: no material improvement

The predeclared material-improvement threshold was development CV macro F1 `0.2463`,
compared with the approved TF-IDF baseline `0.2263`.

Frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings at revision
`c21050a7ef692090620a6d037dd736908f9c7cf6` produced:

- LinearSVC: `0.2242 +/- 0.0091` macro F1;
- LogisticRegression: `0.2003 +/- 0.0033` macro F1.

Neither approach improved on TF-IDF or met the threshold. Full transformer fine-tuning
was blocked before execution because the machine has no CUDA GPU and insufficient safe
CPU-memory/runtime headroom for the required complete three-fold protocol. Synthetic CPU
smoke tests validate both weighted cross-entropy and focal-loss code paths but are not
scientific results.

No pretrained challenger configuration is created. The MYLYN held-out test remains
closed, and another one-shot evaluation is not scientifically justified by this study.
