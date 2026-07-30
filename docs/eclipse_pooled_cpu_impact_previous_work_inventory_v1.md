# Eclipse pooled CPU impact v1: previous-work and mapping inventory

## Repository evidence

- `cadf012` implements the MYLYN-only target/feature redesign preregistered by `39e100f`.
  It tested S6/S3/S2 mappings, field-aware lexical features, preprocessing, safe metadata,
  structural features and label-token controls on MYLYN development only.
- `b24a472` implements the MYLYN-only ordinal study preregistered by `6812010`. Ordinal
  decoding reduced distance errors but did not materially improve exact-class prediction.
- Pooled Eclipse v1/rescue tested lexical word/char/word+char, safe metadata, mild/full class,
  project and duplicate weighting, NB-SVM and simple hierarchy. Rescue stopped at commit
  `631bc39` because no linear candidate passed the gate.
- No repository or all-history commit contains an equivalent pooled Eclipse BM25,
  frozen-sentence-embedding or leakage-safe stacking experiment. No such implementation is
  cherry-picked blindly.

## Source-of-truth mappings

| Severity | S6 diagnostic | Old S3 reference | S2-impact | S3-impact |
|---|---|---|---|---|
| blocker | blocker | HIGH | HIGH_IMPACT | HIGH |
| critical | critical | HIGH | HIGH_IMPACT | HIGH |
| major | major | MEDIUM | HIGH_IMPACT | HIGH |
| normal | normal | MEDIUM | LOWER_IMPACT | MEDIUM |
| minor | minor | LOW | LOWER_IMPACT | LOW |
| trivial | trivial | LOW | LOWER_IMPACT | LOW |

Old S3 and S3-impact are different targets. Old S3 R-08 (`.4420`) cannot support a new
model-improvement claim. L-S3I is the only scientific comparator for S3-impact. S2-impact
is unchanged and uses exact-row rescue R-05 as the scientific comparator. For both impact
tasks, most-frequent Dummy accuracy/Macro-F1/balanced accuracy is sanity context only.
