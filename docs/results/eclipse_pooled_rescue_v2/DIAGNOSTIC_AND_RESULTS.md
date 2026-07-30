# Eclipse pooled rescue v2: development-only diagnostic and results

## Recommendation

**STOP_LINEAR_RESCUE_AND_REDESIGN_TARGET.** No rescue candidate satisfies the combined
development gate against the exact v1 winner. The pooled locked test and reserved MYLYN
held-out test were not accessed, and locked evaluation remains unauthorized.

Rescue v2 is exploratory development work based on v1 results commit
`dc7bd9698234f8c939b9cd659daf49d3d8e240a4`. It reuses development fingerprint
`cde41dd37386b38f6d3009ff42bff9c063d6d140da36bf781faefe34c388c01b` and fold
fingerprint `727b2af6b8451adee768bc60fb6704e7459b09911b8d01099dc6df96d5c6b3bc`.

## V1 OOF error diagnosis

V1 selected-model prevalence compared with actual development OOF prevalence:

| Task/class | Actual | Predicted |
|---|---:|---:|
| S6 blocker | 1.77% | 2.19% |
| S6 critical | 3.51% | 4.03% |
| S6 major | 10.70% | 7.46% |
| S6 minor | 5.17% | 4.06% |
| S6 normal | 76.94% | 80.97% |
| S6 trivial | 1.90% | 1.28% |
| S3 HIGH | 5.29% | 14.43% |
| S3 LOW | 7.07% | 18.11% |
| S3 MEDIUM | 87.65% | 67.45% |
| S2 HIGH_IMPACT | 15.99% | 30.27% |
| S2 LOWER_IMPACT | 84.01% | 69.73% |

The dominant majority-to-minority false positives are S3 `MEDIUM->LOW` 17,846,
S3 `MEDIUM->HIGH` 14,155, and S2 `LOWER_IMPACT->HIGH_IMPACT` 26,179. S6 has a
different mixed pattern: 8,779 `major->normal`, 5,567 `normal->major`, 5,075
`minor->normal`, 3,506 `normal->minor`, 2,833 `normal->critical`, and 2,492
`critical->normal`.

V1 error rates by fold are S6 29.83%/35.18%/30.53%, S3 34.38%/34.00%/31.94%,
and S2 30.71%/27.53%/28.74%. Project error rates range from 23.4% to 42.0% for
S6, 22.9% to 41.2% for S3, and 25.2% to 35.4% for S2. BIRT, MYLYN and TPTP are
among the most difficult project/task combinations; Platform contributes many absolute
errors because of its size, but not always the highest rate.

Character and word+character models did not collapse. Their best v1 Stage-2 Macro-F1 was
within approximately .001-.007 of word models where advanced. However, v1 FeatureUnion
joined two individually L2-normalized blocks without a final global normalization, changing
effective feature scale and regularization. Rescue R-03 corrects this with field-specific
Summary/Description blocks and final global L2 normalization. Its limited gains show that
scaling was not the main scientific bottleneck.

## Privacy-safe label-quality audit

Development contains 145,322 connected duplicate components. Of these, 13,312 have more
than one row and cover 33,988 rows. There are 5,278 connected components with conflicting
severity labels, covering 15,479 rows; the largest component has 47 rows. By contrast,
only 51 exact-text groups conflict, covering 118 rows. Linked duplicates therefore often
represent related reports rather than interchangeable severity labels. Duplicate-aware
weighting is justified, but automatic relabeling or removal is not.

History was scanned only for the 159,867 approved development rows in the eight non-MYLYN
raw projects. MYLYN raw was never opened. History records 10,302 rows with an explicit
severity change. Among 124,306 currently `normal` rows, 121,798 (97.98%) have no recorded
severity-field change. This is only a proxy for “never explicitly severity-reviewed”: an
absence of a severity change does not prove that a human did not review the report.

Per-project proportions of current `normal` rows without a recorded severity change are
BIRT 98.95%, CDT 98.46%, Equinox 98.28%, JDT 98.08%, PDE 98.56%, Papyrus 97.61%,
Platform 97.87%, and TPTP 93.85%. The most common transition in every covered project is
`normal->major`. No History, status, resolution, comments, reporter, assignee, or other
post-submission data was used as a prediction feature.

Temporal label drift is project-specific. Late-minus-early `normal` prevalence changes by
+8.95 pp BIRT, +3.97 CDT, -3.55 Equinox, -1.84 JDT, -10.40 MYLYN, +3.65 PDE,
+0.23 Papyrus, +0.16 Platform, and -12.49 TPTP. Across global time quartiles, `normal`
prevalence is 78.75%, 77.38%, 74.90%, and 78.53%, so a single monotonic global drift
explanation is inadequate.

## Candidate set and budget

The bounded set contains eight candidates per task: field-word alpha .50/.75, normalized
field word+char, duplicate weighting, safe project/product/component metadata with project
weighting, deterministic multiclass NB-SVM, task hierarchy for S6/S3, and a combined
smoothed/project/duplicate/metadata candidate. Platform and OS were unavailable in the
approved processed/MYLYN development inputs and were not recovered from reserved data.

Stage A evaluates eight candidates per task on complete frozen fold 2. Stage B advances
two per task; fold 2 is reused and only folds 1/3 are newly fit. Actual budget is 36 new
fits: 24 Stage A plus 12 additional Stage B fits. S2 R-07 is transparently redundant with
R-01 because hierarchy is inapplicable to binary S2; it is treated as a redundant control,
not independent evidence, and the grid was not expanded post hoc.

## Stage B results against identical-row v1 winners

| Task | Candidate | V1 | Rescue | Delta | Project-macro | Fold wins | Project wins |
|---|---|---:|---:|---:|---:|---:|---:|
| S6 | R-03 | .2485 | .2522 | +.0036 | .2321 | 3/3 | 6/9 |
| S6 | R-08 | .2485 | .2551 | +.0066 | .2314 | 3/3 | 4/9 |
| S3 | R-08 | .4224 | .4420 | +.0196 | .4163 | 3/3 | 5/9 |
| S3 | R-03 | .4224 | .4350 | +.0126 | .4111 | 3/3 | 4/9 |
| S2 | R-05 | .5924 | .5997 | +.0073 | .5927 | 3/3 | 7/9 |
| S2 | R-08 | .5924 | .5982 | +.0059 | .5924 | 3/3 | 7/9 |

S6 R-08 removes sub-.10 recall (minimum .1423), but its gain is too small and it wins only
4/9 projects. S3 R-08 is the closest candidate, but project-bootstrap CI includes zero,
it wins only 5/9 projects, and HIGH/LOW recall falls to .2776/.2647 rather than remaining
around .35. S2 R-05 reaches recall .5090 but HIGH_IMPACT precision is only .2952, below
the desired approximately .35; its project-bootstrap CI also includes zero.

Rescue prevalence is less extreme but still overpredicts minorities: S3 R-08 predicts
HIGH/LOW in 17.23% versus actual 12.39%, and S2 R-05 predicts HIGH_IMPACT in 27.57%
versus actual 15.99%.

## Gate and conclusion

No candidate simultaneously achieves approximately +.02 Macro-F1, non-decreasing
project-macro, at least 2/3 fold wins, at least 7/9 project wins, positive row/project
bootstrap evidence, and better minority precision without material recall collapse.
Consequently no rescue protocol should be frozen and no locked evaluation command is
provided. The compliant next scientific step is target/label redesign; a separately scoped
transformer feasibility study is reasonable only after label reliability and target semantics
are resolved.

## Runtime and resource use

The 36 fit runtimes sum to 10,846 seconds (3.01 hours) in serial execution. Peak measured
RSS is 2.58 GB. Checkpoint/audit state occupies about 213 MB because it contains ignored
row-level resume predictions; published aggregate model tables occupy about 17 KB. A slow
reference bootstrap was replaced after all fits by an exactly equivalent confusion-count
implementation verified against identical RNG resamples; optimized aggregation completed
in about two minutes. No checkpoint or fitted result changed.
