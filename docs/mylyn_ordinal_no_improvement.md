# MYLYN ordinal classification: no material improvement

The development-only study reproduced the approved flat baseline at mean macro-F1
`0.226277`. It evaluated five balanced cumulative LogisticRegression thresholds with
hard-count, monotonic class-probability argmax, and expected-rank decoding, plus Ridge rank
regression. No method met the primary or complete secondary criterion.

The strongest exact-class ordinal method was cumulative hard decoding at macro-F1
`0.223191 +/- 0.018778`. It improved MAE from `0.686291` to `0.612476` (10.76%), reduced
the extreme-error rate from `0.030318` to `0.016399` (45.91%), and raised quadratic kappa
from `0.193841` to `0.210025`. However, it remained below the `0.2463` primary threshold,
its kappa gain (`0.0162`) was below the required `0.0500`, and its within-one gain
(`0.0286`) was just below the required `0.0300`.

Expected-rank and Ridge predictions produced still lower MAE (`0.570874` and `0.479825`)
and fewer extreme errors, but collapsed toward middle labels. Expected-rank predicted no
blocker or trivial reports and only six critical reports; Ridge predicted no blocker,
critical, or trivial reports. Both therefore had zero blocker/critical recall and poor
exact macro-F1 (`0.171204` and `0.158721`). Improved ordinal distance alone is not success
on the fixed six-class task.

The five thresholds were learnable unevenly. Mean binary macro-F1 ranged from `0.591776`
for `y>2` to `0.512468` for `y>4`. The high-severity thresholds were the practical
bottleneck: `y>3` recall was `0.069507`, while the blocker threshold `y>4` recall was only
`0.027348`. This explains the loss of rare high-severity predictions.

Raw cumulative probabilities violated monotonicity for 895 of 5,615 OOF rows (15.94%).
Deterministic PAVA correction was small—mean absolute adjustment `0.001508`, maximum
`0.133694`—and guaranteed valid class probabilities without labels. The predeclared study
did not include an invalid uncorrected argmax comparator, so no causal performance claim
about correction is made.

Excluding the five known conflicting rows did not change method ordering or the decision.
No challenger or secondary configuration is created. The MYLYN held-out test remains
closed because neither the exact-class primary criterion nor the complete ordinal-benefit
criterion was satisfied.
