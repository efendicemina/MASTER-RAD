# MYLYN hierarchical classification: no material improvement

The fixed development-only study reproduced the approved flat TF-IDF baseline at mean
three-fold macro-F1 `0.226277` (folds `0.208387`, `0.222161`, `0.248281`), within the
predeclared `0.0001` tolerance of `0.2263`.

The deployable results were:

- Hierarchy A hard: `0.230993 +/- 0.025956` macro-F1;
- Hierarchy A soft: `0.227281 +/- 0.016157` macro-F1;
- Hierarchy B hard: `0.219763 +/- 0.025193` macro-F1;
- Hierarchy B soft: `0.213929 +/- 0.027665` macro-F1.

No method reached the primary threshold `0.2463`. No method met the complete secondary
minority-benefit rule. Hierarchy A hard improved macro-F1 by only `0.004717`; its mean
blocker/critical recall increased from `0.034972` to `0.047033`, well below the required
`0.0500` absolute gain. Hierarchy A soft increased that recall to `0.077442` (gain
`0.042470`) but still missed the required gain and reduced several non-severe recalls.

Hierarchy A's Level-1 decision routed only about 5% of reports incorrectly, but the
four-class NON_SEVERE child still produced many within-group errors. Its hard OOF errors
were 281 routing errors and 2,330 within-group errors. Soft composition did not solve the
child problem: 273 routing and 2,412 within-group errors.

Hierarchy B had stronger binary child classifiers, which is reflected in the
`0.548361` oracle macro-F1, but deployable routing was the dominant failure. Hard routing
produced 1,926 routing errors versus 632 within-group errors; soft routing reduced routing
errors to 1,675 but increased within-group errors to 753 and lowered macro-F1 further.

Oracle results (`0.391807` for A and `0.548361` for B) are **NON-DEPLOYABLE DIAGNOSTIC
ONLY**. They show substantial headroom under perfect group selection but cannot select a
model or justify test evaluation. The five-conflict-row diagnostic changed results only
slightly and did not alter the ordering or decision.

No challenger or secondary configuration is created. The MYLYN held-out test remains
closed: these development results do not scientifically justify a one-shot evaluation.
