# MYLYN development-only model selection protocol

This protocol is frozen before running the controlled model-development comparisons.
The immutable initial MYLYN held-out test set and its predictions or metrics must not be
read by the development study. Every comparison uses only the original chronological
development period and the same three duplicate-purged expanding-window folds.

## Primary rule

Select the eligible configuration with the highest mean expanding-window CV macro F1,
computed against the fixed labels `blocker`, `critical`, `major`, `normal`, `minor`, and
`trivial` with zero contribution for an absent class.

## Tie-breakers

Configurations whose mean macro F1 differs from the best result by no more than 0.002
absolute are treated as tied. Apply these tie-breakers in order:

1. lower standard deviation of fold macro F1;
2. higher minimum per-class recall;
3. higher mean recall across blocker and critical, provided macro F1 is not reduced by
   more than 0.01 absolute from the best result;
4. lower combined fit and inference time and lower memory use;
5. the simpler and more interpretable representation and model.

A **substantial macro-F1 loss** is defined in advance as more than 0.01 absolute below
the best eligible mean CV macro F1. No minority-class improvement may override that
limit. Product, Component, report identifiers, duplicate identifiers, and any
post-report information are prohibited predictive features.

The selected challenger remains disabled and cannot access the held-out test without a
new, explicit researcher approval.
