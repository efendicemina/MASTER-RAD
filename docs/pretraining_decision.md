# Pre-training Decision

## Software readiness

The ingestion, schema validation, atomic Parquet writing, manifests, resumability, audits and
pre-training checks are automated and covered by synthetic tests. Software readiness does not
imply model or thesis validity.

## Dataset readiness

All nine Eclipse projects must have equal raw/core row counts, zero silent parse losses, valid
UTC timestamps, no forbidden columns and stable source hashes. Project imbalance, temporal
drift, exact duplicates and duplicate links remain explicit dataset risks.

The completed core collection contains 301,464 rows, of which 259,473 are eligible after the
explicit enhancement exclusion. The eligible six-class distribution is: normal 202,442, major
26,337, minor 12,059, critical 9,732, blocker 4,539 and trivial 4,364. Platform contributes
106,803 eligible rows and therefore creates a material pooled-project imbalance. TPTP has only
26 trivial reports and remains the clearest rare-class warning.

There are 1,804 repeated exact-text rows, 40,946 recorded duplicate relationships and 148 exact
text hashes shared across projects. These are leakage-control requirements, not grounds for
silently deleting or relabeling data.

## Experiment readiness

- **Within-project temporal classification — WARNING:** technically feasible, but individual
  project/class support and temporal drift must be interpreted separately; TPTP trivial support
  is particularly limited.
- **Pooled temporal classification — WARNING:** technically feasible, but large projects can
  dominate the pooled objective.
- **Leave-one-project-out — WARNING:** technically feasible only if every held-out project and
  training collection retain the required labels; all six labels currently occur in every
  proposed project test split, but domain shift remains expected.
- **Original six-class classification — WARNING:** preserves the research target but retains
  class imbalance and ambiguous human labeling.
- **Grouped high/medium/low — WARNING:** technically easier, but grouping is an unresolved
  scientific decision and must not be declared superior automatically.

## Scientific readiness

Scientific readiness is not yet granted. A human must review the stratified label-quality
sample, approve target definitions, decide duplicate handling, confirm the deployment scenario
and review temporal/project drift before enabling training.

## Automated decisions

- schema and row-count validation;
- forbidden-column enforcement;
- deterministic temporal split previews;
- duplicate/hash auditing;
- threshold-based technical warnings;
- reproducible label-review sampling.

## Researcher approval required

- primary and secondary experiment designation;
- enhancement exclusion;
- grouped-label validity;
- treatment of conflicting duplicate labels;
- acceptable class-support thresholds;
- interpretation of cross-project generalization.

Recommended starting candidates are original six-class within-project temporal evaluation as
the primary methodological baseline and pooled/cross-project experiments as robustness studies.
Grouped severity should remain secondary until conceptually justified.
