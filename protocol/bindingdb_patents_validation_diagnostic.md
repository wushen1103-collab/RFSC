# BindingDB patent validation-only projection diagnostic

Status: post-external-reveal method development only.

The fixed 1.5x external test was revealed before this diagnostic was designed.
Consequently, this experiment cannot upgrade or replace the preregistered
BindingDB-patent NO-GO decision. Its only purpose is to identify a transferable
next-generation rule for a new, untouched confirmation source.

## Firewall

- Read only `validation_predictions.parquet` from the frozen SVM, RF, and GINE
  runs.
- When a GINE validation prediction was not persisted during training, generate
  it response-free from the frozen checkpoint and validation structure IDs,
  hash-freeze it, and still do not read any response.
- Read only validation molecule responses.
- Assert validation/test molecule disjointness for every task.
- Do not load any frozen test prediction or test response.
- Select one global threshold/scale/center combination across tasks; do not
  select using the already revealed patent test metrics.

## Minimal grid

- Predicted-difference thresholds: 0.25, 0.5, 0.75, 1.0, 1.25 log units.
- Scales: 1.25, 1.5, 1.75, 2.0, 2.25, 2.5.
- Centers: ordinary minimum-L2 and frozen-ensemble-disagreement-weighted.

The ranking objective is lowest macro validation MMP-cliff pair-delta MAE among
settings with no more than 1% macro overall-MAE or cliff-endpoint-MAE cost.

## Frozen selection and follow-up

The full validation grid selected ordinary minimum-L2 centering, a 0.25-log-unit
predicted-difference threshold, and a 1.75 scale. This setting is frozen for one
post-reveal patent-test transfer check. The check remains non-confirmatory even
if it passes all numerical gates, and its outcome must not trigger further
patent-test tuning.
