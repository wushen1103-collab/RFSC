# Phase 4 confirmatory freeze

Frozen before producing predictions on the 23 SCG-feasible targets outside the
six-target development panel.

- Confirmation unit: target; 23 targets, 114 folds, three fixed seeds, 342 runs.
- Methods: ECFP4-RF, GINE, their fixed 0.5/0.5 ensemble, and one global
  label-free gated component-scale operator.
- Frozen operator: select internal-test MMP edges with absolute fixed-ensemble
  predicted delta at least 0.5, form their connected components, and multiply
  component-centered predictions by 1.5. Test responses are prohibited from
  prediction construction.
- The configuration was chosen from the six development targets' validation
  predictions. No confirmatory target has a target-specific parameter.
- Primary comparison: pair-delta MAE versus fixed equal. Overall and cliff
  endpoint MAE must each remain within 1%; at least 12/23 target directions and
  at least 3% target-macro pair improvement are required.
- Secondary anchor: all three metrics versus GINE. RF and fixed equal remain
  explicit controls. Predictions are hash-frozen before the evaluator reads
  confirmatory labels.
- No method, threshold, scale, or target list may change after this freeze. Any
  later alternative is exploratory and cannot enter the confirmatory table.

