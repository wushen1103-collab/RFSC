# BindingDB patent structure-novelty subgroup audit

Status: frozen after the response-free overlap counts and before computing any
subgroup performance metric.

This is a post-hoc robustness diagnostic, not a new confirmation and not a
setting-selection exercise. It uses only the original fixed external candidate
(confidence-weighted BPS, threshold 0.5, scale 1.5) and its original unscaled
reference.

A patent-test molecule is `prior-overlap` when its canonical no-stereo parent
SMILES occurs anywhere in MoleculeACE or corrected ChEMBL 36; otherwise it is
`novel`. The already fixed patent cliff-pair metric is reported for:

- all internal-test cliff pairs;
- both endpoints novel;
- one novel and one prior-overlap endpoint;
- both endpoints prior-overlap.

Point MAE is also reported separately for novel and prior-overlap test
molecules. Aggregate each seed within target, then macro-average eligible
targets and use 10,000 paired target bootstraps. Every nonempty stratum is
retained. No minimum count, decision threshold, filtering, scale change, or
additional patent-test evaluation is authorized from this result.
