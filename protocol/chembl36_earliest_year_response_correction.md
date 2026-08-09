# ChEMBL 36 earliest-year response correction

The original frozen ChEMBL 36 protocol assigned a molecule to its earliest
publication year but aggregated accepted annual medians across every observed
year. That rule was followed exactly, yet it permits later responses to affect
earlier train/validation labels and is too weak for a strict temporal claim.

Before constructing a corrected dataset or prediction, freeze this correction:

1. within a target-molecule-year, retain the original median and one-log-unit
   discordance rejection;
2. for a molecule observed in several years, retain only the earliest accepted
   year's median and replicas as its label;
3. exclude all later-year response groups and record their counts;
4. keep every other curation, split, model, graph, ensemble, projection, seed,
   eligibility, metric, and statistical setting unchanged;
5. do not reuse any old model prediction, because train/validation labels may
   change;
6. label the corrected result post-primary robustness, not a new independent
   confirmation, because the method and issue were identified after the first
   ChEMBL evaluation.

Before corrected predictions were built, the unchanged ensemble weights,
active-edge threshold, scale, evaluation metrics, bootstrap, multiplicity, and
decision gates were copied into
`configs/chembl36_temporal_external_earliest_year.yaml`. The unchanged
confidence-weighted formula and invariant tolerance were separately frozen in
`configs/confidence_weighted_projection_chembl36_earliest_year.yaml`.
