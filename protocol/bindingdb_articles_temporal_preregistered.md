# BindingDB articles temporal external confirmation freeze

Frozen before parsing any affinity value from the downloaded TSV. Header names,
file size, line count, checksum, and source documentation were audited first;
no response distribution or target-specific response statistic was inspected.

## Source independence and task units

- Use the official BindingDB 2026-07 TSV containing only measurements curated
  from journal articles by BindingDB. Do not use the separate ChEMBL, patent,
  PubChem, or PDSP Ki subsets.
- The immutable ZIP has 18,114,757 bytes and MD5
  `f766b06b8d137f7465bcab7ce1f76e56`.
- Retain exact positive numeric Ki and Kd values in nM for human, single-chain
  protein targets with a Swiss-Prot primary ID, a SMILES, and publication year.
  Ki and Kd for the same protein are distinct task units and are never pooled.
- Convert each measurement to `pAffinity = 9-log10(value_nM)` and use the
  repository's fixed RDKit standardizer.

## Duplicate and temporal firewall

- Within target-task, standardized molecule, and publication year, use the
  median only if the pAffinity range is at most 1 log unit; otherwise discard
  that group.
- When a molecule occurs in multiple years, use the response from its earliest
  accepted publication year only. Later-year measurements are excluded rather
  than aggregated backward in time.
- Train is publication year <=2019, validation is 2020--2021, and test is
  >=2022. The earliest-year rule makes standardized molecules disjoint.
- Structure-only eligibility requires >=100/20/30 train/validation/test
  molecules and >=8 internal-test MMP edges. Include all passing tasks. If
  fewer than eight tasks pass, stop without relaxing a threshold.

## Fixed predictors and CW-BPS

- Reuse the exact ChEMBL external SVM, RF, and GINE recipes without per-task
  tuning. The unscaled reference remains 0.5 SVM + 0.25 RF + 0.25 GINE.
- Build the internal-test MMP graph without responses. Activate edges at frozen
  unscaled predicted absolute delta >=0.5 and use scale 1.5.
- Confidence weights are exactly the already frozen ChEMBL exploratory formula:
  reciprocal weighted ensemble disagreement with target-test median+1e-6
  stabilizer. No clipping, grid, or target-specific choice is allowed.
- Hash every base, reference, ordinary-BPS, and CW-BPS test prediction before
  the evaluator reads a test response.

## Endpoints and decision

The primary comparison is CW-BPS versus the unscaled reference, aggregated seed
then task. Report overall MAE, true-cliff endpoint MAE, and true MMP-cliff
pair-delta MAE at a one-log-unit cliff threshold. Use 10,000 target/task
bootstraps, target/task-level one-sided Wilcoxon tests, and Holm correction.

All gates must pass: >=3% pair improvement, pair bootstrap upper bound <0,
>=70% favorable pair tasks, <=1% overall and cliff point costs, both point-cost
bootstrap upper bounds <=1%, and >=8 pair-eligible tasks. Ordinary BPS and
Ki/Kd-stratified rows are prespecified secondary analyses. A failure cannot be
repaired by tuning this test set.
