# Strict-SCG ECFP-SVM baseline addendum

This mandatory strongest-classical-baseline addendum is frozen after the Phase
4 RF/GINE comparison but before any strict-SCG SVM prediction is produced.
Confirmation responses are already known, so SVM is explicitly an addendum,
not retroactively part of the original confirmatory GO decision.

- Source: MoleculeACE official repository commit
  `7e6de0bd2968c56589c580f2a397f01c531ede26`.
- Representation: ECFP4, 1024 bits, matching the RF representation.
- Global public-mode parameters: RBF SVR, C=10, gamma=0.01, epsilon=0.1.
  The public repository has C=10 for 27/30 task configs and gamma=0.01 for
  30/30. Per-target official parameters and new validation tuning are forbidden.
- Deterministic model: one fit per target/fold. For comparison with three-seed
  neural results, the same fold prediction may be paired to each seed, but it
  must not be presented as three independent SVM fits.
- Smoke: CHEMBL287_Ki fold 0, first 128 train and 64 validation molecules;
  validation predictions only, no test prediction or response.
- Full scope: the 23 Phase 4 confirmation targets and their frozen SCG folds.
  Trainer reads outer-train responses only. Predictions and SHA manifests are
  frozen before the addendum evaluator reads any response.

