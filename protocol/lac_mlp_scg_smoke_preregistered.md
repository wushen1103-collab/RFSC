# ICLR 2025 LAC ECFP-MLP strict-SCG smoke (pre-registered)

LAC reports an ECFP-MLP MAE reduction from 0.669 to 0.635 on D3R
(`CHEMBL234`) under the MoleculeACE split. No official implementation was found,
and the paper does not fully report epochs or resolve a curriculum notation
conflict: its prose/equation discard the easiest `R` fraction, whereas one
algorithm line says to select `R|B|` samples. Therefore this experiment is
explicitly a formula-level adaptation, not an official-code reproduction.

The fixed interpretation follows the prose and percentile equation: start with
all samples and linearly discard the bottom-loss fraction up to `R=0.2` over the
first 10% of 150 epochs. All reported settings are retained: ECFP 1024, three
512-wide hidden layers, batch 256, Adam 1e-3 with betas 0.9/0.99, non-cliff node
weight 0.5, edge alpha 0.1, and train-only MMP-cliff pairs. Baseline and LAC use
identical initialization and batch order.

Only `CHEMBL234_Ki/fold_0/seed_0` outer-train labels may enter training. The
fixed final epoch is evaluated on outer validation. No outer-test prediction or
label is authorized. LAC must improve paired MLP overall and cliff MAE by at
least 2%, not worsen pair-delta MAE, activate its edge task in at least half of
training batches, and yield finite metrics. Passing authorizes two fixed
replications only; failure closes the branch without alpha/schedule tuning.
