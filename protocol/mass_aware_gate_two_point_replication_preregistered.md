# Mass-aware gate two-point replication (pre-registered)

The `CHEMBL287_Ki/fold_0` validation-only smoke passed all six fixed checks.
Before inspecting any further mass-aware metrics, this protocol freezes two
replication tasks selected independently of mass-aware outcomes:

1. `CHEMBL287_Ki/fold_1` tests a new SCG fold of the smoke target.
2. `CHEMBL4005_Ki/fold_0` tests cross-target transfer on the same hard task used
   by the prior Multi-RF replication; it was not selected using this method.

For each task, the GINE checkpoints, three seeds, learned logistic gate,
`floor=0.4`, `k=10`, `lambda=0.5`, and 20 q-matched random controls remain
frozen. Only validation pseudo-query labels may be evaluated. Outer-test labels
and metrics are prohibited.

Each task must independently pass the smoke's six criteria: at least 2% cliff
and 5% pair-delta improvement versus normalized learned propagation; no overall
degradation; at least 40% reduction of the normalized rule's pair gap toward
base; and lower cliff MAE than both mass-aware random and similarity controls.

Both 2/2 tasks must pass. A pass authorizes only a frozen validation-only panel
matrix. A failure is NO-GO for this operator with no target-specific tuning.
