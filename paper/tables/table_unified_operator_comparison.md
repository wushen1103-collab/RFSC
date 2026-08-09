# Unified incidence-projection instances

| Instance | Input and incidence action | Inference information | Exact guarantee | Cost used here | Evaluated failure boundary |
|---|---|---|---|---|---|
| Node BPS | Row-space projector selects/amplifies active variation in a node prediction vector | Predictions, structure topology, frozen threshold/scale; no query response | Active-edge amplification, component-mean conservation, minimum correction; **no universal error dominance** | Matrix-free `O(n+m)` | Wrong active components can alter absolute calibration and fail point-risk noninferiority |
| Complete edge Hodge | Column-space projector removes non-gradient variation from a complete ordered pair flow | Pair predictions and complete topology; no response or tuned parameter | Exact identity/reversal/additivity and pair-MSE dominance for scalar-property truth | Closed-form `O(n^2)`; `O(n)` extra beyond input/output | Endpoint-specific MAE, including high-delta tails, need not improve |
| Incomplete edge Hodge | Column-space least-squares integration on observed directed edges | Observed pair predictions and topology; no response | Pair-MSE dominance **only on observed edges**; fitted flow unique up to potential gauge | Dense least-squares backend at benchmark scale | Held-edge gains are empirical; random connected missingness does not cover adversarial/disconnected acquisition |

