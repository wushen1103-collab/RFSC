# ChEMBL 36 temporal ablation freeze

Frozen before external test labels are opened. Every row is secondary to the
main preregistered decision and cannot replace or tune the main scale 1.5 /
threshold 0.5 BPS row.

| Ablation | What it tests | Expected if mechanism is correct | Priority |
|---|---|---|---|
| Identity (`s=1`) | Whether gains require the projection | Reverts exactly to the unscaled reference | Must-run |
| All internal-test MMP edges | Whether selective boundaries matter | More point-error cost than selective BPS | Must-run |
| Random q-matched MMP edges | Whether any equal-sized edge subset works | Smaller pair gain than predicted-boundary edges | Must-run |
| Low-delta q-matched edges | Falsification/anti-boundary control | No pair advantage and potentially larger point cost | Must-run |
| SVM-only projection | Base complementarity | Smaller balanced gain than the fused reference | Must-run |
| RF/GINE-only projection | Dependence on the classical SVM anchor | Pair gain may remain but absolute level weakens | Must-run |
| ECFP4 kNN graph | Dependence on the MMP graph definition | Direction should remain but may attenuate | Must-run |
| 3x3 scale/threshold grid | Local robustness, not selection | Main row lies on a stable Pareto region | Must-run |

Random controls use seed 2901 plus a stable target/seed digest and sample without
replacement exactly the number of main active MMP edges. The anti-boundary row
uses the same count of edges with the smallest absolute unscaled predicted
difference. The kNN graph is the undirected union of each test molecule's ten
highest ECFP4 Tanimoto neighbors with similarity at least 0.4. It is constructed
from standardized test structures only and never reads responses.

Report per-target rows, target-macro effects, confidence intervals, favorable
directions, active coverage, component-size distributions, correction norms,
runtime, and memory. A failed control is retained; no control parameter may be
revised after seeing test results. No figure is generated during this stage.
