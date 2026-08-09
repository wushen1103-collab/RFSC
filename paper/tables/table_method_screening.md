| method | evidence | status |
| --- | --- | --- |
| GINE backbone | 6-target SCG panel, 87 paired runs | anchor |
| Fixed equal RF/GINE | overall/cliff -8.51%/-5.91% vs GINE; pair +1.22% | strong control; not promoted as BPS |
| BPS-GNN gated component scale (global 1.5/0.5) | 23-target confirmation vs fixed equal +1.013%/+1.041%/-2.746%; pair 22/23, bootstrap95 [-3.681%,-1.890%] | confirmatory Pareto result; locked point GO narrowly false |
| Component-preserving RF/GINE (scale 1.25) | development vs fixed equal overall/cliff/pair +0.41%/+0.22%/-1.73%; pair improved 5/6 targets | development fixed candidate |
| Pair-aligned RF/GINE | outer-test pair -1.15% vs fixed equal, but overall/cliff +4.66%/+3.06% | closed trade-off |
| Guarded pair-aligned RF/GINE | validation safety guard; test overall +0.73%, pair -0.34% vs fixed equal | closed; no pair gain |
| Three-seed fold mean | overall/cliff -4.62%/-2.52% vs seed-mean GINE; pair +1.48% | closed; variance reduction insufficient |
| Oracle boundary floor 0.4 | SCG oracle cliff -7.23% vs global; 6/6 | mechanism supported |
| Learned gate / mass-aware / dual | replications failed frozen gates | closed |
| PrismNet random-init | pair gain did not transfer; point/cliff worsened | closed |
| GraphCliff / LAC / CliffLoss | formula-level strict-SCG smokes failed | closed |
| ExtraTrees descriptors | 0/2 replication; cliff pair anti-correlated on CHEMBL4616 | closed |
| TabPFN-CheMeleonFP | 0/2 replication; cliff amplitude collapse | closed |
| Frozen MoLFormer Ridge | CHEMBL287 +116.59%/+133.04%/+103.75% overall/cliff/pair | closed |
| Context-local MMP transformation expert | train-only gate passed, but validation smoke pair worsened in 3/3 seeds | closed |
| MTPNet / SemiMol / MaskMol | audit only: leakage, missing protocol, or legacy assets | hold |
| CamS-LLaMA | reports 0.624 RMSE after Enamine675M pretraining under KPGT/official split; no linked checkpoint/code | different-protocol literature SOTA anchor |
