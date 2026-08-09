# BindingDB patent fixed-rule performance by prior-structure novelty

`Novel` means absent from all MoleculeACE and corrected ChEMBL structures. This
is a post-hoc robustness diagnostic; lower relative change is better.

| Metric | Stratum | Tasks | Observations | Relative change | Bootstrap 95% CI | Favorable |
|---|---|---:|---:|---:|---:|---:|
| Pair MAE | All | 42 | 31,792 pairs | -1.266% | [-1.943%, -0.677%] | 23/42 |
| Pair MAE | Both novel | 41 | 23,831 pairs | -0.782% | [-1.440%, -0.204%] | 17/41 |
| Pair MAE | Mixed | 10 | 211 pairs | -5.133% | [-16.821%, +1.333%] | 5/10 |
| Pair MAE | Both prior-overlap | 12 | 7,750 pairs | -1.171% | [-3.478%, +0.945%] | 8/12 |
| Overall MAE | Novel molecules | 42 | 9,973 molecules | +0.260% | [+0.080%, +0.460%] | 9/42 |
| Overall MAE | Prior-overlap molecules | 23 | 3,723 molecules | +0.250% | [-0.057%, +0.586%] | 5/23 |

The both-novel macro interval excludes zero, but its favorable-direction count
is low. This supports overlap robustness, not external confirmation.
