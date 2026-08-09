# Corrected earliest-year-only ChEMBL 36 projection results

Relative changes are against the same frozen unscaled RF/GINE reference. Lower
is better.

| Method | Metric | Targets | Relative change | Bootstrap 95% | Favorable targets |
|---|---|---:|---:|---:|---:|
| Ordinary BPS | Overall MAE | 36 | +0.681% | [+0.385%,+1.031%] | 8/36 |
| Ordinary BPS | Cliff-endpoint MAE | 35 | +0.775% | [+0.0005%,+1.612%] | 15/35 |
| Ordinary BPS | Pair-delta MAE | 35 | **-3.725%** | **[-5.674%,-2.111%]** | **29/35** |
| Confidence-weighted BPS | Overall MAE | 36 | +0.673% | [+0.333%,+1.060%] | 9/36 |
| Confidence-weighted BPS | Cliff-endpoint MAE | 35 | +0.751% | [-0.142%,+1.717%] | 13/35 |
| Confidence-weighted BPS | Pair-delta MAE | 35 | **-3.727%** | **[-5.692%,-2.106%]** | **29/35** |

Both candidates keep point-estimate costs below the frozen 1% margin, but CI
upper bounds exceed 1%; strict statistical noninferiority remains false.

