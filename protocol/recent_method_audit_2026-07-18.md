# Recent activity-cliff method audit (2026-07-18)

This audit uses primary papers and official repositories only. It is a collision and
reproducibility screen, not a claim of reproduced performance.

## SemiMol (IJCAI 2024)

- Primary paper: <https://www.ijcai.org/proceedings/2024/672>
- Uses an unlabeled pool stated to be 100--10,000 times larger than the labeled set.
- The paper does not identify the experimental unlabeled pool and no official implementation was
  found. A strict-SCG reproduction would therefore introduce an unaudited distribution and is not
  run.

## ACANet v3 / ACA loss (ChemRxiv v5, 2026)

- Official repository: <https://github.com/shenwanxiang/ACANet>
- Frozen local source commit: `4c43e6024345d0ec4d40fdc88ad57224c898b6da` (release v3).
- The v2.0-to-v3 repository diff changes only package version declarations in
  `clsar/__init__.py` and `clsar/model/train.py`; the ACA loss is unchanged.
- The already completed official-loss paired SCG experiment remains the relevant result; repeating
  it would waste compute.

## GraphCliff (KDD 2026 / arXiv:2511.03170v2)

- Primary paper: <https://arxiv.org/abs/2511.03170v2>
- Uses GINE short-range filtering, a learnable Chebyshev long-range filter, sigmoid gating,
  residual connections, and SAGPool.
- No official code, random seeds, or complete hyperparameters were found in the paper or author
  pages. We therefore label our implementation a formula-level adaptation.
- It is evaluated first as a single validation-only strict-SCG smoke test. It is a competing
  backbone, not a BPS-GNN contribution, which prevents novelty collision.
- Smoke result: validation MAE 0.759055 versus 0.632746 for the frozen GINE anchor (+19.962%).
  The run stopped after 44 epochs with its best checkpoint at epoch 24. It crossed the
  preregistered +5% rejection boundary, so no test labels were read and no hyperparameter search or
  additional GraphCliff run is permitted.

## LAC / pairwise cliff objectives (ICLR 2025)

- Primary paper: <https://openreview.net/forum?id=6DHIkLv5i3>
- Its explicit pairwise activity-cliff objective overlaps the signed-MMP/SQRL family already audited
  and tested. It is retained as related work rather than copied into the proposed method.

## Multi-RF Fusion (OGB leaderboard / arXiv:2603.20724v1)

- Primary paper and official code: <https://github.com/stef41/Multi-RF-Fusion>
- Audited official repository commit: `e020cb9fd73e3dcbbe6161dd7408264d2d91af6b`.
- The released classifier fuses FCFP2, FCFP3, ECFP2, MACCS, and hashed atom-pair bits and uses
  `max_features=0.20`; OGB lists the complete method first on ogbg-molhiv as of this audit.
- Rank averaging and class weighting do not transfer to pActivity regression. The strict-SCG smoke
  therefore uses only the exact released 4,263-dimensional feature ordering and the 0.20 feature
  fraction in a squared-error RF regressor.
- The single-fold smoke uses 500 trees and eight CPU workers, never materializes outer-test labels or
  predictions, and has a preregistered joint overall/cliff/pair gate before any replication.
- Smoke result: the RF validation RMSE improved by 6.081%. After fixed 50/50 blending with the
  frozen GINE validation prediction, overall endpoint MAE, cliff-endpoint MAE, and cliff-pair delta
  MAE improved by 2.251%, 6.010%, and 9.386%, respectively. All four preregistered gates passed;
  the next allowed action is a two-point unseen-fold replication.
- Two-point replication result: CHEMBL287 fold 1 passed all four gates (RF RMSE -5.329%; fixed
  equal overall/cliff/pair -2.179%/-0.237%/-2.920%), but CHEMBL4005 fold 0 failed all four (RF
  RMSE +35.349%; fixed equal overall/cliff/pair +16.695%/+21.182%/+5.291%). Only one of two
  unseen tasks passed, so the route is NO-GO. No outer-test prediction or label was materialized;
  per-target selection, feature tuning, and the 87-run matrix are prohibited.

## RDKit-descriptor ExtraTrees (bioRxiv/arXiv 2026 benchmark)

- Primary paper: <https://arxiv.org/abs/2604.26498v3>; supplementary results:
  <https://www.biorxiv.org/content/10.64898/2026.04.29.721568v4.supplementary-material>.
- The paper evaluates 300-tree ExtraTrees on ECFP4, ECFP6, MACCS, and RDKit descriptors and
  reports compact classical models as the largest winner family across 26 endpoints and three
  splits. Its structure-separated split is SVD plus MiniBatchKMeans, not strict SCG.
- The paper and its sole released supplementary PDF do not provide the claimed reproducibility
  package, descriptor cleaning, or complete ExtraTrees parameters; no author-controlled public
  repository was found. Our run is therefore a configuration-level adaptation using pinned
  RDKit's registered 2D descriptors and disclosed sklearn settings, not an official reproduction.
- The single pre-registered validation-only smoke passed all four gates. ExtraTrees RMSE improved
  16.546% versus ECFP-RF; the frozen 0.50 GINE + 0.25 RF + 0.25 ExtraTrees mixture improved
  overall/cliff/pair MAE 6.393%/10.121%/12.794% versus frozen equal RF/GINE. No outer-test label
  or prediction was materialized. Only a separately pre-registered two-point validation
  replication is authorized.
- The two-point replication passed 0/2 tasks and 4/8 checks. On CHEMBL2971 Ki, ExtraTrees RMSE
  changed +2.874% and the mixture changed overall/cliff/pair +1.633%/+6.278%/-1.292%. On
  CHEMBL4616 EC50, ExtraTrees RMSE changed -4.533%, but the mixture changed
  -2.912%/+5.373%/+17.349%. Post-hoc pair diagnostics found ExtraTrees delta correlation/sign
  accuracy 0.216/45.1% and -0.599/15.4%, respectively. Preserve NO-GO; do not select descriptor
  backbones per target or tune ensemble weights.

## CheMeleon and TabPFN-CheMeleonFP

- CheMeleon's official MIT source tree is pinned at
  `f1a01b62f2f467bf96959748b624a24d2f81f215`; its public 34.9 MB checkpoint was pretrained on
  one million PubChem structures with deterministic Mordred targets and does not use MoleculeACE
  activity labels. The paper explicitly reports that it still struggles with activity cliffs.
- The newer TabPFN-CheMeleonFP paper (<https://arxiv.org/abs/2604.16123v2>) reports a stronger
  frozen-embedding in-context workflow. Its author Zenodo archive (record 19605631, MD5
  `d004e61dfaa3d96bcb975fae62784fb6`) pins `tabpfn==7.0.1`, uses TabPFN-2.6, train-only
  standardization and `[-6,6]` clipping, and evaluates five fixed seeds on official MoleculeACE
  splits.
- The archive contains no model checkpoint. Official TabPFN-2.6 access requires Prior Labs
  registration, model-license acceptance, and a token. Per the experiment mandate, no adapted
  smoke or environment mutation is performed until the user completes that account action.
