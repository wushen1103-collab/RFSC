| key | group | method | implementation | result_source | n_targets | n_runs | n_seeds | overall_mae_mean | overall_mae_std | cliff_mae_mean | cliff_mae_std | pair_delta_mae_mean | pair_delta_mae_std | split |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| svm_ecfp | Classical fingerprint | ECFP4-SVM | project rerun of MoleculeACE-style ECFP4 SVM | Re-run | 29 | 429 | 3 | 0.7346 | 0.0000 | 0.8250 | 0.0000 | 1.3604 | 0.0000 | identical strict SCG |
| rf_ecfp | Classical fingerprint | ECFP4-RF | project rerun of MoleculeACE ECFP4 RF | Re-run | 29 | 429 | 3 | 0.7511 | 0.0006 | 0.8405 | 0.0017 | 1.4370 | 0.0003 | identical strict SCG |
| gcn | Common supervised GNN | GCN | common PyG implementation | Re-run | 29 | 429 | 3 | 0.8116 | 0.0046 | 0.8732 | 0.0050 | 1.4159 | 0.0037 | identical strict SCG |
| graphsage | Common supervised GNN | GraphSAGE | common PyG implementation | Re-run | 29 | 429 | 3 | 0.8045 | 0.0048 | 0.8683 | 0.0039 | 1.4033 | 0.0049 | identical strict SCG |
| gin | Common supervised GNN | GIN | common PyG implementation | Re-run | 29 | 429 | 3 | 0.8016 | 0.0079 | 0.8679 | 0.0056 | 1.3883 | 0.0092 | identical strict SCG |
| gine | Common supervised GNN | GINE | common PyG implementation | Re-run | 29 | 429 | 3 | 0.8013 | 0.0052 | 0.8704 | 0.0028 | 1.3919 | 0.0079 | identical strict SCG |
| gatv2 | Common supervised GNN | GATv2 | common PyG implementation | Re-run | 29 | 429 | 3 | 0.8098 | 0.0091 | 0.8687 | 0.0132 | 1.4155 | 0.0076 | identical strict SCG |
| mpnn | Common supervised GNN | Neural MPNN | NNConv + Set2Set common PyG implementation | Re-run | 29 | 429 | 3 | 0.8350 | 0.0024 | 0.9108 | 0.0074 | 1.4562 | 0.0045 | identical strict SCG |
| attentivefp | Common supervised GNN | AttentiveFP | common PyG implementation | Re-run | 29 | 429 | 3 | 0.7922 | 0.0049 | 0.8636 | 0.0112 | 1.4178 | 0.0038 | identical strict SCG |
| chemprop_dmpnn | Molecular message passing | Chemprop D-MPNN | official Chemprop 2.2.1 | Re-run | 29 | 429 | 3 | 0.7944 | 0.0027 | 0.8512 | 0.0031 | 1.4157 | 0.0026 | identical strict SCG |
| acaloss | Direct cliff-aware competitor | ACANet ACA loss / GINE | official pinned ACA loss on identical GINE backbone | Re-run | 29 | 429 | 3 | 0.7989 | 0.0098 | 0.8738 | 0.0081 | 1.4020 | 0.0024 | identical strict SCG |
| svm_fixed_mixture | Internal frozen reference | SVM/RF/GINE equal-shape mix | fixed prediction ensemble | Ours | 29 | 429 | 3 | 0.7094 | 0.0017 | 0.7978 | 0.0013 | 1.3828 | 0.0021 | identical strict SCG |
| svm_bps_selected | Proposed method | BPS | fixed response-free incidence projection | Ours | 29 | 429 | 3 | 0.7152 | 0.0013 | 0.8023 | 0.0011 | 1.3297 | 0.0031 | identical strict SCG |
