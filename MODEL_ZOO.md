# Model Zoo

Large checkpoint files are not stored directly in the Git repository. Publish them as GitHub Release assets. Most target-task checkpoints are larger than the 100 MiB regular GitHub file limit, so Release assets are the intended distribution channel for this repository.

The table below records the checkpoint produced by the latest local experiments and the file name to use for each GitHub Release asset. `method_id` and `run_dir` are included to make the weight-to-result mapping explicit.

| method_id | Model label | Role | Runs | Macro-F1 | BAcc | Release asset | Size |
| --- | --- | --- | ---: | --- | --- | --- | ---: |
| `source_resnet50` | ResNet50 source pretrain | source-task checkpoint | 1 |  |  | `best_resnet50_imagenet.pth` | 90.03 MB |
| `DHTL_wo_KMD` | DHTL-Net | target-task checkpoint | 30 | 90.82+/-0.81 | 90.88+/-0.76 | `DHTL_wo_KMD_seed0_best_model.pth` | 211.85 MB |
| `DAF` | DHTL-Net w/o KMD | target-task checkpoint | 30 | 90.31+/-0.77 | 90.38+/-0.73 | `DAF_seed0_best_model.pth` | 212.14 MB |
| `fusion_residual` | Residual | baseline checkpoint | 30 | 88.37+/-1.02 | 88.46+/-1.00 | `fusion_residual_seed0_best_model.pth` | 180.00 MB |
| `fusion_fixed_interp` | FixedInterp | baseline checkpoint | 30 | 86.79+/-0.47 | 86.91+/-0.63 | `fusion_fixed_interp_seed0_best_model.pth` | 180.00 MB |
| `fusion_learnable_interp` | LearnableInterp | baseline checkpoint | 30 | 87.51+/-0.94 | 87.63+/-1.00 | `fusion_learnable_interp_seed0_best_model.pth` | 180.00 MB |
| `fusion_gating` | Gating | baseline checkpoint | 30 | 87.35+/-0.10 | 87.44+/-0.08 | `fusion_gating_seed0_best_model.pth` | 182.68 MB |
| `fusion_adapter` | Adapter | baseline checkpoint | 30 | 87.81+/-0.73 | 88.01+/-0.76 | `fusion_adapter_seed0_best_model.pth` | 182.67 MB |
| `transfer_frozen` | Frozen | baseline checkpoint | 30 | 78.55+/-0.25 | 78.84+/-0.33 | `transfer_frozen_seed0_best_model.pth` | 90.01 MB |
| `transfer_all` | All | baseline checkpoint | 30 | 88.56+/-1.32 | 88.66+/-1.37 | `transfer_all_seed0_best_model.pth` | 90.01 MB |
| `transfer_l4` | FT-S4 | baseline checkpoint | 30 | 87.36+/-1.20 | 87.47+/-1.17 | `transfer_l4_seed0_best_model.pth` | 90.01 MB |
| `transfer_l34` | FT-S3-S4 | baseline checkpoint | 30 | 87.76+/-0.81 | 87.91+/-0.83 | `transfer_l34_seed0_best_model.pth` | 90.01 MB |
| `transfer_l234` | FT-S2-S4 | baseline checkpoint | 30 | 88.51+/-1.46 | 88.68+/-1.40 | `transfer_l234_seed0_best_model.pth` | 90.01 MB |

Expected local placement after downloading weights:

```text
checkpoints/
  best_resnet50_imagenet.pth
  DHTL_wo_KMD_seed0_best_model.pth
  DAF_seed0_best_model.pth
```

Note: The display labels follow the latest `docs/results/summaries/method_summary.csv`; use `method_id` and `run_dir` when matching checkpoints to runs.

