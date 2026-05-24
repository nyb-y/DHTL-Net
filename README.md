# DHTL-Net for Binary Ultrasound IQA

This repository provides the code for DHTL-Net on binary ultrasound image quality assessment (IQA), including training, baseline comparison, result aggregation, top-1 frame recommendation, and figure generation.

The repository is organized as a code-first release. Model weights are distributed separately through GitHub Releases, and summarized experiment outputs are kept under `docs/`.

## Repository Structure

```text
tools/proposed_method/  DHTL-Net training and ablation modules
tools/baselines/        Baseline training scripts
tools/analysis/         Result aggregation scripts
tools/recommendation/   Top-1 recommendation script
tools/visualization/    Figure generation scripts
configs/commands/       Reproducible command examples
checkpoints/            Local placeholder for downloaded weights
docs/datasets/          Dataset metadata and source links
docs/results/           Summary tables, raw exports, and metadata
docs/figures/           Summary and qualitative figures
MODEL_ZOO.md            Checkpoint list and Release asset names
```

## Deployment

1. Clone the repository.

```bash
git clone https://github.com/nyb-y/DHTL-Net.git
cd DHTL-Net
```

2. Create a Python environment.

```bash
conda create -n dhtl-iqa python=3.10
conda activate dhtl-iqa
```

3. Install PyTorch for your CUDA version.

Install the matching `torch` and `torchvision` builds from the official PyTorch installation page:

```bash
pip install torch torchvision
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

4. Prepare the target-task dataset.

The target-task dataset should be organized as:

```text
data/mendeley_iqa_701515_grouped900/
  train/
    usable/
    unusable/
  val/
    usable/
    unusable/
  test/
    usable/
    unusable/
```

Public dataset sources:

- Source-task dataset `FETAL_PLANES_DB_data`: https://zenodo.org/records/3904280
- Target-task dataset source: https://www.kaggle.com/datasets/orvile/dataset-for-fetus-framework
- Target-task dataset source mirror: https://zenodo.org/records/11005384

Dataset metadata and grouping files are provided in `docs/datasets/`.

5. Download weights from GitHub Releases.

The `.pth` files are not committed to the Git repository. Download the checkpoint files from the [DHTL-Net Releases](https://github.com/nyb-y/DHTL-Net/releases) page and place them under `checkpoints/`.

At minimum, the command examples expect:

```text
checkpoints/
  best_resnet50_imagenet.pth
```

Additional target-task checkpoints and their expected Release asset names are listed in `MODEL_ZOO.md`.

6. Run training.

The training command examples are configured for 30 seeds (`0` to `29`):

```bash
bash configs/commands/train_dhtl.sh
bash configs/commands/train_dhtl_wo_kmd.sh
bash configs/commands/train_fusion_baselines.sh
bash configs/commands/train_transfer_baselines.sh
```

The main training entry points are:

```text
tools/proposed_method/train_dhtl.py
tools/proposed_method/train_dhtl_wo_kmd.py
tools/baselines/train_fusion_baselines_minimal.py
tools/baselines/train_transfer_baselines_minimal.py
```

7. Aggregate results and generate figures.

On Windows:

```bat
configs\commands\aggregate_results_windows.bat
```

The generated summary tables and figures are stored in `docs/results/` and `docs/figures/`.

8. Run top-1 recommendation evaluation.

```bash
bash configs/commands/recommend_top1.sh
```

The recommendation summary is stored in `docs/results/recommendation/`.

## Results

The released results summarize 12 methods over 30 seeds. The full result tables are available in `docs/results/summaries/`, and checkpoint-to-result mappings are documented in `MODEL_ZOO.md`.

## Notes

- Weight files should be uploaded as GitHub Release assets, not committed into the Git repository.
- Before running commands, adjust dataset and checkpoint paths if your local layout differs from the examples.
