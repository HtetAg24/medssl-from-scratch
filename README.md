# From Patches to Volumes: Resource-Efficient 3D Medical Image Segmentation (Single-GPU)

This repository contains the code and reproducible pipeline for my MSc Data Science dissertation:
**“From Patches to Volumes: A Resource-Efficient Baseline for 3D Medical Image Segmentation”**.

## Summary
- Task: 3D medical image segmentation on **MSD Task02_Heart (MRI)** and **MSD Task09_Spleen (CT)**.
- Model: compact **3D Residual U-Net with Squeeze-and-Excitation (SE) attention + deep supervision**.
- Training: patch-based, foreground-aware sampling, Dice + Cross-Entropy (DiceCE), OneCycle LR, AMP.
- Focus: **patch-level capacity** (with a documented roadmap for full-volume inference).

## Key Results (Patch-level validation)
- Heart (MRI): best soft Dice **0.782**, best thresholded Dice **0.864** at τ=0.75
- Spleen (CT): best soft Dice **0.742**, optimal τ=0.50 (soft ≈ hard)

## Environment
Tested on Windows 11 + WSL2 (Ubuntu 22.04) with RTX 4070 (12GB), Python 3.10+ (Conda).

## Data (NOT included)
This repo does not include Medical Segmentation Decathlon data.
Download:
- Task02_Heart (MRI)
- Task09_Spleen (CT)
Then place them under:
- /home/htetaung/data/MSD/Task02_Heart/{imagesTr,labelsTr}
- /home/htetaung/data/MSD/Task09_Spleen/{imagesTr,labelsTr}

## Quickstart
### 1) Create environment
conda env create -f environment.yml
conda activate medssl3d

### 2) Train
bash scripts/train_heart.sh
bash scripts/train_spleen.sh

### 3) Evaluate / threshold sweep
python scripts/eval_threshold_sweep.py --task heart
python scripts/eval_threshold_sweep.py --task spleen

## Repo contents
- notebooks/: final dissertation notebook (figure/table generation)
- src/: model + training pipeline
- scripts/: training/evaluation scripts
- results/: exported figures/tables/log snippets

## Citation
If you use this work, please cite:
- the dissertation (see CITATION.cff)
- Medical Segmentation Decathlon


## Next steps to approach nnU-Net v2 strength
- add spacing-aware resampling + target voxel size per task
- add mirror TTA and softmax ensembling across flips
- remove small components (post-processing)
- implement 5-fold CV + model ensembling
- deep supervision heads at intermediate decoder scales
- robust intensity normalization (CT windowing, MRI z-score)
- experiment configs per task (Heart vs Spleen)
