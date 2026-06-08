#!/usr/bin/env bash
#SBATCH --job-name=visualize
#SBATCH --output=projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.log
#SBATCH --error=projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.log
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# Exit immediately if a command exits with a non-zero status
set -e

CODEBASE_ROOT="/home/btq48260/DAAD_2026_adversarial_augmentation"
cd "${CODEBASE_ROOT}"

module load GCC/11.3.0 CUDA/11.8.0

# Activate the environment
source /home/btq48260/.miniforge/etc/profile.d/conda.sh
conda activate AA

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD="/home/btq48260/.miniforge/lib/libstdc++.so.6"

python projects/adv_aug/scripts/visualize.py \
    projects/adv_aug/configs/focalformer3d/FocalFormer3D_L.py \
    projects/adv_aug/runs/focalformer3d_val/predictions.pkl \
    --dataset-split val \
    --sample-idx 0 \
    --out-html projects/adv_aug/runs/focalformer3d_val/visualizations/selected/sample_000000.html
