#!/usr/bin/env bash

#SBATCH --job-name=focalformer3d-pred
#SBATCH --output=projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.out
#SBATCH --error=projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.err
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e

CODEBASE_ROOT="/home/btq48260/DAAD_2026_adversarial_augmentation"
cd "${CODEBASE_ROOT}"

module load GCC/11.3.0 CUDA/11.8.0

source /home/btq48260/.miniforge/etc/profile.d/conda.sh
conda activate AA

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD="/home/btq48260/.miniforge/lib/libstdc++.so.6"

CONFIG="projects/adv_aug/configs/focalformer3d/FocalFormer3D_L.py"
CHECKPOINT="projects/adv_aug/checkpoints/focalformer3d/FocalFormer3D_L.pth"
OUT_PKL="projects/adv_aug/runs/focalformer3d_val/predictions.pkl"

echo "=== FocalFormer3D Prediction Export ==="
echo "Config:     ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUT_PKL}"
echo "======================================="

python tools/test.py "${CONFIG}" "${CHECKPOINT}" --out "${OUT_PKL}"
