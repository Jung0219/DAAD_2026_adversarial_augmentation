#!/usr/bin/env bash

#SBATCH --job-name=focalformer3d-pred
#SBATCH --output=/beegfs/jung/mmdet3d_legacy/projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.out
#SBATCH --error=/beegfs/jung/mmdet3d_legacy/projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

CODEBASE_ROOT="/beegfs/jung/mmdet3d_legacy"
cd "${CODEBASE_ROOT}"

source /beegfs/jung/miniforge3/etc/profile.d/conda.sh
conda activate AA_legacy

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD="/beegfs/jung/miniforge3/envs/AA_legacy/lib/libstdc++.so.6"

CONFIG="projects/adv_aug/configs/focalformer3d/FocalFormer3D_L.py"
CHECKPOINT="projects/adv_aug/checkpoints/focalformer3d/FocalFormer3D_L.pth"
OUT_PKL="projects/adv_aug/runs/focalformer3d_val/predictions.pkl"

echo "=== FocalFormer3D Prediction Export ==="
echo "Config:     ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUT_PKL}"
echo "======================================="

python tools/test.py "${CONFIG}" "${CHECKPOINT}" --out "${OUT_PKL}"
