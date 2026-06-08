#!/usr/bin/env bash
#SBATCH --job-name=visualize
#SBATCH --output=projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.log
#SBATCH --error=projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.log
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# Exit immediately if a command exits with a non-zero status
set -e

CODEBASE_ROOT="/beegfs/jung/mmdet3d_legacy"
cd "${CODEBASE_ROOT}"

# Activate the environment
source /beegfs/jung/miniforge3/etc/profile.d/conda.sh
conda activate AA_legacy

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD="/beegfs/jung/miniforge3/envs/AA_legacy/lib/libstdc++.so.6"

python projects/adv_aug/scripts/visualize.py \
    projects/adv_aug/configs/focalformer3d/FocalFormer3D_L.py \
    projects/adv_aug/runs/focalformer3d_val/predictions.pkl \
    --dataset-split val \
    --sample-idx 0 \
    --out-html projects/adv_aug/runs/focalformer3d_val/visualizations/selected/sample_000000.html
