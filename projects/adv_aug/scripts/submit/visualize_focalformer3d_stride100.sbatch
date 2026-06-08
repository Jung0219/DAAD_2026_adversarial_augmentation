#!/usr/bin/env bash

#SBATCH --job-name=focalformer3d-viz100
#SBATCH --output=/beegfs/jung/mmdet3d_legacy/projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.out
#SBATCH --error=/beegfs/jung/mmdet3d_legacy/projects/adv_aug/runs/focalformer3d_val/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e

CODEBASE_ROOT="/beegfs/jung/mmdet3d_legacy"
cd "${CODEBASE_ROOT}"

source /beegfs/jung/miniforge3/etc/profile.d/conda.sh
conda activate AA_legacy

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD="/beegfs/jung/miniforge3/envs/AA_legacy/lib/libstdc++.so.6"

CONFIG="projects/adv_aug/configs/focalformer3d/FocalFormer3D_L.py"
PREDICTIONS="projects/adv_aug/runs/focalformer3d_val/predictions.pkl"
LEGACY_PREDICTIONS="projects/adv_aug/runs/focalformer3d_val_predictions.pkl"
OUT_DIR="projects/adv_aug/runs/focalformer3d_val/visualizations/stride100"

if [ ! -f "${PREDICTIONS}" ] && [ -f "${LEGACY_PREDICTIONS}" ]; then
    mv "${LEGACY_PREDICTIONS}" "${PREDICTIONS}"
fi

if [ -d "projects/adv_aug/runs/logs" ]; then
    find projects/adv_aug/runs/logs -maxdepth 1 -type f -name 'focalformer3d-pred_*' \
        -exec mv -t projects/adv_aug/runs/focalformer3d_val/logs {} +
    rmdir projects/adv_aug/runs/logs 2>/dev/null || true
fi

echo "=== FocalFormer3D Stride-100 Visualization ==="
echo "Config:      ${CONFIG}"
echo "Predictions: ${PREDICTIONS}"
echo "Output dir:  ${OUT_DIR}"
echo "=============================================="

python projects/adv_aug/scripts/visualize.py \
    "${CONFIG}" \
    "${PREDICTIONS}" \
    --dataset-split val \
    --sample-stride 100 \
    --out-dir "${OUT_DIR}" \
    --score-thr 0.35 \
    --point-downsample 10
