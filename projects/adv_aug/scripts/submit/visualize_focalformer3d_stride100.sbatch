#!/usr/bin/env bash

#SBATCH --job-name=focalformer3d-viz100
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
