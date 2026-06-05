#!/usr/bin/env bash

# ==============================================================================
# SLURM Batch Script for FocalFormer3D (Legacy Environment)
#
# Submit this script with:
#   sbatch projects/adv_aug/FocalFormer3D/run_focalformer3d.sbatch {train|test|visualize}
# ==============================================================================

#SBATCH --job-name=focalformer3d-legacy
#SBATCH --output=projects/adv_aug/FocalFormer3D/checkpoints/slurm_%j.out
#SBATCH --error=projects/adv_aug/FocalFormer3D/checkpoints/slurm_%j.err
#SBATCH --partition=gpu            # Verify partition name on your cluster
#SBATCH --account=etechnik_gpu      # GPU partition allowed account for user jung
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=4          # CPU cores matching dataloader num_workers
#SBATCH --mem=32G                  # Total RAM allocated
#SBATCH --time=12:00:00            # Time limit (hh:mm:ss)

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration Variables (Change these as needed) ---
CONFIG="projects/adv_aug/FocalFormer3D/projects/configs/focalformer3d/FocalFormer3D_LC.py"
CHECKPOINT="projects/adv_aug/FocalFormer3D/checkpoints/FocalFormer3D_LC.pth"
SHOW_DIR="projects/adv_aug/FocalFormer3D/predictions_vis"
# --------------------------------------------------------

# Resolve codebase root directory path
START_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CURRENT_DIR="$START_DIR"
CODEBASE_ROOT=""

while [ "$CURRENT_DIR" != "/" ]; do
    if [ -f "$CURRENT_DIR/tools/train.py" ]; then
        CODEBASE_ROOT="$CURRENT_DIR"
        break
    fi
    CURRENT_DIR="$(dirname "$CURRENT_DIR")"
done

if [ -z "$CODEBASE_ROOT" ]; then
    CODEBASE_ROOT="/beegfs/jung/mmdet3d_legacy"
fi

cd "${CODEBASE_ROOT}"

echo "=== PATH RESOLUTION ==="
echo "SLURM_SUBMIT_DIR:  ${SLURM_SUBMIT_DIR}"
echo "START_DIR:         ${START_DIR}"
echo "CODEBASE_ROOT:     ${CODEBASE_ROOT}"
echo "Current Directory: $(pwd)"
echo "PYTHONPATH:        ${PYTHONPATH}"
echo "======================="

# Initialize and activate the Conda environment (AA_legacy)
CONDA_PROFILE="/beegfs/jung/miniforge3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate AA_legacy
else
    source /beegfs/jung/miniforge3/bin/activate AA_legacy
fi

# Set python path to current workspace root
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
# Fix standard library version mismatch on HPC (GLIBCXX_3.4.30)
export LD_PRELOAD="/beegfs/jung/miniforge3/envs/AA_legacy/lib/libstdc++.so.6"

usage() {
    echo "Usage: sbatch $0 {train|test|visualize}"
    exit 1
}

ACTION=$1

if [ -z "$ACTION" ]; then
    echo "Error: No action provided."
    usage
fi

case "$ACTION" in
    train)
        echo "=== SLURM WORKER: STARTING TRAINING ==="
        echo "Config: ${CONFIG}"
        echo "======================================="
        python tools/train.py "${CONFIG}"
        ;;
    test)
        echo "=== SLURM WORKER: RUNNING EVALUATION ==="
        echo "Config:     ${CONFIG}"
        echo "Checkpoint: ${CHECKPOINT}"
        echo "========================================"
        python tools/test.py "${CONFIG}" "${CHECKPOINT}" --eval bbox
        ;;
    visualize)
        echo "=== SLURM WORKER: GENERATING PREDICTION VISUALIZATIONS ==="
        echo "Config:     ${CONFIG}"
        echo "Checkpoint: ${CHECKPOINT}"
        echo "Output Dir: ${SHOW_DIR}"
        echo "=========================================================="
        python tools/test.py "${CONFIG}" "${CHECKPOINT}" --show-dir "${SHOW_DIR}"
        ;;
    *)
        echo "Error: Unknown action '$ACTION'."
        usage
        ;;
esac
