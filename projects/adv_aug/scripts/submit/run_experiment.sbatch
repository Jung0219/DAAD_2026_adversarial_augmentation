#!/usr/bin/env bash

# ==============================================================================
# SLURM Batch Script for Adversarial Augmentation Experiments
#
# Submit this script with:
#   sbatch projects/adv_aug/scripts/submit/run_experiment.sbatch {train|test} {model}
#
# Examples:
#   sbatch projects/adv_aug/scripts/submit/run_experiment.sbatch train pointpillars
#   sbatch projects/adv_aug/scripts/submit/run_experiment.sbatch test focalformer3d
# ==============================================================================

#SBATCH --job-name=adv-aug-exp
#SBATCH --output=projects/adv_aug/runs/adhoc/logs/%x_%j.log
#SBATCH --error=projects/adv_aug/runs/adhoc/logs/%x_%j.log
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Exit immediately if a command exits with a non-zero status
set -e

ACTION=$1
MODEL=$2

usage() {
    echo "Usage: sbatch $0 {train|test} {pointpillars|centerpoint|focalformer3d|pillarnest}"
    exit 1
}

if [ -z "$ACTION" ] || [ -z "$MODEL" ]; then
    echo "Error: Missing action or model arguments."
    usage
fi

CODEBASE_ROOT="/home/btq48260/DAAD_2026_adversarial_augmentation"
cd "${CODEBASE_ROOT}"

# Load modules
module load GCC/11.3.0 CUDA/11.8.0

# Activate the environment
source /home/btq48260/.miniforge/etc/profile.d/conda.sh
conda activate AA

# Set PYTHONPATH and LD_PRELOAD
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD="/home/btq48260/.miniforge/lib/libstdc++.so.6"

# Define configurations and checkpoints
case "$MODEL" in
    pointpillars)
        CONFIG="projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py"
        CHECKPOINT="projects/adv_aug/checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth"
        ;;
    centerpoint)
        CONFIG="projects/adv_aug/configs/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py"
        CHECKPOINT="projects/adv_aug/checkpoints/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f.pth"
        ;;
    focalformer3d)
        CONFIG="projects/adv_aug/configs/focalformer3d/FocalFormer3D_L.py"
        CHECKPOINT="projects/adv_aug/checkpoints/focalformer3d/FocalFormer3D_L.pth"
        ;;
    pillarnest)
        CONFIG="projects/adv_aug/configs/pillarnest/pillarnest_tiny.py"
        CHECKPOINT="projects/adv_aug/checkpoints/pillarnest/pillarnest_tiny.pth"
        ;;
    *)
        echo "Error: Unknown model '$MODEL'."
        usage
        ;;
esac

case "$ACTION" in
    train)
        echo "=== SLURM WORKER: STARTING TRAINING ==="
        echo "Model:  ${MODEL}"
        echo "Config: ${CONFIG}"
        echo "======================================="
        CUDA_LAUNCH_BLOCKING=1 python tools/train.py "${CONFIG}" --no-validate --cfg-options runner.max_epochs=1 total_epochs=1
        ;;
    test)
        echo "=== SLURM WORKER: RUNNING EVALUATION ==="
        echo "Model:      ${MODEL}"
        echo "Config:     ${CONFIG}"
        echo "Checkpoint: ${CHECKPOINT}"
        echo "========================================"
        python tools/test.py "${CONFIG}" "${CHECKPOINT}" --eval bbox
        ;;
    *)
        echo "Error: Unknown action '$ACTION'."
        usage
        ;;
esac
