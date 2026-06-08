#!/usr/bin/env bash

# ==============================================================================
# SLURM Batch Script for Adversarial Attack Demo (Legacy Environment)
#
# Submit this script with:
#   sbatch projects/adv_aug/attacks/run_demo.sbatch
# ==============================================================================

#SBATCH --job-name=mmdet3d-attack-demo
#SBATCH --output=/beegfs/jung/mmdet3d_legacy/projects/adv_aug/attacks/slurm_%j.out
#SBATCH --error=/beegfs/jung/mmdet3d_legacy/projects/adv_aug/attacks/slurm_%j.err
#SBATCH --partition=gpu            # Verify partition name on your cluster
#SBATCH --account=etechnik_gpu      # GPU partition allowed account for user jung
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=4          # CPU cores matching dataloader num_workers
#SBATCH --mem=32G                  # Total RAM allocated
#SBATCH --time=00:30:00            # Time limit (hh:mm:ss)

# Exit immediately if a command exits with a non-zero status
set -e

CODEBASE_ROOT="/beegfs/jung/mmdet3d_legacy"
cd "${CODEBASE_ROOT}"

echo "=== PATH RESOLUTION ==="
echo "CODEBASE_ROOT:     ${CODEBASE_ROOT}"
echo "Current Directory: $(pwd)"
echo "======================="

# Initialize and activate the Conda environment (AA_legacy)
CONDA_PROFILE="/beegfs/jung/miniforge3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"
    conda activate AA_legacy
else
    source /beegfs/jung/miniforge3/bin/activate AA_legacy
fi

# Set python path to current codebase root
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
# Fix standard library version mismatch on HPC (GLIBCXX_3.4.30)
export LD_PRELOAD="/beegfs/jung/miniforge3/envs/AA_legacy/lib/libstdc++.so.6"

echo "=== RUNNING ATTACK DEMO SCRIPT ==="
python projects/adv_aug/attacks/mmdet3d_attack_demo.py
echo "=== ATTACK DEMO SCRIPT COMPLETED ==="
