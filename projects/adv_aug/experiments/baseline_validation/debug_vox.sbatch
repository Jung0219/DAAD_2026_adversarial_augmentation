#!/usr/bin/env bash
#SBATCH --job-name=debug-vox
#SBATCH --output=experiments/baseline_validation/logs/debug_vox_%j.log
#SBATCH --error=experiments/baseline_validation/logs/debug_vox_%j.log
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

module load 2025 GCCcore/13.3.0
conda activate AA

cd /beegfs/jung/DAAD_2026_adversarial_augmentation
export PYTHONPATH=.

python experiments/baseline_validation/test_voxelization.py
