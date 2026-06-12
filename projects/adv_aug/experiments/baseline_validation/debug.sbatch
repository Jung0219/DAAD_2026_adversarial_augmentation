#!/usr/bin/env bash
#SBATCH --job-name=debug-adv-aug
#SBATCH --output=experiments/baseline_validation/logs/debug_%j.log
#SBATCH --error=experiments/baseline_validation/logs/debug_%j.log
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00

module load 2025 GCCcore/13.3.0
conda activate AA

cd /beegfs/jung/DAAD_2026_adversarial_augmentation
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

python experiments/baseline_validation/debug_inference.py
