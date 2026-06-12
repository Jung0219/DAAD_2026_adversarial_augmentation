#!/usr/bin/env bash
#SBATCH --job-name=exp002_gen
#SBATCH --output=projects/adv_aug/runs/EXP-002/logs/generation_%j.out
#SBATCH --error=projects/adv_aug/runs/EXP-002/logs/generation_%j.err
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -e

cd /beegfs/jung/DAAD_2026_adversarial_augmentation

eval "$(/beegfs/jung/miniforge3/bin/conda shell.bash hook)"
conda activate AA_legacy

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p projects/adv_aug/runs/EXP-002/logs
mkdir -p results/EXP-002

python tools/attack_pointpillars_generation.py
