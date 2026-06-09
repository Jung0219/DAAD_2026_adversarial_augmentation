#!/bin/bash
#SBATCH --job-name=exp002_gen
#SBATCH --output=projects/adv_aug/runs/EXP-002/logs/generation_%j.out
#SBATCH --error=projects/adv_aug/runs/EXP-002/logs/generation_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate AA
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p projects/adv_aug/runs/EXP-002/logs
mkdir -p results/EXP-002

python tools/attack_pointpillars_generation.py
