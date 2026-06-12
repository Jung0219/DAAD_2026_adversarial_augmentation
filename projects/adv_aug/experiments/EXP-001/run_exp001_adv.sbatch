#!/usr/bin/env bash
#SBATCH --job-name=exp001-adv
#SBATCH --output=projects/adv_aug/runs/exp001_adv_mini/exp001_adv_%j.log
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -e

CODEBASE_ROOT="/beegfs/jung/DAAD_2026_adversarial_augmentation"
cd "${CODEBASE_ROOT}"

eval "$(/beegfs/jung/miniforge3/bin/conda shell.bash hook)"
conda activate AA_legacy

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD="/beegfs/jung/miniforge3/lib/libstdc++.so.6"

mkdir -p projects/adv_aug/runs/exp001_adv_mini

echo "=== STARTING EXP001 ADVERSARIAL TRAINING ==="
python tools/train.py \
    projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_adv_mini.py \
    --work-dir projects/adv_aug/runs/exp001_adv_mini \
    --no-validate
