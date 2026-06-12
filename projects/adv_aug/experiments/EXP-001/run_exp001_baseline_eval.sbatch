#!/usr/bin/env bash
#SBATCH --job-name=exp001-baseline-eval
#SBATCH --output=projects/adv_aug/runs/exp001_baseline_mini/exp001_baseline_eval_%j.log
#SBATCH --error=projects/adv_aug/runs/exp001_baseline_mini/exp001_baseline_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -e

CODEBASE_ROOT="/beegfs/jung/DAAD_2026_adversarial_augmentation"
cd "${CODEBASE_ROOT}"

eval "$(/beegfs/jung/miniforge3/bin/conda shell.bash hook)"
conda activate AA_legacy

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD="/beegfs/jung/miniforge3/lib/libstdc++.so.6"

echo "=== STARTING EXP-001 BASELINE EVALUATION ==="
python tools/test.py \
    projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_baseline_mini.py \
    projects/adv_aug/runs/exp001_baseline_mini/latest.pth \
    --eval bbox \
    --out projects/adv_aug/runs/exp001_baseline_mini/eval_results.pkl
echo "=== EVALUATION COMPLETE ==="
