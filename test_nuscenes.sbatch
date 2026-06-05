#!/bin/bash
#SBATCH --job-name=test_nuscenes
#SBATCH --account=etechnik_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=test_nuscenes_%j.log
#SBATCH --error=test_nuscenes_%j.err

# Print diagnostics
echo "Starting job ${SLURM_JOB_ID} on node $(hostname) at $(date)"

# Load modules
module load 2025 GCCcore/13.3.0
module load nodejs/20.13.1

# Activate environment
source /beegfs/jung/miniforge3/etc/profile.d/conda.sh
conda activate AA_legacy

# Set PYTHONPATH
export PYTHONPATH=/beegfs/jung/mmdet3d_legacy:$PYTHONPATH

# Run test/evaluation
python tools/test.py projects/adv_aug/PillarNeSt/projects/configs/pillarnest/pillarnest_tiny.py projects/adv_aug/PillarNeSt/checkpoints/PillarNeSt/pillarnest_tiny.pth --eval bbox
