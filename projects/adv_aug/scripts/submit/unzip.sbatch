#!/usr/bin/env bash

#SBATCH --job-name=extract-data
#SBATCH --output=/beegfs/jung/DAAD_2026_adversarial_augmentation/projects/adv_aug/scripts/logs/extract-data_%j.out
#SBATCH --error=/beegfs/jung/DAAD_2026_adversarial_augmentation/projects/adv_aug/scripts/logs/extract-data_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /beegfs/jung/data
mkdir -p nuscenes kitti pretrain_weight

echo "Starting extraction at $(date)"

echo "Extracting nuScenes..."
tar -xvf nuscenes.tar -C nuscenes

echo "Extracting KITTI..."
tar -xvf kitti.tar -C kitti

echo "Extracting pretrained weights..."
tar -xvf pretrain_weight.tar -C pretrain_weight

echo "Finished extraction at $(date)"
du -sh nuscenes kitti waymo pretrain_weight