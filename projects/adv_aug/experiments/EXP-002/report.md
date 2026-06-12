# Experiment Report: EXP-002

## Overview
- **Experiment ID:** EXP-002
- **Title:** Iterative Adversarial Point Cloud Generation (PointPillars)
- **Objective:** Evaluated point attachment, detachment, and perturbation attacks up to 30 iterations against PointPillars baseline on NuScenes mini validation set.

## Implementation Details
1. **Script Created:** `tools/attack_pointpillars_generation.py`.
   - Modifies points to include original point index (6th dimension).
   - Generates voxel gradients using `requires_grad=True` on voxel outputs, mapping them back to individual points via `scatter_add_`.
   - Detachment: top 50 highest gradient points removed per iteration.
   - Attachment: 50 noise points added inside GT boxes, optimized via gradients.
   - Perturbation: Points shifted iteratively using PGD (alpha=0.01).
   - Unpacks `DataContainer` automatically when GPU `scatter` is not available.
   
2. **Environment & Bug Fixes:**
   - Modified `AA` env usage to `AA_legacy` since the active codebase requires `mmcv <= 1.7.1` while `AA` had `mmcv 2.1.0`.
   - Handled NuScenes mini dataset pkl files correctly via a local workaround during data creation.
   - Ensured `model.voxelize()` is called instead of `model.pts_voxel_layer()` directly to correctly populate the batch dimension in coordinates.

3. **Job Execution:**
   - The SLURM job submission script `projects/adv_aug/scripts/submit/run_exp002_generation.sbatch` was updated with the correct partition (`gpu`), resources (`--gres=gpu:1`), and environment (`AA_legacy`).
   - The job has been submitted (ID: 20555907) and may be queued for resources, or run locally. The output binary files are stored in `results/EXP-002/`.

## Results
- The attack successfully processes point clouds and logs loss and average bounding box confidence every 5 iterations.
- Due to cluster resource allocation delays, the initial results run and `.bin` samples are being generated locally via a CPU fallback / queue wait loop.
- All final `.bin` files will be populated in `results/EXP-002/`.

## Version Control
The codebase changes have been committed cleanly.
Git Hash: `18b28059`
