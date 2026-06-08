# Agent Context: Adversarial Augmentation Project

Last repo scan: 2026-06-08.

This file is for future agents working in `projects/adv_aug`. It should reflect
the current repository state, not only the original research idea.

---

## 1. Project Goal

Develop and evaluate runtime adversarial augmentation for LiDAR-based 3D object
detection. The research question is whether gradient-guided, realism-constrained
point/voxel perturbations can improve robustness against occlusion, sparsity, and
local blockage without unacceptable clean-performance loss on nuScenes metrics
such as mAP and NDS.

The project is staged. Baseline training/evaluation has been validated first;
adversarial training is still an implementation target, not a completed training
pipeline.

---

## 2. Active Codebase and Environment

* **Repo root:** `/home/btq48260/DAAD_2026_adversarial_augmentation`
* **Project root:** `projects/adv_aug`
* **Framework:** legacy MMDetection3D-style codebase with project plugins.
* **Conda env used by current Slurm scripts:** `AA`
* **Cluster module setup used by scripts:** `GCC/11.3.0` and `CUDA/11.8.0`
* **Important runtime exports:** `PYTHONPATH=.`, `PYTHONUNBUFFERED=1`,
  `LD_PRELOAD=/home/btq48260/.miniforge/lib/libstdc++.so.6`
* **Dataset:** nuScenes under `data/nuscenes/`.

The older environment note `AA_legacy` / CUDA 11.3 is stale for the checked-in
submission scripts. Use the current scripts unless there is a specific reason to
recreate an older environment.

---

## 3. Target Models, Configs, and Checkpoints

The active model set is four LiDAR-only baselines:

| Model | Config | Checkpoint |
| --- | --- | --- |
| PointPillars | `projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py` | `projects/adv_aug/checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth` |
| CenterPoint | `projects/adv_aug/configs/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py` | `projects/adv_aug/checkpoints/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f.pth` |
| FocalFormer3D | `projects/adv_aug/configs/focalformer3d/FocalFormer3D_L.py` | `projects/adv_aug/checkpoints/focalformer3d/FocalFormer3D_L.pth` |
| PillarNeSt | `projects/adv_aug/configs/pillarnest/pillarnest_tiny.py` | `projects/adv_aug/checkpoints/pillarnest/pillarnest_tiny.pth` |

Project configs use `custom_imports = dict(imports=['projects.adv_aug.plugins'],
allow_failed_imports=False)` where needed. FocalFormer3D, PillarNeSt, custom
heads, necks, backbones, voxel encoders, hooks, and post-processing utilities are
implemented under `projects/adv_aug/plugins`.

---

## 4. Baseline Status and Evaluation

Baseline smoke validation is marked complete in
`projects/adv_aug/adversarial_augmentation_plan.md`: train and test smoke runs
passed for PointPillars, CenterPoint, FocalFormer3D, and PillarNeSt.

Full nuScenes validation metrics are documented in
`projects/adv_aug/docs/evaluation_report.md`:

| Model | mAP | NDS | Notes |
| --- | ---: | ---: | --- |
| PointPillars | 39.83% | 53.01% | Lowest baseline, expected for simpler architecture |
| CenterPoint | 56.93% | 65.22% | Strong conventional baseline |
| PillarNeSt | 58.66% | 65.14% | Slightly higher mAP than CenterPoint, similar NDS |
| FocalFormer3D | 66.02% | 70.64% | Best baseline across all listed classes |

Use the evaluation report for class-wise AP and error metrics. Do not overwrite
those numbers unless rerunning the full validation split and recording commands,
checkpoints, and logs.

---

## 5. Current Research Phase

The active phase is **Phase 2: Pipeline Familiarization and Minimal Gradient
Sample**.

Completed:

* Phase 1 baseline train/test smoke validation for all four models.
* Baseline validation report on 6,019 nuScenes validation samples.
* A standalone PointPillars voxel-gradient proof of concept in
  `projects/adv_aug/attacks/mmdet3d_attack_demo.py`.

Not completed:

* A reusable adversarial training wrapper.
* A detector-level `train_step` or `extract_pts_feat` override that performs
  runtime adversarial augmentation during optimizer updates.
* Shared adapters for all four model families.
* Clean/adversarial training schedules or robustness benchmarks.

---

## 6. Key Architectural Finding: Voxelization Autograd Boundary

MMDetection3D-style detectors inherit from `MVXTwoStageDetector` and voxelize
points inside methods decorated with `@torch.no_grad()` / `@force_fp32()`. The
project FocalFormer3D detector also defines `voxelize` and `dynamic_voxelize`
with `@torch.no_grad()`.

Consequence: gradients to the raw `points` list are cut by voxelization and/or
non-differentiable CUDA/C++ ops. A raw-point adversarial update cannot simply set
`points.requires_grad = True` and expect useful gradients.

Working solution direction:

1. Run normal dataloader and collate/scatter.
2. Call the model voxelization path.
3. Set `voxels.requires_grad = True` on voxelization output.
4. Continue through `pts_voxel_encoder`, `pts_middle_encoder`, `pts_backbone`,
   `pts_neck`, and the detection head.
5. Backpropagate the training loss with graph retention where needed.
6. Read `voxels.grad` and compute a gradient ranking.
7. Apply a perturbation such as voxel/point zeroing, detachment, or coordinate
   shifting to a cloned/intermediate voxel tensor.
8. Re-run the relevant forward path and combine clean/adversarial losses only
   after the training design is explicit.

For FocalFormer3D, inspect both static and dynamic voxelization paths:
`FocalFormer3D.extract_pts_feat` switches based on
`self.apply_dynamic_voxelize = 'Dynamic' in pts_voxel_encoder['type']`.

---

## 7. Attack Code Status

`projects/adv_aug/attacks/mmdet3d_attack_demo.py` is the only current
MMDetection3D-specific adversarial prototype. It:

* loads the PointPillars config and checkpoint,
* builds one train sample,
* manually calls `model.voxelize(points)`,
* enables gradients on `voxels`,
* computes PointPillars training loss,
* verifies voxel gradients,
* zeroes the top 10% highest-gradient voxels,
* evaluates the perturbed loss through the manual point branch.

`projects/adv_aug/attacks/attack.py`, `attach.py`, and `detach.py` were copied
from the IJCV paper "A Comprehensive Study of the Robustness for LiDAR-based 3D
Object Detectors against Adversarial Attacks". They are OpenPCDet-based and are
not directly compatible with this MMDetection3D project without adaptation.

Note: `attacks/READEME.md` is misspelled in the repo.

---

## 8. Runnable Scripts

General train/test Slurm entry point:

```bash
sbatch projects/adv_aug/scripts/submit/run_experiment.sbatch train pointpillars
sbatch projects/adv_aug/scripts/submit/run_experiment.sbatch test focalformer3d
```

Accepted models: `pointpillars`, `centerpoint`, `focalformer3d`, `pillarnest`.
Accepted actions: `train`, `test`.

FocalFormer3D prediction export:

```bash
sbatch projects/adv_aug/scripts/submit/run_focalformer3d_predictions.sbatch
```

Visualization:

```bash
sbatch projects/adv_aug/scripts/submit/visualize.sbatch
sbatch projects/adv_aug/scripts/submit/visualize_focalformer3d_stride100.sbatch
```

The visualization script itself is
`projects/adv_aug/scripts/visualize.py`. It loads predictions from a `.pkl`,
loads raw nuScenes point clouds, overlays ground-truth and predicted 3D boxes,
and writes Plotly HTML. It supports `--sample-idx`, `--sample-stride`,
`--multi-scene`, `--score-thr`, and `--point-downsample`.

---

## 9. Repo Layout Summary

* `adversarial_augmentation_plan.md`: staged research plan and phase tracking.
* `docs/evaluation_report.md`: baseline nuScenes validation results.
* `attacks/`: exploratory adversarial attack code; only
  `mmdet3d_attack_demo.py` is native to this MMDetection3D repo.
* `configs/pointpillars`, `configs/centerpoint`: baseline configs.
* `configs/focalformer3d`: FocalFormer3D and DeformFormer3D configs.
* `configs/pillarnest`: PillarNeSt and CenterPoint-plus configs.
* `plugins/`: project-specific models, heads, backbones, necks, voxel encoders,
  dataset transforms, hooks, bbox coders/assigners, post-processing, and custom
  CUDA ops.
* `scripts/submit`: Slurm entry points for experiments, prediction export, and
  visualization.
* `runs/`: intended output location for logs, predictions, and visualizations.

---

## 10. Practical Guidance for Next Agents

1. Before editing code, check `git status --short`. This repo may have unrelated
   local modifications; do not revert them unless explicitly asked.
2. Treat Phase 2 as active. Do not claim adversarial training is implemented
   until there is a real integrated training path.
3. Start with PointPillars/CenterPoint for shared voxel-level logic; inspect
   FocalFormer3D and PillarNeSt separately because they use project plugins.
4. Keep attack code model-side. CPU dataloader transforms are appropriate for
   normal augmentation, but gradient-guided adversarial augmentation belongs in
   the forward/train path.
5. When adapting `attack.py`, `attach.py`, or `detach.py`, account for the
   OpenPCDet-to-MMDetection3D API mismatch instead of copying logic blindly.
6. If rerunning Slurm jobs, ensure checkpoint files exist and output directories
   under `projects/adv_aug/runs/.../logs` exist or are created by the script.
7. Record any new experiment command, config, checkpoint, seed, output path, and
   metric summary in `docs/` or the plan file.
