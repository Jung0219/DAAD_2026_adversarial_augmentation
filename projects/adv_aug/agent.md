# Agent Context: Adversarial Augmentation Project

This file provides critical context for AI agents working on this repository to help them understand the codebase structure, constraints, and current design decisions.

---

## 1. Project Goal
Implement **runtime adversarial augmentation** (e.g. point detachment/shifting) inside the training loop of 3D object detectors to improve their robustness against occlusion and sparsity, without sacrificing clean performance (mAP/NDS).

---

## 2. Target Environment & Models
* **Codebase:** Legacy MMDetection3D (v0.18.1) located at `/beegfs/jung/mmdet3d_legacy`.
* **Conda Env:** `AA_legacy` (PyTorch 1.12.1, CUDA 11.3).
* **Primary Model:** PointPillars (Configuration: `configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py`).
* **Secondary Model:** CenterPoint (Configuration: `configs/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py`).

---

## 3. Core Architectural Constraint & Discovery
### The Voxelization Autograd Boundary
During the forward pass of MMDetection3D detectors (inheriting from `MVXTwoStageDetector` at [mvx_two_stage.py](file:///beegfs/jung/mmdet3d_legacy/mmdet3d/models/detectors/mvx_two_stage.py#L211-L236)), the `voxelize` method is decorated with `@torch.no_grad()` and calls a non-differentiable C++/CUDA backend:

```python
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        ...
```

**Consequence:** Any gradients with respect to the raw input `points` list are completely cut, resulting in `None` gradients.

### The Solution: Voxel-Level Perturbation
Rather than backpropagating to the raw point list, we must intercept and perturb the intermediate `voxels` tensor returned by `voxelize()`.
1. Set `voxels.requires_grad = True` on the output of `self.voxelize(points)`.
2. Compute the forward pass loss, run `loss.backward(retain_graph=True)`.
3. Read the gradients from `voxels.grad` (shape: `[num_voxels, max_points_per_voxel, num_features]`).
4. Apply the adversarial perturbation (e.g., zeroing out points with high gradient norm for detachment, or shifting coordinates) directly to the `voxels` tensor.
5. Pass the perturbed `voxels` into the Voxel Feature Encoder (`self.pts_voxel_encoder`) and proceed with the training step.

## 4. Current Status & Next Steps
* **Phase 2 (Completed):** Standalone demo [mmdet3d_attack_demo.py](file:///beegfs/jung/mmdet3d_legacy/projects/adv_aug/attacks/mmdet3d_attack_demo.py) is implemented. It validates that voxel-level gradients successfully flow to 100% of active voxels and that a 10% voxel detachment/zeroing-out attack successfully increases the training loss from `3.2167` to `4.2415`.
* **Phase 3 (Next Step):** Implement the training wrapper `AdversarialAugmentationWrapper` or override the detector's `train_step`/`extract_pts_feat` to perform this voxel-level attack dynamically during the optimizer loop.

