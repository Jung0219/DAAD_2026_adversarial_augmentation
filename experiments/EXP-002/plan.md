# Research Experiment Specification Template

---

## 1. Overview
* **Experiment ID:** EXP-002
* **Title:** Iterative Adversarial Point Cloud Generation (PointPillars)
* **Objective/Hypothesis:** Evaluate point attachment, detachment, and perturbation attacks up to 30 iterations (in increments of 5) against the PointPillars baseline. Goal is to generate sample point clouds (`.bin`) and measure the average loss/confidence drop across a subset of 10 validation frames to observe effectiveness.

## 2. Code Implementation & Setup
* **Target Files:** `tools/attack_pointpillars_generation.py` (New script to be created)
* **Changes Required:** 
  Implement an iterative attack script that loads a NuScenes mini sample and applies three types of attacks:
  - [x] **Perturbation**: Use PGD to perturb existing points. Voxel gradients will be mapped back to continuous points by appending a 5th point feature (original point index) before `mmcv.ops.Voxelization`.
  - [x] **Detachment**: At each iteration, calculate the $L_2$ norm of the gradient for each point and remove the top 50 points with the highest gradients (up to 1500 points removed at iteration 30).
  - [x] **Attachment**: Initialize 50 new random noise points inside the 3D ground truth bounding boxes per iteration, and optimize their positions using PGD gradients.
  - [x] Run the attack on 10 frames from the validation set. Save the perturbed point clouds at iterations 0, 5, 10, 15, 20, 25, and 30 into a `results/EXP-002/` directory for visual inspection.
  - [x] Track the model's loss and the confidence scores of the predicted bounding boxes directly in the script at each 5-iteration interval.

## 3. Configuration & SLURM Resources
* **Base Config:** `projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_baseline_mini.py`
* **Config Overrides:** None needed for generation script.
* **SLURM Partition:** `gpu`
* **GPU Count (GPUS / GPUS_PER_NODE):** 1
* **CPUs per Task (CPUS_PER_TASK):** 4
* **Estimated Runtime / Time Limit:** 01:00:00

## 4. Execution Commands
```bash
# E.g. running the standalone generation script via SLURM
sbatch projects/adv_aug/scripts/submit/run_exp002_generation.sbatch
```

## 5. Evaluation Settings
Since this is a generation task, the evaluation metrics (loss drop, bounding box confidence drop) will be printed directly by the script as it processes the 10 frames.

## 6. Execution Rules & Guardrails
1. **Codebase Edits:** Edits are allowed in any codebase file as necessary to complete the task or resolve bugs. However, **every single modification must be tracked, explained, and summarized in the final report.**
2. **Reproducibility:** Create a clean git commit of all changes and configs before launching. Record the git commit hash in the report.
3. **SLURM Resource Management:** No hardcoded job limits; submit jobs as needed and let the SLURM scheduler manage execution queueing and runtimes.
4. **Autonomy & Debugging:** If the job fails:
   * Inspect logs in the work directory.
   * Modify configs or code to resolve the error.
   * Commit changes with prefix `[debug-EXP-XXX]`.
   * **Retry up to 5 times** before pausing and alerting the user.
5. **Report Generation & Storage:**
   * Write the final report to `experiments/reports/EXP-002.md`.
   * Store generated sample paths or logs in the report.
