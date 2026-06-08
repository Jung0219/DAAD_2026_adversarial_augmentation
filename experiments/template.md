# Research Experiment Specification Template

Use this template to define an experiment task for the AI assistant. Copy this template, save it as `experiments/experiment_N.md` under the `experiments/` directory, fill in the details, and prompt the agent:
*"Run the experiment described in [experiments/experiment_1.md](file:///home/btq48260/DAAD_2026_adversarial_augmentation/experiments/experiment_1.md)."*

---

## 1. Overview
* **Experiment ID:** [e.g., EXP-001]
* **Title:** [Brief descriptive title]
* **Objective/Hypothesis:** [What research question are we answering? E.g., Testing adversarial augmentation strength on NuScenes dataset]

## 2. Code Implementation & Setup
* **Target Files:** [Which files or modules need modification? E.g., `mmdet3d/datasets/pipelines/transforms_3d.py`]
* **Changes Required:** 
  Describe the logic, math, or augmentation you want implemented. Note: The agent may modify files outside this list if necessary to make the experiment run, but all modified files must be documented.
  - [ ] Detail 1: ...
  - [ ] Detail 2: ...

## 3. Configuration & SLURM Resources
* **Base Config:** [Path to base config file, e.g., `configs/nuscenes/det3d_res50.py`]
* **Config Overrides:** [List any hyperparameters to change in the config]
* **SLURM Partition:** [Default: gpu]
* **GPU Count (GPUS / GPUS_PER_NODE):** [Default: 8]
* **CPUs per Task (CPUS_PER_TASK):** [Default: 5]
* **Estimated Runtime / Time Limit:** [e.g., 24:00:00 (SLURM queue scheduling will manage limits)]

## 4. Execution Commands
State the exact script and arguments to submit.
```bash
# E.g. submitting training via tools/slurm_train.sh
./tools/slurm_train.sh <PARTITION> <JOB_NAME> <CONFIG> <WORK_DIR>
```

## 5. Evaluation Settings
Specify the test or validation script, weights to evaluate, and evaluation metrics:
```bash
# E.g. evaluating the trained checkpoint
./tools/slurm_test.sh <PARTITION> <JOB_NAME> <CONFIG> <CHECKPOINT> --eval bbox
```

## 6. Execution Rules & Guardrails
1. **Codebase Edits:** Edits are allowed in any codebase file as necessary to complete the task or resolve bugs. However, **every single modification must be tracked, explained, and summarized in the final report.**
2. **Reproducibility:** Create a clean git commit of all changes and configs before launching. Record the git commit hash in the report.
3. **SLURM Resource Management:** No hardcoded job limits; submit jobs as needed and let the SLURM scheduler manage execution queueing and runtimes.
4. **Autonomy & Debugging:** If the job fails (OOM, NaN loss, coding bugs, cluster/SLURM failure):
   * Inspect logs in the work directory.
   * Modify configs or code to resolve the error (e.g., lower batch size, add gradient clipping, fix syntax).
   * Commit changes with prefix `[debug-EXP-XXX]`.
   * **Retry up to 5 times** before pausing and alerting the user.
5. **Report Generation & Storage:**
   * Write the final report to `experiments/reports/EXP-XXX.md` (where `EXP-XXX` is the Experiment ID).
   * Store any report assets (plots, curves, log snippets) under `experiments/reports/assets/`.
   * Generate an agent chat artifact containing the same report for easy user reading.
   * **Report Contents MUST include:**
     - Experiment ID & Title
     - Git commit hash of the implementation
     - List of modified files with explanations/diffs
     - Configuration overrides
     - Training logs & visualizations (loss curves, learning rate)
     - Evaluation metrics (mAP, NDS, class-specific metrics)
     - A detailed conclusion and recommended next steps
