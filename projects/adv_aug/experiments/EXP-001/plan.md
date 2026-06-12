# Experiment Specification: EXP-001

## 1. Overview
* **Experiment ID:** EXP-001
* **Title:** Feature-Space PGD Adversarial Augmentation on PointPillars (NuScenes Mini)
* **Objective/Hypothesis:** Verify that a gradient-based adversarial augmentation pipeline can be integrated into the PointPillars training loop as a runtime augmentation strategy without errors. Build intuition for where to modify the training pipeline. This is a **pipeline validation experiment** — metrics are secondary to correctness.

---

## 2. Design Decisions (from interview)

| Decision | Choice |
|----------|--------|
| **Perturbation target** | FPN neck output (256-ch BEV feature maps, right before Anchor3DHead) |
| **Perturbation method** | PGD: `x_adv = x + ε · sign(∇L)`, iterated 5 times |
| **ε (epsilon)** | 8/255 ≈ 0.031 (L∞ budget) |
| **α (step size)** | ε/5 ≈ 0.00627 per PGD step |
| **PGD iterations** | 5 |
| **Loss strategy** | `L = L_clean + λ · L_adv` with λ = 1.0 |
| **Implementation** | `AdvMVXFasterRCNN` subclassing `MVXFasterRCNN`, overriding `forward_train()` |
| **File location** | `projects/adv_aug/plugins/models/detectors/adv_mvx_faster_rcnn.py` |
| **Weight init** | Train from scratch (random init) |
| **Dataset** | NuScenes mini (~323 train, ~81 val) |
| **Epochs** | 5 |
| **Baseline comparison** | Yes — parallel vanilla PointPillars run on same mini data |

---

## 3. Code Implementation & Setup

### New Files to Create
1. **`projects/adv_aug/plugins/models/detectors/adv_mvx_faster_rcnn.py`**
   - `AdvMVXFasterRCNN` class subclassing `MVXFasterRCNN`
   - Overrides `forward_train()` to:
     1. Extract features normally (voxelize → encoder → backbone → neck)
     2. Compute clean loss via `forward_pts_train(clean_feats, ...)`
     3. Run 5-step PGD on the neck output feature maps:
        - Clone neck output, set `requires_grad=True`
        - Forward through head, compute loss, backprop to get gradient
        - Update: `x_adv = x_adv + α · sign(grad)`
        - Clamp: `x_adv = clamp(x_adv, x_clean - ε, x_clean + ε)`
     4. Compute adversarial loss via `forward_pts_train(adv_feats, ...)`
     5. Return combined loss: `L_clean + λ · L_adv`
   - Config parameters: `adv_eps`, `adv_alpha`, `adv_steps`, `adv_lambda`

2. **`projects/adv_aug/plugins/models/detectors/__init__.py`** (update)
   - Register `AdvMVXFasterRCNN` in the plugin `__init__.py`

3. **`projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_adv_mini.py`**
   - Adversarial training config for mini dataset
   - `model.type = 'AdvMVXFasterRCNN'`
   - `model.adv_eps = 0.031`, `model.adv_alpha = 0.00627`, `model.adv_steps = 5`, `model.adv_lambda = 1.0`
   - Override `ann_file` to mini info pkl
   - `samples_per_gpu = 2`, `workers_per_gpu = 2`
   - `runner.max_epochs = 5`

4. **`projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_baseline_mini.py`**
   - Vanilla PointPillars baseline on mini dataset (no adversarial augmentation)
   - Same mini data, same 5 epochs, same batch size

### Data Preparation
- Generate NuScenes mini info pkl files:
  ```bash
  python tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/nuscenes --extra-tag nuscenes --version v1.0-mini
  ```
  This will produce `data/nuscenes/nuscenes_mini_infos_train.pkl` and `data/nuscenes/nuscenes_mini_infos_val.pkl`.

---

## 4. Architecture Diagram

```
┌─────────────────── AdvMVXFasterRCNN.forward_train() ───────────────────┐
│                                                                         │
│  Points ──▶ Voxelize ──▶ HardVFE ──▶ Scatter ──▶ SECOND ──▶ FPN       │
│                                                              │          │
│                                                        neck_feats      │
│                                                        (clean)         │
│                                                         │    │         │
│                              ┌──────────────────────────┘    │         │
│                              ▼                               ▼         │
│                    ┌─── PGD Loop (5 steps) ───┐     Anchor3DHead      │
│                    │  x_adv = clone(neck)      │         │             │
│                    │  x_adv.requires_grad=True │      L_clean          │
│                    │  for i in range(5):        │                      │
│                    │    loss = Head(x_adv)      │                      │
│                    │    grad = ∇loss / ∇x_adv   │                      │
│                    │    x_adv += α·sign(grad)   │                      │
│                    │    clamp to [x-ε, x+ε]     │                      │
│                    └────────────┬───────────────┘                      │
│                                 ▼                                      │
│                          Anchor3DHead                                  │
│                                 │                                      │
│                              L_adv                                     │
│                                                                         │
│  Total Loss = L_clean + λ · L_adv                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Configuration & SLURM Resources

| Parameter | Value |
|-----------|-------|
| **Base Config** | `configs/_base_/models/hv_pointpillars_fpn_nus.py` + `configs/_base_/datasets/nus-3d.py` |
| **SLURM Partition** | `c23g` |
| **GPU Count** | 1 |
| **CPUs per Task** | 4 |
| **Memory** | 32 GB |
| **Time Limit** | 06:00:00 |
| **Optimizer** | AdamW, lr=0.001, weight_decay=0.01 |
| **LR Schedule** | Step decay (steps at epoch 4) for 5-epoch run |
| **Warmup** | Linear, 100 iters (reduced for mini dataset) |
| **Batch size** | 2 per GPU |
| **Workers** | 2 per GPU |
| **Epochs** | 5 |
| **Evaluation interval** | Every 5 epochs (end of training) |

---

## 6. Execution Commands

### Step 1: Generate mini dataset info files
```bash
python tools/create_data.py nuscenes --root-path data/nuscenes --out-dir data/nuscenes --extra-tag nuscenes --version v1.0-mini
```

### Step 2: Train baseline (vanilla PointPillars on mini)
```bash
python tools/train.py \
    projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_baseline_mini.py \
    --work-dir projects/adv_aug/runs/exp001_baseline_mini \
    --no-validate \
    --cfg-options runner.max_epochs=5
```

### Step 3: Train adversarial (AdvMVXFasterRCNN on mini)
```bash
python tools/train.py \
    projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_adv_mini.py \
    --work-dir projects/adv_aug/runs/exp001_adv_mini \
    --no-validate \
    --cfg-options runner.max_epochs=5
```

---

## 7. Evaluation Settings

```bash
python tools/test.py \
    projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_adv_mini.py \
    projects/adv_aug/runs/exp001_adv_mini/latest.pth \
    --eval bbox
```

---

## 8. Success Criteria

This is a pipeline validation experiment. Success means:
- [x] `AdvMVXFasterRCNN` registers and loads correctly
- [x] Training completes all 5 epochs without errors
- [x] Both clean and adversarial losses are computed and decrease over training
- [x] Gradients flow correctly through the PGD loop (no NaN/Inf)
- [x] The adversarial training run produces a valid checkpoint
- [x] Baseline and adversarial training logs can be compared side-by-side

Metrics (mAP, NDS) are **not** expected to be meaningful on 5 epochs of mini data.

---

## 9. Execution Rules & Guardrails
1. **Codebase Edits:** All modified files must be documented in the final report with diffs.
2. **Reproducibility:** Commit all changes before launching. Record git hash.
3. **Auto-Debugging:** Up to 5 retries on failure before alerting user.
4. **Report:** Write to `experiments/reports/EXP-001.md` with training logs, loss curves, and conclusions.
