# Research Plan: Runtime Adversarial Augmentation in MMDetection3D

This document tracks the project plan for adding runtime adversarial augmentation to
LiDAR-based 3D object detection in legacy MMDetection3D. It reflects the current
working plan and status, not completed experimental results.

---

## 1. Project Objective

The long-term goal is to test whether gradient-guided, realism-constrained point
cloud or voxel modifications can improve detector robustness against occlusion,
sparsity, and local blockage without causing unacceptable clean-performance loss
on metrics such as mAP and NDS.

This project is intentionally staged. The current priority is not adversarial
training yet. The first priority is to verify that the baseline models train and
test correctly on regular data.

---

## 2. Target Environment and Models

* **Codebase:** Legacy MMDetection3D v0.18.1 at
  `/home/btq48260/DAAD_2026_adversarial_augmentation`.
* **Conda environment:** `AA_legacy` with Python 3.8, PyTorch 1.12.1, and CUDA
  11.3.
* **Dataset:** nuScenes, using small smoke runs first and larger benchmarks only
  after setup is stable.

Target models:

| Model | Role | Current status |
| --- | --- | --- |
| PointPillars | Primary baseline and first model to inspect | Pending smoke validation |
| CenterPoint | Secondary baseline | Pending smoke validation |
| FocalFormer3D | External/custom architecture target | Pending smoke validation |
| PillarNeSt | External/custom architecture target | Pending smoke validation |

---

## 3. MMDetection3D Pipeline Areas to Understand

For normal, non-adversarial augmentation, MMDetection3D is mostly config-driven:

```text
config.py
  data.train.pipeline
      -> Dataset __getitem__
      -> CPU pipeline transforms
      -> collate / scatter
      -> Detector.train_step()
      -> Detector.forward_train()
      -> loss / optimizer step
```

Regular dataloader transforms are expected to be broadly transferable across
models that consume the same dataset format and task inputs. This includes common
operations such as loading points, loading 3D annotations, random flip, rotation,
scaling, point filtering, object filtering, point shuffle, formatting, and
collection.

Runtime adversarial augmentation is different. Because it depends on model loss
gradients, it cannot be implemented only as a CPU dataloader transform. The
project must identify where the dataloader output enters the model and where a
gradient-informed modification can be injected.

Key areas to study during Phase 2:

* `data.train.pipeline` and the final batch keys it produces.
* `train_step`, which receives the collated batch during training.
* `forward_train`, which computes detector training losses.
* `extract_pts_feat`, or equivalent model-specific point feature extraction.
* `voxelize`, `pts_voxel_encoder`, `pts_middle_encoder`, `pts_backbone`,
  `pts_neck`, and the detection head path.

The working assumption is that a shared dataloader pipeline can be reused across
the models, but gradient handling may need model-specific adapters or hook points.
PointPillars and CenterPoint are expected to be more similar. FocalFormer3D and
PillarNeSt may require extra inspection because they use custom project/plugin
code.

---

## 4. Three-Phase Roadmap

### Phase 1: Setup and Baseline Model Validation

**Status:** Completed.

Goal: confirm that all four models can train and test on regular, non-adversarial
data before any adversarial augmentation work is introduced.

Validation depth: smoke runs only. A smoke run should be long enough to catch
configuration, import, checkpoint, CUDA, data pipeline, plugin registration, and
train/test execution errors. Full benchmark performance is not required in this
phase.

Tracking table:

| Model | Train smoke test | Test smoke test | Notes |
| --- | --- | --- | --- |
| PointPillars | Passed | Passed | Checked via standard tools and attacks/mmdet3d_attack_demo.py |
| CenterPoint | Passed | Passed | Checked via tools/train.py and tools/test.py |
| FocalFormer3D | Passed | Passed | Checked via projects/adv_aug/FocalFormer3D/tools/test.py (handles plugin compilation/registration) |
| PillarNeSt | Passed | Passed | Checked via tools/train.py and tools/test.py (plugins auto-imported via mmdet3d/models/__init__.py) |


Exit criteria:

* Each model has a documented train smoke-test result.
* Each model has a documented test smoke-test result.
* Any failures are recorded with the failing command, config, checkpoint, and
  observed error.
* No adversarial augmentation code is introduced during this phase.

### Phase 2: Pipeline Familiarization and Minimal Gradient Sample

**Status:** Not started.

Goal: learn where MMDetection3D should be customized by running a small,
controlled gradient-informed sample through the existing dataloader and model
pipeline.

This phase is not the actual adversarial augmentation experiment. It is a focused
probing phase to understand the ecosystem:

* Use the regular config-driven dataloader pipeline.
* Fetch and collate a normal sample or small batch.
* Run the batch through the model.
* Identify exactly where CPU dataloader transforms end.
* Identify where GPU/model-side processing begins.
* Inspect voxelization and point feature extraction.
* Compute a simple gradient signal.
* Generate a simple gradient-informed modified sample, even if the modification
  is crude and not research-grade.

Questions this phase should answer:

* Which batch keys are shared across the four models?
* Where can gradients actually be observed?
* Is the useful gradient at raw point level, voxel level, feature level, or a
  model-specific intermediate tensor?
* Which hook point can be shared across models?
* Which models require custom handling?

Expected tracking table:

| Model | Dataloader inspected | Gradient sample generated | Hook point understood | Notes |
| --- | --- | --- | --- | --- |
| PointPillars | Pending | Pending | Pending | - |
| CenterPoint | Pending | Pending | Pending | - |
| FocalFormer3D | Pending | Pending | Pending | - |
| PillarNeSt | Pending | Pending | Pending | - |

### Phase 3: Adversarial Augmentation Experiments

**Status:** Not started.

Goal: implement and compare actual adversarial augmentation strategies after the
baseline setup and pipeline familiarization phases are complete.

Potential experiment directions:

* Voxel or point detachment/masking.
* Gradient-ranked feature zeroing.
* Coordinate or feature shifting.
* Clean/adversarial loss blending.
* Different perturbation ratios, schedules, and attack frequencies.
* Shared attack logic with model-specific adapters where needed.

Evaluation should compare:

* Clean validation performance.
* Robustness under synthetic perturbations such as occlusion or sparsification.
* Robustness under direct white-box or gradient-guided attacks.
* Tradeoffs between robustness gain and clean mAP/NDS degradation.

---

## 5. Current Decisions

1. The active project phase is **Phase 2: Pipeline Familiarization and Minimal Gradient Sample**.
2. All baseline models (PointPillars, CenterPoint, FocalFormer3D, PillarNeSt) have been successfully smoke-tested on standard training and evaluation pipelines.
3. The existing exploratory attack scripts are not counted as completed project
   milestones.
4. Regular augmentation belongs primarily in `data.train.pipeline`.
5. Gradient-guided adversarial augmentation will likely require customization
   inside the model training path, not only the dataloader.
6. The implementation phase can proceed as the four baseline model paths are
   smoke-tested on regular data.

---

## 6. Immediate Next Step

Begin Phase 2 pipeline familiarization by understanding the voxelization autograd boundary and implementing voxel-level gradient-informed perturbations inside the models' training loop (e.g., implementing/overriding `AdversarialAugmentationWrapper` or detector `train_step`).

---

## 7. Baseline Evaluation Metrics

Full nuScenes validation metrics on 6,019 val samples (see `projects/adv_aug/docs/evaluation_report.md` for full per-class breakdown):

| Model | mAP | NDS | Notes |
|-------|----:|----:|-------|
| PointPillars | 39.83% | 53.01% | Lowest baseline, expected for simpler architecture |
| CenterPoint | 56.93% | 65.22% | Strong conventional baseline |
| PillarNeSt | 58.66% | 65.14% | Slightly higher mAP than CenterPoint, similar NDS |
| FocalFormer3D | 66.02% | 70.64% | Best baseline across all listed classes |

---

## 8. Phase 2 Completion Tracker

**Completed so far:**
- Phase 1 baseline train/test smoke validation for all four models.
- Baseline validation report on 6,019 nuScenes validation samples.
- Standalone PointPillars voxel-gradient proof of concept in `projects/adv_aug/attacks/mmdet3d_attack_demo.py`.

**Not yet completed:**
- A reusable adversarial training wrapper.
- A detector-level `train_step` or `extract_pts_feat` override for runtime adversarial augmentation.
- Shared adapters for all four model families.
- Clean/adversarial training schedules or robustness benchmarks.

---

## 9. Attack Code Status

| File | Status | Notes |
|------|--------|-------|
| `projects/adv_aug/attacks/mmdet3d_attack_demo.py` | ✅ Working prototype | MMDet3D-native; PointPillars only; voxel-gradient zeroing |
| `projects/adv_aug/attacks/attack.py` | ⚠️ Not compatible | Copied from IJCV paper; OpenPCDet-based API |
| `projects/adv_aug/attacks/attach.py` | ⚠️ Not compatible | OpenPCDet-based |
| `projects/adv_aug/attacks/detach.py` | ⚠️ Not compatible | OpenPCDet-based |

When adapting `attack.py`, `attach.py`, or `detach.py`, account for the OpenPCDet-to-MMDetection3D API mismatch — do not copy logic blindly.
