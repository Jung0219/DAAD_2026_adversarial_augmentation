# Codebase Architecture Reference

## 1. Full Repository Structure

```
DAAD_2026_adversarial_augmentation/          ← Repo root
├── configs/                                 ← MMDetection3D standard configs
│   ├── _base_/
│   │   ├── datasets/
│   │   │   ├── nus-3d.py                    ← NuScenes 3D detection dataset config
│   │   │   ├── nus-mono3d.py
│   │   │   ├── kitti-3d-3class.py
│   │   │   ├── kitti-3d-car.py
│   │   │   ├── s3dis-seg.py
│   │   │   ├── scannet-3d-18class.py
│   │   │   ├── scannet-seg.py
│   │   │   └── sunrgbd-3d-10class.py
│   │   ├── models/
│   │   │   ├── hv_pointpillars_fpn_nus.py   ← PointPillars FPN model (NuScenes)
│   │   │   ├── hv_pointpillars_secfpn_kitti.py
│   │   │   ├── hv_pointpillars_fpn_lyft.py
│   │   │   ├── centerpoint01_second_secfpn_nus.py
│   │   │   └── ... (other base models)
│   │   ├── schedules/
│   │   │   ├── schedule_2x.py               ← 24-epoch schedule, AdamW, lr=0.001
│   │   │   ├── cyclic_20e.py
│   │   │   ├── schedule_3x.py
│   │   │   └── seg_cosine_50e.py
│   │   └── default_runtime.py               ← Logging, checkpointing, custom_imports
│   ├── pointpillars/                        ← 14 PointPillars config variants
│   ├── centerpoint/
│   ├── second/
│   └── ... (other model configs)
│
├── mmdet3d/                                 ← Core framework code
│   ├── __init__.py                          ← Version checks (mmcv ≤ 1.7.1)
│   ├── datasets/
│   │   ├── nuscenes_dataset.py              ← NuScenesDataset class
│   │   └── pipelines/
│   │       ├── transforms_3d.py             ← Main 3D augmentations (1426 lines)
│   │       ├── loading.py                   ← Point cloud & annotation loaders
│   │       ├── formating.py                 ← Batch formatting (DefaultFormatBundle3D)
│   │       ├── dbsampler.py                 ← GT database sampling
│   │       ├── data_augment_utils.py        ← Noise/augmentation utilities
│   │       └── test_aug.py                  ← TTA (MultiScaleFlipAug3D)
│   └── models/
│       ├── detectors/
│       │   ├── mvx_two_stage.py             ← MVXTwoStageDetector (base for PP, CP)
│       │   ├── voxelnet.py                  ← VoxelNet (single-stage)
│       │   └── centerpoint.py
│       ├── voxel_encoders/
│       │   ├── voxel_encoder.py             ← HardVFE (used by PointPillars FPN)
│       │   └── pillar_encoder.py            ← PillarFeatureNet (used by KITTI PP)
│       ├── middle_encoders/
│       │   ├── pillar_scatter.py            ← PointPillarsScatter
│       │   └── sparse_encoder.py
│       ├── backbones/
│       │   └── second.py                    ← SECOND backbone
│       ├── necks/
│       │   └── second_fpn.py               ← SECONDFPN neck
│       └── dense_heads/
│           ├── anchor3d_head.py             ← Anchor3DHead (PointPillars/SECOND)
│           └── centerpoint_head.py
│
├── tools/
│   ├── train.py                             ← Main training entry point
│   ├── test.py                              ← Main test entry point
│   ├── slurm_train.sh                       ← Generic SLURM training launcher
│   ├── slurm_test.sh                        ← Generic SLURM test launcher
│   └── data_converter/
│       └── nuscenes_converter.py            ← NuScenes data preparation
│
├── projects/adv_aug/                        ← *** This project ***
│   ├── attacks/
│   │   ├── mmdet3d_attack_demo.py           ← Native MMDet3D attack prototype
│   │   ├── attack.py                        ← OpenPCDet-based (not mmdet3d-native)
│   │   ├── attach.py                        ← OpenPCDet-based
│   │   └── detach.py                        ← OpenPCDet-based
│   ├── checkpoints/
│   │   ├── pointpillars/
│   │   ├── centerpoint/
│   │   ├── focalformer3d/
│   │   └── pillarnest/
│   ├── configs/
│   │   ├── _base_ → symlink              ← Symlink to ../../configs/_base_
│   │   ├── pointpillars/
│   │   ├── centerpoint/
│   │   ├── focalformer3d/
│   │   └── pillarnest/
│   ├── docs/
│   │   └── evaluation_report.md             ← Baseline validation results
│   ├── plugins/
│   │   ├── __init__.py                      ← Plugin registry
│   │   ├── core/                            ← BBox assigners/coders, hooks
│   │   ├── datasets/pipelines/              ← Multi-view image transforms
│   │   ├── models/backbones/                ← ConvNeXt_PC
│   │   ├── models/dense_heads/              ← FocalDecoder, CenterPlusHead
│   │   ├── models/detectors/                ← FocalFormer3D
│   │   ├── models/necks/                    ← FocalEncoder
│   │   ├── models/utils/                    ← Custom CUDA ops (locatt_ops)
│   │   └── models/voxel_encoders/           ← HeightPillarFeatureNet
│   ├── runs/                                ← Output directory for experiments
│   └── scripts/
│       ├── submit/
│       │   ├── run_experiment.sbatch        ← Main SLURM job script
│       │   ├── run_focalformer3d_predictions.sbatch
│       │   ├── visualize.sbatch
│       │   └── visualize_focalformer3d_stride100.sbatch
│       ├── tools/
│       ├── visualize.py
│       └── start_visualization_port_forward.sh
│
├── docs/                                    ← Project documentation
│   ├── architecture.md                      ← THIS FILE
│   ├── environment.md                       ← Env vars, conda, SLURM config
│   ├── scripts.md                           ← Models, checkpoints, runnable commands
│   └── adversarial_augmentation_plan.md     ← Research phase tracking
│
└── experiments/
    ├── experiment_guide.md                  ← Experiment workflow SOP
    └── template.md                          ← Blank experiment spec
```

---

## 2. PointPillars Architecture — Full Forward Path

This is the primary model for adversarial augmentation development.

```
Raw Points → Voxelization (@torch.no_grad) → HardVFE → PointPillarsScatter → SECOND → FPN → Anchor3DHead
```

### Component Details

| Stage | Module | Type | Key Parameters | Source File |
|-------|--------|------|----------------|-------------|
| Voxelization | `pts_voxel_layer` | HardVoxelization | voxel_size=[0.25,0.25,8], max_points=64, max_voxels=(30k,40k), range=[-50,-50,-5,50,50,3] | `mmdet3d/models/detectors/mvx_two_stage.py` L211 |
| Voxel Encoder | `pts_voxel_encoder` | `HardVFE` | in_ch=4, feat_ch=[64,64], with_cluster_center, with_voxel_center | `mmdet3d/models/voxel_encoders/voxel_encoder.py` |
| Middle Encoder | `pts_middle_encoder` | `PointPillarsScatter` | in_ch=64, output_shape=[400,400] → dense pseudo-image | `mmdet3d/models/middle_encoders/pillar_scatter.py` |
| Backbone | `pts_backbone` | `SECOND` | 3 stages: layers=[3,5,5], strides=[2,2,2], out_ch=[64,128,256] | `mmdet3d/models/backbones/second.py` |
| Neck | `pts_neck` | `FPN` | in_ch=[64,128,256] → out_ch=256, 3 levels | mmdet (external) |
| Head | `pts_bbox_head` | `Anchor3DHead` | 10 classes, in_ch=256, FocalLoss + SmoothL1 + CrossEntropy | `mmdet3d/models/dense_heads/anchor3d_head.py` |

### Detector Class Hierarchy
```
MVXFasterRCNN → MVXTwoStageDetector → Base3DDetector
```

### Key Methods in `MVXTwoStageDetector` (`mvx_two_stage.py`)
- `voxelize(points)` (L211) — **decorated `@torch.no_grad()`**, returns `(voxels, num_points, coors)`
- `extract_pts_feat(points)` (L191) — calls voxelize → encoder chain → backbone → neck
- `forward_pts_train(pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas)` — head loss
- `forward_train(...)` — full training forward (calls extract_feat + forward_pts_train)
- `train_step(data, optimizer)` — called by runner; calls forward_train + parse_losses

### SecFPN Variant
Uses `SECONDFPN` neck: in_ch=[64,128,256], upsample_strides=[1,2,4], out_ch=[128,128,128] → concatenated 384ch → head in_ch=384.

---

## 3. Training Data Pipeline

### NuScenes Train Pipeline (from `configs/_base_/datasets/nus-3d.py`)
```python
train_pipeline = [
    LoadPointsFromFile(coord_type='LIDAR', load_dim=5, use_dim=5),
    LoadPointsFromMultiSweeps(sweeps_num=10),
    LoadAnnotations3D(with_bbox_3d=True, with_label_3d=True),
    GlobalRotScaleTrans(rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    RandomFlip3D(flip_ratio_bev_horizontal=0.5),
    PointsRangeFilter(point_cloud_range),
    ObjectRangeFilter(point_cloud_range),
    ObjectNameFilter(classes),
    PointShuffle(),
    DefaultFormatBundle3D(class_names),
    Collect3D(keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

### Registered Pipeline Transforms (`mmdet3d/datasets/pipelines/__init__.py`)
- **Loading**: `LoadPointsFromFile`, `LoadPointsFromMultiSweeps`, `LoadAnnotations3D`, `LoadMultiViewImageFromFiles`, `NormalizePointsColor`, `PointSegClassMapping`
- **Transforms**: `ObjectSample`, `RandomFlip3D`, `ObjectNoise`, `GlobalRotScaleTrans`, `PointShuffle`, `ObjectRangeFilter`, `PointsRangeFilter`, `PointSample`, `IndoorPointSample`, `BackgroundPointsFilter`, `VoxelBasedPointSampler`, `GlobalAlignment`, `RandomDropPointsColor`, `RandomJitterPoints`, `ObjectNameFilter`
- **Formatting**: `DefaultFormatBundle`, `DefaultFormatBundle3D`, `Collect3D`
- **TTA**: `MultiScaleFlipAug3D`
- **DB Sampler**: `DataBaseSampler`

### Key Transform Source Files
| File | Lines | Purpose |
|------|-------|---------|
| `mmdet3d/datasets/pipelines/transforms_3d.py` | 1426 | Main 3D augmentations |
| `mmdet3d/datasets/pipelines/loading.py` | ~700 | Data loading transforms |
| `mmdet3d/datasets/pipelines/formating.py` | ~300 | Batch formatting |
| `mmdet3d/datasets/pipelines/dbsampler.py` | ~300 | GT database sampling |
| `mmdet3d/datasets/pipelines/data_augment_utils.py` | ~450 | Noise/augmentation utilities |

---

## 4. NuScenes Dataset Configuration

| Property | Value |
|----------|-------|
| Dataset type | `NuScenesDataset` |
| Data root | `data/nuscenes/` |
| Classes (10) | car, truck, trailer, bus, construction_vehicle, bicycle, motorcycle, pedestrian, traffic_cone, barrier |
| Point cloud range | [-50, -50, -5, 50, 50, 3] |
| Train ann_file | `data/nuscenes/nuscenes_infos_train.pkl` (1.47 GB, ~28k samples) |
| Val ann_file | `data/nuscenes/nuscenes_infos_val.pkl` (310 MB, ~6k samples) |
| samples_per_gpu | 4 |
| workers_per_gpu | 4 |

### NuScenes Mini
- Metadata at `data/nuscenes/v1.0-mini/`
- No pre-existing mini info pkl file — **must be generated** via `tools/create_data.py` or `tools/data_converter/nuscenes_converter.py` with `--version v1.0-mini`
- Mini split: ~10 scenes, ~404 samples (train ~323, val ~81)

---

## 5. Key Architectural Finding: Voxelization Autograd Boundary

MMDetection3D-style detectors inherit from `MVXTwoStageDetector` and voxelize
points inside methods decorated with `@torch.no_grad()` / `@force_fp32()`. The
project FocalFormer3D detector also defines `voxelize` and `dynamic_voxelize`
with `@torch.no_grad()`.

**Consequence:** gradients to the raw `points` list are cut by voxelization and/or
non-differentiable CUDA/C++ ops. A raw-point adversarial update cannot simply set
`points.requires_grad = True` and expect useful gradients.

### Working Solution Direction

1. Run normal dataloader and collate/scatter.
2. Call the model voxelization path.
3. Set `voxels.requires_grad = True` on voxelization output.
4. Continue through `pts_voxel_encoder`, `pts_middle_encoder`, `pts_backbone`, `pts_neck`, and the detection head.
5. Backpropagate the training loss with graph retention where needed.
6. Read `voxels.grad` and compute a gradient ranking.
7. Apply a perturbation (voxel/point zeroing, detachment, or coordinate shifting) to a cloned/intermediate voxel tensor.
8. Re-run the relevant forward path and combine clean/adversarial losses only after the training design is explicit.

For FocalFormer3D, inspect both static and dynamic voxelization paths:
`FocalFormer3D.extract_pts_feat` switches based on
`self.apply_dynamic_voxelize = 'Dynamic' in pts_voxel_encoder['type']`.

---

## 6. Plugin Registry

`projects/adv_aug/plugins/__init__.py` registers the following custom modules:

| Category | Modules |
|----------|---------|
| Backbones | `ConvNeXt_PC` |
| Necks | `FocalEncoder` |
| Heads | `FocalDecoder`, `CenterPlusHead` |
| Voxel Encoders | `HeightPillarFeatureNet` |
| Detectors | `FocalFormer3D` |
| BBox Assigners | `HungarianAssigner3D`, `HeuristicAssigner3D` |
| BBox Coders | `TransFusionBBoxCoder`, `CenterPointBBoxCoder` |
| Hooks | `Fading` |

These are loaded automatically via `custom_imports` in `default_runtime.py`.
Any new config inheriting from `default_runtime.py` does **not** need a separate
`custom_imports` declaration unless using additional plugin paths.

---

## 7. Config Inheritance Chains

### PointPillars (NuScenes FPN)
```
projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py
  └── configs/_base_/models/hv_pointpillars_fpn_nus.py      ← model arch
  └── configs/_base_/datasets/nus-3d.py                      ← dataset + pipeline
  └── configs/_base_/schedules/schedule_2x.py                ← optimizer + lr + epochs
  └── configs/_base_/default_runtime.py                      ← logging + plugins
```

### CenterPoint
```
projects/adv_aug/configs/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py
  └── configs/_base_/default_runtime.py
  (model, dataset, schedule defined inline)
```

---

## 8. Key File Quick Reference

| Purpose | Path |
|---------|------|
| **Training script** | `tools/train.py` |
| **Test script** | `tools/test.py` |
| **SLURM train** | `tools/slurm_train.sh` |
| **Project SLURM** | `projects/adv_aug/scripts/submit/run_experiment.sbatch` |
| **PP base model** | `configs/_base_/models/hv_pointpillars_fpn_nus.py` |
| **NuScenes dataset** | `configs/_base_/datasets/nus-3d.py` |
| **Schedule** | `configs/_base_/schedules/schedule_2x.py` |
| **Runtime** | `configs/_base_/default_runtime.py` |
| **Project PP config** | `projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py` |
| **MVXTwoStageDetector** | `mmdet3d/models/detectors/mvx_two_stage.py` |
| **PointPillarsScatter** | `mmdet3d/models/middle_encoders/pillar_scatter.py` |
| **SECOND backbone** | `mmdet3d/models/backbones/second.py` |
| **Anchor3DHead** | `mmdet3d/models/dense_heads/anchor3d_head.py` |
| **3D transforms** | `mmdet3d/datasets/pipelines/transforms_3d.py` |
| **Pipeline registry** | `mmdet3d/datasets/pipelines/__init__.py` |
| **Attack demo** | `projects/adv_aug/attacks/mmdet3d_attack_demo.py` |
| **Research plan** | `docs/adversarial_augmentation_plan.md` |
| **Eval report** | `projects/adv_aug/docs/evaluation_report.md` |
