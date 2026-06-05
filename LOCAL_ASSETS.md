# Local Assets Required After Cloning

This repository intentionally does not track datasets, model weights, generated
pickle files, compiled extensions, logs, or runtime outputs. After cloning the
repo on a new server, recreate the local assets below.

## Datasets

Place datasets under the repo `data/` directory.

### nuScenes

Expected root:

```text
data/nuscenes/
```

Required for the current Phase 1 smoke tests:

```text
data/nuscenes/samples/
data/nuscenes/sweeps/
data/nuscenes/maps/
data/nuscenes/v1.0-mini/
data/nuscenes/v1.0-trainval/
data/nuscenes/nuscenes_infos_train.pkl
data/nuscenes/nuscenes_infos_val.pkl
```

Current local source on this server:

```text
/beegfs/jung/mmdet3d_legacy/data/nuscenes/
```

### KITTI

Archive source on this server:

```text
/beegfs/chandorkar/kai_data/kitti.tar
```

Expected root after extraction:

```text
data/kitti/
```

Expected structure:

```text
data/kitti/ImageSets/
data/kitti/training/calib/
data/kitti/training/image_2/
data/kitti/training/label_2/
data/kitti/training/velodyne/
data/kitti/testing/calib/
data/kitti/testing/image_2/
data/kitti/testing/velodyne/
```

If processed files are missing, run from the repo root:

```bash
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

### Waymo

Archive source on this server:

```text
/beegfs/chandorkar/kai_data/waymo.tar
```

Expected root after extraction:

```text
data/waymo/
```

Expected raw/converted structure:

```text
data/waymo/waymo_format/training/
data/waymo/waymo_format/validation/
data/waymo/waymo_format/testing/
data/waymo/waymo_format/gt.bin
data/waymo/kitti_format/ImageSets/
```

If converted files are missing, run from the repo root:

```bash
python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 128 --extra-tag waymo
```

## Model Weights

Model weights are ignored by Git. Restore them manually into the same paths.

### PointPillars

```text
projects/adv_aug/PointPillars/checkpoints/
```

Known local files:

```text
hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth
hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth
```

### CenterPoint

```text
projects/adv_aug/CenterPoint/checkpoints/
```

Known local files:

```text
centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f.pth
centerpoint_01voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_052355-a6928835.pth
```

### FocalFormer3D

```text
projects/adv_aug/FocalFormer3D/checkpoints/
```

Known local files:

```text
DeformFormer3D_C_R50.pth
DeformFormer3D_L.pth
FocalFormer3D_L.pth
FocalFormer3D_LC.pth
r50_fpn_voxel_0075.pth
```

### PillarNeSt

```text
projects/adv_aug/PillarNeSt/checkpoints/PillarNeSt/
```

Known local files:

```text
pillarnest_base.pth
pillarnest_large.pth
pillarnest_small.pth
pillarnest_tiny.pth
```

## Generated Evaluation Artifacts

These are ignored and should be regenerated or copied only when needed:

```text
projects/adv_aug/custom_eval/generated.pkl
projects/adv_aug/custom_eval/own_mAP_data.pickle
projects/adv_aug/custom_eval/visualizations/
```

## Compiled Extensions

Compiled CUDA/C++ artifacts are ignored. They should be rebuilt on each server.

Common generated artifacts include:

```text
*.so
*.o
*.cuda.o
.ninja_deps
.ninja_log
build.ninja
```

FocalFormer3D custom ops may generate artifacts under:

```text
projects/adv_aug/FocalFormer3D/projects/mmdet3d_plugin/models/utils/ops/
```

## Runtime Outputs

Do not restore these unless debugging a previous run:

```text
work_dirs/
*.out
*.err
*.log
*.log.json
```

SLURM jobs and MMDetection3D training/testing will recreate them as needed.
