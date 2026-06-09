# Models, Checkpoints & Runnable Scripts

## 1. Target Models, Configs, and Checkpoints

| Model | Config | Checkpoint |
|-------|--------|------------|
| PointPillars | `projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py` | `projects/adv_aug/checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth` |
| CenterPoint | `projects/adv_aug/configs/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus.py` | `projects/adv_aug/checkpoints/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20200930_201619-67c8496f.pth` |
| FocalFormer3D | `projects/adv_aug/configs/focalformer3d/FocalFormer3D_L.py` | `projects/adv_aug/checkpoints/focalformer3d/FocalFormer3D_L.pth` |
| PillarNeSt | `projects/adv_aug/configs/pillarnest/pillarnest_tiny.py` | `projects/adv_aug/checkpoints/pillarnest/pillarnest_tiny.pth` |

Project configs use `custom_imports = dict(imports=['projects.adv_aug.plugins'], allow_failed_imports=False)` where needed. The default runtime also sets this globally.

---

## 2. Train / Test (General)

```bash
# Generic SLURM launcher
sbatch projects/adv_aug/scripts/submit/run_experiment.sbatch train pointpillars
sbatch projects/adv_aug/scripts/submit/run_experiment.sbatch test  focalformer3d
```

Accepted models: `pointpillars`, `centerpoint`, `focalformer3d`, `pillarnest`
Accepted actions: `train`, `test`

```bash
# Direct (non-SLURM) training
PYTHONPATH=. python tools/train.py <CONFIG> --work-dir <WORK_DIR>

# Direct (non-SLURM) testing
PYTHONPATH=. python tools/test.py <CONFIG> <CHECKPOINT> --eval bbox
```
