# Data TODO

## KITTI

- [ ] Extract `/beegfs/chandorkar/kai_data/kitti.tar` under `/beegfs/jung/mmdet3d_legacy/data/`.
- [ ] Confirm the final root is `/beegfs/jung/mmdet3d_legacy/data/kitti/`.
- [ ] Confirm the expected raw structure exists:
  - `data/kitti/ImageSets/`
  - `data/kitti/training/calib/`
  - `data/kitti/training/image_2/`
  - `data/kitti/training/label_2/`
  - `data/kitti/training/velodyne/`
  - `data/kitti/testing/calib/`
  - `data/kitti/testing/image_2/`
  - `data/kitti/testing/velodyne/`
- [ ] If needed, run:
  ```bash
  python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
  ```
- [ ] Confirm generated files exist:
  - `data/kitti/kitti_infos_train.pkl`
  - `data/kitti/kitti_infos_val.pkl`
  - `data/kitti/kitti_infos_test.pkl`
  - `data/kitti/kitti_infos_trainval.pkl`
  - `data/kitti/kitti_dbinfos_train.pkl`
  - `data/kitti/kitti_gt_database/`

## Waymo

- [ ] Extract `/beegfs/chandorkar/kai_data/waymo.tar` under `/beegfs/jung/mmdet3d_legacy/data/`.
- [ ] Confirm the final root is `/beegfs/jung/mmdet3d_legacy/data/waymo/`.
- [ ] Confirm the expected raw structure exists:
  - `data/waymo/waymo_format/training/`
  - `data/waymo/waymo_format/validation/`
  - `data/waymo/waymo_format/testing/`
  - `data/waymo/waymo_format/gt.bin`
  - `data/waymo/kitti_format/ImageSets/`
- [ ] If needed, run:
  ```bash
  python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 128 --extra-tag waymo
  ```
- [ ] Confirm generated files exist:
  - `data/waymo/kitti_format/waymo_infos_train.pkl`
  - `data/waymo/kitti_format/waymo_infos_val.pkl`
  - `data/waymo/kitti_format/waymo_infos_test.pkl`
  - `data/waymo/kitti_format/waymo_infos_trainval.pkl`
  - `data/waymo/kitti_format/waymo_dbinfos_train.pkl`
  - `data/waymo/kitti_format/waymo_gt_database/`
