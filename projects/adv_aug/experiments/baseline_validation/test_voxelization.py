import torch
from mmcv.ops import Voxelization
from mmcv import Config
from mmdet3d.datasets import build_dataset

def test_voxelization():
    cfg = Config.fromfile("projects/adv_aug/configs/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py")
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    
    data = dataset[0]
    points = data['points'][0].data.cuda()
    
    print(f"Points shape: {points.shape}")
    print(f"Points min: {points.min(dim=0)[0]}")
    print(f"Points max: {points.max(dim=0)[0]}")
    
    voxel_layer = Voxelization(
        voxel_size=[0.1, 0.1, 0.2],
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num_points=10,
        max_voxels=120000
    )
    
    voxels, coors, num_points = voxel_layer(points)
    
    print(f"Voxels shape: {voxels.shape}")
    print(f"Num points shape: {num_points.shape}")
    print(f"Coors shape: {coors.shape}")
    
    if len(voxels) > 0:
        print(f"Voxels max: {voxels.max().item()}, min: {voxels.min().item()}")
        print(f"Voxels sum: {voxels.sum().item()}")

if __name__ == "__main__":
    test_voxelization()
