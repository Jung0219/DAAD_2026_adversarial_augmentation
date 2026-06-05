"""Run official MMDetection3D CenterPoint model on synthetic point clouds.

Usage:
    python projects/adv_aug/centerpoint/demo_mmdet3d.py
"""

from __future__ import annotations

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

import os
import sys
# Add codebase root to sys.path automatically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import mmdet3d.models
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


def make_random_points(num_points: int, pc_range: list[float], num_features: int = 5) -> torch.Tensor:
    """Generate random points within the configured point cloud range."""
    xyz = torch.empty(num_points, 3)
    # Fill with points within the point cloud range
    xyz[:, 0].uniform_(pc_range[0] + 1.0, pc_range[3] - 1.0)
    xyz[:, 1].uniform_(pc_range[1] + 1.0, pc_range[4] - 1.0)
    xyz[:, 2].uniform_(pc_range[2] + 0.5, pc_range[5] - 0.5)
    
    # Point intensity/reflectance
    intensity = torch.rand(num_points, 1)
    
    if num_features >= 5:
        # For nuScenes, CenterPoint often expects a 5th dimension (timestamp / time lag)
        timestamp = torch.zeros(num_points, 1)
        return torch.cat((xyz, intensity, timestamp), dim=1)
        
    return torch.cat((xyz, intensity), dim=1)


def main() -> None:
    # Initialize the default scope for the registry
    init_default_scope('mmdet3d')

    # 1. Load the official CenterPoint config
    config_path = 'configs/centerpoint/centerpoint_voxel01_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
    print(f"Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)

    # 2. Build the official CenterPoint model from registry
    print("Building model using mmdet3d library registry...")
    model = MODELS.build(cfg.model).eval()
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Model successfully built and moved to {device}")

    # 3. Create dummy point cloud inputs within the point cloud range
    pc_range = cfg.model.data_preprocessor.voxel_layer.point_cloud_range
    print(f"Configured Point Cloud Range: {pc_range}")
    
    # Check expected input features size
    # By default on nuScenes, it expects 5-dim points [x, y, z, intensity, timestamp]
    in_channels = cfg.model.pts_voxel_encoder.num_features
    print(f"Model pts voxel encoder input features: {in_channels}")
    
    points_1 = make_random_points(6000, pc_range, num_features=in_channels).to(device)
    points_2 = make_random_points(4500, pc_range, num_features=in_channels).to(device)
    
    # 4. Prepare data samples for the prediction mode
    data_samples = [
        Det3DDataSample(metainfo=dict(box_type_3d=LiDARInstance3DBoxes)),
        Det3DDataSample(metainfo=dict(box_type_3d=LiDARInstance3DBoxes))
    ]
    
    # MMDetection3D preprocessor expects data containing 'inputs' and 'data_samples'
    data = {
        'inputs': {
            'points': [points_1, points_2]
        },
        'data_samples': data_samples
    }

    # Preprocess the data (collates batches and performs voxelization)
    print("Preprocessing data (voxelizing points)...")
    preprocessed_data = model.data_preprocessor(data, training=False)
    batch_inputs = preprocessed_data['inputs']
    batch_data_samples = preprocessed_data['data_samples']

    # 5. Mode A: Get raw network outputs (Not supported by CenterPoint base class)
    print("\n--- Mode A: Running in 'tensor' mode (Skipped for CenterPoint) ---")

    # 6. Mode B: Get decoded predictions (with NMS and score thresholds)
    print("\n--- Mode B: Running in 'predict' mode (returns decoded boxes) ---")
    with torch.no_grad():
        predictions = model(batch_inputs, data_samples=batch_data_samples, mode='predict')
        
    for idx, sample in enumerate(predictions):
        pred_instances = sample.pred_instances_3d
        print(f"Sample {idx}:")
        print(f"  Number of detected boxes: {len(pred_instances.bboxes_3d)}")
        if len(pred_instances.bboxes_3d) > 0:
            print(f"  bboxes_3d shape: {tuple(pred_instances.bboxes_3d.tensor.shape)}")
            print(f"  scores_3d shape: {tuple(pred_instances.scores_3d.shape)}")
            print(f"  labels_3d shape: {tuple(pred_instances.labels_3d.shape)}")


if __name__ == '__main__':
    main()
