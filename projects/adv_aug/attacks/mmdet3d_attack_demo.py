import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add codebase root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector

def sum_losses(losses):
    total_loss = 0
    for k, v in losses.items():
        if 'loss' in k:
            if isinstance(v, list):
                total_loss += sum(x.mean() for x in v)
            else:
                total_loss += v.mean()
    return total_loss

def print_losses(losses):
    for k, v in losses.items():
        if 'loss' in k:
            if isinstance(v, list):
                val = sum(x.mean() for x in v).item()
            else:
                val = v.mean().item()
            print(f"  {k}: {val:.4f}")

def main():
    print("=== MMDetection3D Adversarial Attack Prototyping ===")
    
    # 1. Load configuration file
    config_path = 'configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py'
    print(f"Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # Use CPU/GPU settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Build the training dataset to get access to point clouds and annotations
    print("Building dataset...")
    # Point data root to absolute path if needed
    cfg.data.train.data_root = 'data/nuscenes/'
    if 'dataset' in cfg.data.train:
        cfg.data.train.dataset.data_root = 'data/nuscenes/'
    dataset = build_dataset(cfg.data.train)
    print(f"Dataset successfully built. Total samples: {len(dataset)}")
    
    # 3. Get a single sample
    print("Fetching single sample...")
    sample = dataset[0]
    # Check sample keys
    print("Sample keys:", list(sample.keys()))
    print("Points shape:", sample['points'].data.shape)
    print("Ground truth boxes count:", len(sample['gt_bboxes_3d'].data))
    
    # 4. Collate the sample into a batch of size 1
    print("Collating sample into batch...")
    data_batch = collate([sample], samples_per_gpu=1)
    
    # Move batch data to GPU
    if torch.cuda.is_available():
        data_batch = scatter(data_batch, [0])[0]
        print("Moved data batch to GPU.")
    
    # 5. Build detector model
    print("Building detector...")
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    # Load pre-trained checkpoint
    checkpoint_path = 'projects/adv_aug/checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth'
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    model = model.to(device)
    model.train() # Must be in train mode to compute loss
    
    # Put BatchNorm layers in eval mode so they don't update statistics during attack
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            
    # 6. Gradient Verification: Manual Forward Pass to get Intermediate Voxels
    print("\n--- Running Manual Forward Pass (Original Data) ---")
    points_list = data_batch['points'] # list of Tensors
    
    # Run voxelize manually
    voxels, num_points, coors = model.voxelize(points_list)
    print(f"Voxel shape: {voxels.shape}")
    print(f"Coordinates shape: {coors.shape}")
    print(f"Number of points per voxel shape: {num_points.shape}")
    
    # Enable requires_grad on the voxels tensor!
    voxels.requires_grad = True
    
    # Extract features manually
    voxel_features = model.pts_voxel_encoder(voxels, num_points, coors)
    batch_size = coors[-1, 0] + 1
    x = model.pts_middle_encoder(voxel_features, coors, batch_size)
    x = model.pts_backbone(x)
    if model.with_pts_neck:
        x = model.pts_neck(x)
        
    pts_feats = x
    
    # Run the head training forward to get losses
    losses = model.forward_pts_train(
        pts_feats, 
        data_batch['gt_bboxes_3d'], 
        data_batch['gt_labels_3d'], 
        data_batch['img_metas']
    )
    
    # Sum all losses
    loss = sum_losses(losses)
    print(f"Original Loss: {loss.item():.4f}")
    print_losses(losses)
            
    # 7. Compute Gradients w.r.t input voxels
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    # Retrieve gradients
    grad = voxels.grad
    if grad is None:
        print("\n[ERROR] Gradient w.r.t voxels is None!")
        return
        
    print("\n--- Gradient Verification ---")
    print(f"Voxel gradient shape: {grad.shape}")
    
    # Compute gradient magnitude per voxel
    # Shape of grad: [num_voxels, max_num_points_per_voxel, num_features]
    # We sum the gradients over the point dimension and features dimension to get a per-voxel gradient norm
    grad_norm = torch.norm(grad, dim=(1, 2))
    print(f"Voxel gradient norm stats - Max: {grad_norm.max().item():.6f}, Mean: {grad_norm.mean().item():.6f}")
    
    non_zero_grads = (grad_norm > 0).sum().item()
    total_voxels = len(grad_norm)
    print(f"Non-zero gradients count: {non_zero_grads} / {total_voxels} ({non_zero_grads / total_voxels * 100:.2f}%)")
    
    # 8. Voxel-Level Point Detachment (Removal) Attack Logic
    print("\n--- Running Voxel-Level Point Detachment Attack ---")
    # We remove voxels with the LARGEST gradient magnitudes (highest impact on loss)
    # Zeroing out these voxels in the voxels tensor is equivalent to removing them.
    ratio_to_remove = 0.10 # Remove 10% of voxels
    num_remove = int(total_voxels * ratio_to_remove)
    
    # Get indices of voxels with largest gradients
    values, remove_indices = torch.topk(grad_norm, num_remove, largest=True)
    
    # Create perturbed voxels tensor
    perturbed_voxels = voxels.detach().clone()
    perturbed_voxels[remove_indices] = 0 # zero out all features of points in these voxels
    print(f"Zeroed out {num_remove} voxels (10% of original {total_voxels}).")
    
    # 9. Evaluate model performance on perturbed voxels
    print("\n--- Running Forward Pass (Perturbed Voxel Data) ---")
    with torch.no_grad():
        perturbed_voxel_features = model.pts_voxel_encoder(perturbed_voxels, num_points, coors)
        perturbed_x = model.pts_middle_encoder(perturbed_voxel_features, coors, batch_size)
        perturbed_x = model.pts_backbone(perturbed_x)
        if model.with_pts_neck:
            perturbed_x = model.pts_neck(perturbed_x)
        perturbed_pts_feats = perturbed_x
        
        perturbed_losses = model.forward_pts_train(
            perturbed_pts_feats, 
            data_batch['gt_bboxes_3d'], 
            data_batch['gt_labels_3d'], 
            data_batch['img_metas']
        )
    
    perturbed_loss = sum_losses(perturbed_losses)
    print(f"Perturbed Loss: {perturbed_loss.item():.4f}")
    print_losses(perturbed_losses)
            
    loss_change = perturbed_loss.item() - loss.item()
    print(f"Loss change: {loss_change:.4f} (expecting a change/drop since voxels were perturbed)")
    
    print("\n=== Prototyping Script Successfully Completed! ===")

if __name__ == '__main__':
    main()
