import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add codebase root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from mmdet3d.models.voxel_encoders.utils import get_paddings_indicator

def sum_losses(losses):
    total_loss = 0
    for k, v in losses.items():
        if 'loss' in k:
            if isinstance(v, list):
                total_loss += sum(x.mean() for x in v)
            else:
                total_loss += v.mean()
    return total_loss

def get_predictions_confidence_drop(model, data_batch, points, original_mean_conf):
    """Run forward test to get bounding boxes and measure confidence drop."""
    # Temporarily replace points
    old_points = data_batch['points']
    data_batch['points'] = [points]
    
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data_batch)
    
    data_batch['points'] = old_points
    
    # results is a list of dicts. We have batch_size=1
    boxes_3d = results[0]['pts_bbox']['boxes_3d']
    scores_3d = results[0]['pts_bbox']['scores_3d']
    
    mean_conf = scores_3d.mean().item() if len(scores_3d) > 0 else 0.0
    return mean_conf

def apply_attack(model, data_batch, max_iters=30, alpha=0.01):
    points = data_batch['points'][0].clone() # [N, 5]
    device = points.device
    
    # Evaluate initial confidence
    with torch.no_grad():
        initial_results = model(return_loss=False, rescale=True, **data_batch)
    if len(initial_results[0]['pts_bbox']['scores_3d']) > 0:
        initial_conf = initial_results[0]['pts_bbox']['scores_3d'].mean().item()
    else:
        initial_conf = 0.0
        
    print(f"Iter 0: Initial Conf={initial_conf:.4f}")
    
    all_points_history = {0: points.clone().cpu().numpy()}
    
    for iteration in range(1, max_iters + 1):
        num_points = points.shape[0]
        
        # 1. Attachment: Add 50 new noise points inside GT boxes
        gt_boxes = data_batch['gt_bboxes_3d'][0]
        if len(gt_boxes) > 0:
            num_gt = len(gt_boxes.tensor)
            # Sample random GT boxes
            box_indices = torch.randint(0, num_gt, (50,))
            sampled_boxes = gt_boxes.tensor[box_indices].to(device)
            # Sample random positions within these boxes [-0.5, 0.5] * wlh
            noise = (torch.rand(50, 3, device=device) - 0.5)
            wlh = sampled_boxes[:, 3:6]
            local_pts = noise * wlh
            
            # Simple rotation and translation
            yaws = sampled_boxes[:, 6]
            cos_y = torch.cos(yaws)
            sin_y = torch.sin(yaws)
            x = local_pts[:, 0] * cos_y - local_pts[:, 1] * sin_y
            y = local_pts[:, 0] * sin_y + local_pts[:, 1] * cos_y
            z = local_pts[:, 2]
            
            global_pts = torch.stack([x, y, z], dim=1) + sampled_boxes[:, :3]
            
            # Additional features: intensity, ring index
            extra_feats = torch.zeros(50, 2, device=device)
            extra_feats[:, 0] = 0.5 # mid intensity
            extra_feats[:, 1] = 0.0 # ring 0
            
            new_points = torch.cat([global_pts, extra_feats], dim=1)
            points = torch.cat([points, new_points], dim=0)
            num_points = points.shape[0]
            
        # Append indices (6th channel)
        indices = torch.arange(num_points, device=device, dtype=points.dtype).unsqueeze(1)
        points_with_idx = torch.cat([points, indices], dim=1)
        
        # Forward pass until voxelization
        voxels, coors, voxel_num_points = [], [], []
        res_voxels, res_coors, res_num_points = model.pts_voxel_layer(points_with_idx)
        
        voxel_point_indices = res_voxels[..., 5].long()
        voxels_5d = res_voxels[..., :5].detach().clone()
        voxels_5d.requires_grad = True
        
        # Encoder forward
        voxel_features = model.pts_voxel_encoder(voxels_5d, res_num_points, res_coors)
        batch_size = res_coors[-1, 0] + 1
        x = model.pts_middle_encoder(voxel_features, res_coors, batch_size)
        x = model.pts_backbone(x)
        if model.with_pts_neck:
            x = model.pts_neck(x)
        
        losses = model.forward_pts_train(
            x, 
            data_batch['gt_bboxes_3d'], 
            data_batch['gt_labels_3d'], 
            data_batch['img_metas']
        )
        
        loss = sum_losses(losses)
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        grad = voxels_5d.grad # [M, max_points, 5]
        
        # Accumulate gradients to points
        point_grads = torch.zeros((num_points, 5), device=device)
        mask = get_paddings_indicator(res_num_points, res_voxels.shape[1], axis=0)
        valid_grads = grad[mask]
        valid_indices = voxel_point_indices[mask]
        point_grads.scatter_add_(0, valid_indices.unsqueeze(1).expand(-1, 5), valid_grads)
        
        # 2. Perturbation: PGD
        # Perturb x,y,z only
        sign_data = point_grads.sign()
        points[:, :3] = points[:, :3] + alpha * sign_data[:, :3]
        
        # 3. Detachment: remove top 50 points based on gradient magnitude
        grad_norms = torch.norm(point_grads[:, :3], dim=1)
        _, top_indices = torch.topk(grad_norms, k=min(50, num_points))
        keep_mask = torch.ones(num_points, dtype=torch.bool, device=device)
        keep_mask[top_indices] = False
        points = points[keep_mask]
        
        if iteration % 5 == 0:
            conf = get_predictions_confidence_drop(model, data_batch, points, initial_conf)
            print(f"Iter {iteration}: Loss={loss.item():.4f}, Mean Conf={conf:.4f}, Points={len(points)}")
            all_points_history[iteration] = points.clone().cpu().numpy()
            
    return all_points_history

def main():
    config_path = 'projects/adv_aug/configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_baseline_mini.py'
    cfg = Config.fromfile(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cfg.data.val.data_root = 'data/nuscenes/'
    dataset = build_dataset(cfg.data.val)
    
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    checkpoint_path = 'projects/adv_aug/checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth'
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    model = model.to(device)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            
    out_dir = 'results/EXP-002'
    os.makedirs(out_dir, exist_ok=True)
    
    num_frames = min(10, len(dataset))
    for i in range(num_frames):
        print(f"\\n--- Processing Frame {i} ---")
        sample = dataset[i]
        data_batch = collate([sample], samples_per_gpu=1)
        if torch.cuda.is_available():
            data_batch = scatter(data_batch, [0])[0]
            
        points_history = apply_attack(model, data_batch, max_iters=30, alpha=0.01)
        
        for iter_num, pts in points_history.items():
            out_file = os.path.join(out_dir, f'frame_{i}_iter_{iter_num}.bin')
            pts.astype(np.float32).tofile(out_file)

if __name__ == '__main__':
    main()
