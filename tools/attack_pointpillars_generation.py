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


def direct_inference(model, points_tensor, img_metas):
    """Run inference directly through model internals.

    Bypasses forward_test/simple_test which expect test-mode data layout.
    Works with raw [N, 5] point tensors from train-pipeline data.
    """
    with torch.no_grad():
        voxels, num_points, coors = model.voxelize([points_tensor])
        voxel_features = model.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = model.pts_middle_encoder(voxel_features, coors, batch_size)
        x = model.pts_backbone(x)
        if model.with_pts_neck:
            x = model.pts_neck(x)
        outs = model.pts_bbox_head(x)
        bbox_list = model.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=True)
    return bbox_list

def get_predictions_confidence_drop(model, data_batch, points, original_mean_conf):
    """Run forward test to get bounding boxes and measure confidence drop."""
    img_metas = data_batch['img_metas']
    bbox_list = direct_inference(model, points, img_metas)
    scores_3d = bbox_list[0][1]
    mean_conf = scores_3d.mean().item() if len(scores_3d) > 0 else 0.0
    return mean_conf

def apply_attack(model, data_batch, max_iters=30, alpha=0.01):
    points_list = data_batch['points']
    if hasattr(points_list, 'data'):
        points_list = points_list.data
    points_tensor = points_list[0]
    if hasattr(points_tensor, 'data'):
        points_tensor = points_tensor.data
    if isinstance(points_tensor, list):
        points_tensor = points_tensor[0]
    points = points_tensor.clone() # [N, 5]
    device = points.device
    
    # Evaluate initial confidence via direct inference
    img_metas = data_batch['img_metas']
    bbox_list = direct_inference(model, points, img_metas)
    if len(bbox_list[0][1]) > 0:
        initial_conf = bbox_list[0][1].mean().item()
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
            
            # Match feature dimension of existing points (4 or 5)
            n_extra = points.shape[1] - 3
            extra_feats = torch.zeros(50, n_extra, device=device)
            extra_feats[:, 0] = 0.5  # mid intensity
            
            new_points = torch.cat([global_pts, extra_feats], dim=1)
            points = torch.cat([points, new_points], dim=0)
            num_points = points.shape[0]
            
        # Append indices (6th channel)
        indices = torch.arange(num_points, device=device, dtype=points.dtype).unsqueeze(1)
        points_with_idx = torch.cat([points, indices], dim=1)
        
        # Forward pass until voxelization
        voxels_tuple = model.voxelize([points_with_idx])
        res_voxels, res_num_points, res_coors = voxels_tuple
        
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
    # Override to train pipeline so GT boxes are available for attachment attack
    cfg.data.val.test_mode = False
    cfg.data.val.pipeline = cfg.data.train.pipeline
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
        else:
            for k, v in data_batch.items():
                if hasattr(v, 'data'):
                    val = v.data[0]
                    if isinstance(val, list) and len(val) > 0 and hasattr(val[0], 'data'):
                        val = [x.data for x in val]
                    elif hasattr(val, 'data'):
                        val = val.data
                    data_batch[k] = val
            if isinstance(data_batch['img_metas'], list) and not isinstance(data_batch['img_metas'][0], list):
                data_batch['img_metas'] = [data_batch['img_metas']]
            
        points_history = apply_attack(model, data_batch, max_iters=30, alpha=0.01)
        
        for iter_num, pts in points_history.items():
            out_file = os.path.join(out_dir, f'frame_{i}_iter_{iter_num}.bin')
            pts.astype(np.float32).tofile(out_file)

if __name__ == '__main__':
    main()
