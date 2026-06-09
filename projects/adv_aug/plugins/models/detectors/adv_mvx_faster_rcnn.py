import torch
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_faster_rcnn import MVXFasterRCNN


@DETECTORS.register_module()
class AdvMVXFasterRCNN(MVXFasterRCNN):
    """AdvMVXFasterRCNN with Feature-Space PGD Adversarial Augmentation."""

    def __init__(self, adv_eps=0.031, adv_alpha=0.00627, adv_steps=5, adv_lambda=1.0, **kwargs):
        super(AdvMVXFasterRCNN, self).__init__(**kwargs)
        self.adv_eps = adv_eps
        self.adv_alpha = adv_alpha
        self.adv_steps = adv_steps
        self.adv_lambda = adv_lambda

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function with adversarial augmentation."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        losses = dict()
        
        if pts_feats:
            # 2. Compute clean loss
            losses_pts_clean = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                      gt_labels_3d, img_metas,
                                                      gt_bboxes_ignore)
            
            # Rename keys to indicate clean loss
            for k, v in losses_pts_clean.items():
                if isinstance(v, list):
                    losses[f'{k}_clean'] = [v_i for v_i in v]
                else:
                    losses[f'{k}_clean'] = v
                    
            # 3. PGD Loop
            clean_feats_detached = [f.detach().clone() for f in pts_feats]
            adv_feats = [f.detach().clone() for f in pts_feats]
            
            # Save original training state and set to eval for PGD loop to avoid updating BatchNorm stats
            is_training = self.pts_bbox_head.training
            self.pts_bbox_head.eval()
            
            for step in range(self.adv_steps):
                for f in adv_feats:
                    f.requires_grad = True
                
                # Forward through head
                adv_losses = self.forward_pts_train(adv_feats, gt_bboxes_3d,
                                                    gt_labels_3d, img_metas,
                                                    gt_bboxes_ignore)
                
                # Compute total loss for backprop
                loss_adv_step = 0
                for k, v in adv_losses.items():
                    if 'loss' in k:
                        if isinstance(v, list):
                            loss_adv_step += sum(v)
                        else:
                            loss_adv_step += v
                            
                # Backprop to get gradient
                self.zero_grad()
                loss_adv_step.backward()
                
                # Update
                with torch.no_grad():
                    for i, f in enumerate(adv_feats):
                        grad = f.grad.sign()
                        f.data = f.data + self.adv_alpha * grad
                        # Clamp
                        f.data = torch.max(torch.min(f.data, clean_feats_detached[i] + self.adv_eps), clean_feats_detached[i] - self.adv_eps)
                        f.grad = None

            if is_training:
                self.pts_bbox_head.train()
                
            # Re-attach the adversarial features to the computational graph
            # This ensures gradients flow back to the backbone
            final_adv_feats = [pts_feats[i] + (adv_feats[i] - clean_feats_detached[i]).detach() for i in range(len(pts_feats))]
            
            # 4. Compute adversarial loss
            losses_pts_adv = self.forward_pts_train(final_adv_feats, gt_bboxes_3d,
                                                    gt_labels_3d, img_metas,
                                                    gt_bboxes_ignore)
            
            # 5. Return combined loss
            for k, v in losses_pts_adv.items():
                if isinstance(v, list):
                    losses[f'{k}_adv'] = [v_i * self.adv_lambda for v_i in v]
                else:
                    losses[f'{k}_adv'] = v * self.adv_lambda
                    
        # Assume img_feats is not used in this specific experiment based on context,
        # but keep it for completeness
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
            
        return losses
