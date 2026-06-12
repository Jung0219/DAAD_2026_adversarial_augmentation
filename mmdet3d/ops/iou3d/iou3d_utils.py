import torch

try:
    from . import iou3d_cuda
except ImportError:
    iou3d_cuda = None


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    from mmcv.ops import boxes_iou_bev as mmcv_boxes_iou_bev
    return mmcv_boxes_iou_bev(boxes_a, boxes_b)


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    from mmcv.ops import nms_bev
    return nms_bev(boxes, scores, thresh, pre_maxsize, post_max_size)


def nms_normal_gpu(boxes, scores, thresh):
    from mmcv.ops import nms_normal_bev
    return nms_normal_bev(boxes, scores, thresh)
