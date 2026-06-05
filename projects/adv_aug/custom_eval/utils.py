import numpy as np
import torch
import math
    
def max_recall_ind(confidence):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

def center_distance(gt_box, pred_box) -> float:
    """
    Based on the NuScenes code, adapted to fit the code!
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    gt_center = gt_box[:2]
    pred_center = pred_box[:2]
    return torch.norm(gt_center - pred_center)

def velocity_l2(gt_box, pred_box) -> float:
    """
    Based on the NuScenes code, adapted to fit the code!
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box[5:]) - np.array(gt_box[5:]))

def yaw_diff(gt_box, eval_box, period: float = 2*np.pi) -> float:
    """
    Based on the NuScenes code, adapted to fit the code!
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = gt_box[6]
    yaw_pred = eval_box[6]
    
    diff = (yaw_pred - yaw_gt + math.pi) % period - math.pi 
    return abs(diff)

def attr_acc(gt_box, pred_box) -> float:
    """
    TODO: Problem: I currently do not use any attributes and I don't know whether other datasets use them. I will skip this for now!
    Based on the NuScenes code, adapted to fit the code!
    Computes the classification accuracy for the attribute of this class (if any).
    If the GT class has no attributes or the annotation is missing attributes, we assign an accuracy of nan, which is
    ignored later on.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Attribute classification accuracy (0 or 1) or nan if GT annotation does not have any attributes.
    
    if gt_box.attribute_name == '':
        # If the class does not have attributes or this particular sample is missing attributes, return nan, which is
        # ignored later. Note that about 0.4% of the sample_annotations have no attributes, although they should.
        acc = np.nan
    else:
        # Check that label is correct.
        acc = float(gt_box.attribute_name == pred_box.attribute_name)
    return acc
    """
    return np.nan

def scale_iou(sample_annotation, sample_result) -> float:
    """
    Based on the NuScenes code!
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = sample_annotation[3:6]
    sr_size = sample_result[3:6]
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IoU
    min_wlh = torch.min(sa_size, sr_size)
    volume_annotation = torch.prod(sa_size)
    volume_result = torch.prod(sr_size)
    intersection = torch.prod(min_wlh)
    union = volume_annotation + volume_result - intersection
    iou = intersection / union

    return iou

def cummean(x: np.array) -> np.array:
    """
    Copied from NuScenes code!
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)