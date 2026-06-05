import pickle
from collections import defaultdict
import numpy as np
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.bbox import BaseInstance3DBoxes
import torch
import math
import os.path as osp
import pandas as pd
import argparse

# Own imports
from utils import center_distance, velocity_l2, yaw_diff, attr_acc, scale_iou, cummean
from metrics import AP_nus, TP_nus, mAP, mTP, NDS
from sample import Sample
from summary_table import SummaryTable

def prep_nuscenes(results, threshold=0.2, adversarial=False, verbose=True, dist_mode=True):
    """
    Prepares result data for evaluation.
    Heavily inspired by the code in the nuscenes devkit! (https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py)
    Args:
        results: list of results. Contains original results, (optional) adversarial results and ground truth per sample
        verbose: prints?
        dist_mode: NuScenes uses a distance based threshold instead of an IoU based one. Non-dist mode not implemented yet!!!
    Output:
        data that can be used to calculate the metrics
    """
    # Aggregating the predictions and ground truths, because NDS is dataset wide, not result specific
    gt_boxes = []
    gt_labels = []

    pred_boxes = []
    pred_labels = []
    scores = []
    
    for res in results:
        # Extract information from results
        gt_boxes.extend(res["gt_boxes"][0][0])
        gt_labels.extend(res["gt_labels"][0][0])
        if adversarial:
            pred_boxes.extend(res["adv_result"][0]["pts_bbox"]["boxes_3d"])
            pred_labels.extend(res["adv_result"][0]["pts_bbox"]["labels_3d"])
            scores.extend(res["adv_result"][0]["pts_bbox"]["scores_3d"])
        else:
            pred_boxes.extend(res["result"][0]["pts_bbox"]["boxes_3d"])
            pred_labels.extend(res["result"][0]["pts_bbox"]["labels_3d"])
            scores.extend(res["result"][0]["pts_bbox"]["scores_3d"])

    # Group prediction for labels
    grouped_preds = defaultdict(lambda: {'boxes': [], 'scores': []})
    for box, label, score in zip(pred_boxes, pred_labels, scores):
        key = int(label.item())
        grouped_preds[key]['boxes'].append(box)
        grouped_preds[key]['scores'].append(score)
            
    accumulated_data = defaultdict(lambda: {'recall': [], 'precision': [], 'confidence': [], 'trans_err': [], 'vel_err': [], 'scale_err':[], 'orient_err': [], 'attr_err': [] })
    # Iterate through every label/class and gather information
    for label, data in grouped_preds.items():
        if verbose:
            print("Found {} PRED of class {} out of {} total.".
                format(len(data['scores']), label, len(pred_boxes)))
        # Amount of positives in the gt
        npos = len([1 for gt_label in gt_labels if gt_label == label])
        # print("num pos: ", npos)
        # scores as float
        scores = [s.item() for s in data['scores']]
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Reorder boxes and scores according to sorted indices
        sorted_boxes = [data['boxes'][i] for i in sorted_indices]
        sorted_scores = [data['scores'][i] for i in sorted_indices]
        grouped_preds[label] = {'boxes': sorted_boxes, 'scores': sorted_scores}
        # Accumulators
        tp = []
        fp = []
        conf = []
        # match_data holds the extra metrics we calculate for each match.
        match_data = {'trans_err': [],
                    'vel_err': [],
                    'scale_err': [],
                    'orient_err': [],
                    'attr_err': [],
                    'conf': []}
        # if it does not exist in the gt
        if npos == 0:
            continue
        # Match and accumulate match data
        taken = set()
        for idx, box in enumerate(sorted_boxes):
            min_dist = np.inf
            match_gt_idx = None
            gt_box_match = None
            # Next steps: compare to gt_boxes using distance function or iou, find closest ones (take the highest confidence that is in threshold)
            # For now the NuScenes way: using distance, not IoU
            for gt_idx, gt_box in enumerate(gt_boxes):
                # Find closest match among ground truth boxes
                if gt_labels[gt_idx] == label and not gt_idx in taken:
                    this_distance = center_distance(gt_box, box)
                    # print("Distance: ", this_distance)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx
                        gt_box_match = gt_box
            # If the closest match is close enough according to threshold we have a match!
            is_match = min_dist < threshold
            if is_match:
                taken.add(match_gt_idx)
                # print("Found match for: ", match_gt_idx)
                #  Update tp, fp and confs.
                tp.append(1)
                fp.append(0)
                conf.append(sorted_scores[idx])

                match_data['trans_err'].append(center_distance(gt_box_match, box))
                match_data['vel_err'].append(velocity_l2(gt_box_match, box))
                match_data['scale_err'].append(1 - scale_iou(gt_box_match, box))

                # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                period = np.pi if label == 'barrier' else 2 * np.pi #TODO: Do I really keep this? the labels I currently use are numbers, not class names
                match_data['orient_err'].append(yaw_diff(gt_box_match, box, period=period))

                match_data['attr_err'].append(1 - attr_acc(gt_box_match, box))
                match_data['conf'].append(sorted_scores[idx])

            else:
                # No match. Mark this as a false positive.
                tp.append(0)
                fp.append(1)
                conf.append(sorted_scores[idx])

        # Check if we have any matches. If not, just return a "no predictions" array.
        if len(match_data['trans_err']) == 0:
            continue
        # Accumulate.
        tp = np.cumsum(tp).astype(float)
        fp = np.cumsum(fp).astype(float)
        conf = np.array(conf)

        # Calculate precision and recall.
        prec = tp / (fp + tp)
        rec = tp / float(npos)

        rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
        prec = np.interp(rec_interp, rec, prec, right=0)
        conf = np.interp(rec_interp, rec, conf, right=0)
        rec = rec_interp

        # resample
        for key in match_data.keys():
            if key == "conf":
                continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

            else:
                # For each match_data, we first calculate the accumulated mean.
                tmp = cummean(np.array(match_data[key]))

                # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
                match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]
        # Group data into dict with class label as key
        accumulated_data[int(label)]['recall'].extend(rec)
        accumulated_data[int(label)]['precision'].extend(conf)
        accumulated_data[int(label)]['confidence'].extend(conf)
        accumulated_data[int(label)]['trans_err'].extend(match_data['trans_err'])
        accumulated_data[int(label)]['vel_err'].extend(match_data['vel_err'])
        accumulated_data[int(label)]['scale_err'].extend(match_data['scale_err'])
        accumulated_data[int(label)]['orient_err'].extend(match_data['orient_err'])
        accumulated_data[int(label)]['attr_err'].extend(match_data['attr_err'])
    # Data is ready for NuScenes :D
    #print("Acc data: ", accumulated_data)
    return accumulated_data
        
def calc_NDS(results, threshold = 0.2, adversarial=False, verbose=True):
    if verbose:
        print('Accumulating metric data...', flush=True)
    data= prep_nuscenes(results, threshold=threshold, adversarial=adversarial, verbose = verbose)

    if verbose:
        print('Calculating metrics...', flush=True)
    meanAP = mAP(data)

    if verbose:
        print(f"mAP@{threshold}: {meanAP}", flush=True)

    meanTP = mTP(data)
    if verbose:
        print("mTP: ", meanTP, flush=True)

    nds = NDS(data)
    if verbose:
        print(f"NDS@{threshold}: ", nds, flush=True)
    return nds

if __name__ == "__main__":
    # Parser arguments
    attack_list = ["iou_detachment", "iou_attachment", "iou_perturbation", "fgsm", "pgd", "lidattack", "det", "att", "per", "lid"]
    model_list = ["centerpoint", "pillarnest", "pointpillars", "cp", "pn", "pp"]
    dataset_list = ["kitti", "nuscenes", "waymo", "kit", "nus", "way"]

    parser = argparse.ArgumentParser(description='Evaluation Pipeline for Adversarial attacks')
    parser.add_argument('--model', default="pp", help='Model Name', type=str.lower)
    parser.add_argument('--dataset', default="nus", help='Dataset Name', type=str.lower)
    parser.add_argument('--attack', default="att", help='Attack Name', type=str.lower)
    parser.add_argument('--mode', default="single", help="Are there multiple result files?", type=str.lower) # single, multi, auto
    parser.add_argument('--suffix', default="", help="Add suffix to file name?", type=str.lower)
    parser.add_argument('--reduced', action='store_true',help="Reduced Data?")
    parser.add_argument('--innout_thresh', default=0.8, help="Innout Threshold", type=float)
    parser.add_argument('--save', help='Save Path', type=str)
    parser.add_argument('--input_suffix', default="", help='if the input file has a suffix that deviates fron standart', type=str)
    parser.add_argument('--eps', action='store_true',help="Epsilon experiments?")
    parser.add_argument('--full', action='store_true',help="Also compute prediction box tables for mAP computation?")


    args = parser.parse_args()

    mode = args.mode
    # Process Parser args
    if args.attack == "iou_attachment" or args.attack == "att":
        attack = "att"
    elif args.attack == "iou_detachment" or args.attack == "det":
        attack = "det"
    elif args.attack == "iou_perturbation" or args.attack == "per":
        attack = "per"
    elif args.attack == "lidattack" or args.attack == "lid":
        attack = "lid"
    elif args.attack == "fgsm":
        attack = "fgsm"
    elif args.attack == "pgd":
        attack = "pgd"

    if args.dataset == "kitti" or args.dataset == "kit":
        dataset = "kit"
    elif args.dataset == "nuscenes" or args.dataset == "nus":
        dataset = "nus"
    elif args.dataset == "waymo" or args.dataset == "way":
        dataset = "way"

    if args.model == "centerpoint" or args.model == "cp":
        model = "cp"
    elif args.model == "pillarnest" or args.model == "pn":
        model = "pn"
    elif args.model == "pointpillars" or args.model == "pp":
        model = "pp"
    elif args.model == "focalformer3d" or args.model == "ff":
        model = "ff"

    verbose = True
    if args.reduced:
        base_path = f"/beegfs/krink/Projects/master-thesis/model_results/red/{dataset}/{model}/{attack}"
    else:
        #base_path = f"/beegfs/krink/Projects/master-thesis/model_results/{dataset}/{model}/{attack}"
        base_path = f"/beegfs/krink/Projects/adversarial-attacks/final_pickles/{dataset}/{model}/{attack}"
    if args.eps:
        base_path = f"/beegfs/krink/Projects/master-thesis/model_results/eps/{dataset}/{model}/{attack}"

    path = osp.join(base_path, f"sample_results{args.input_suffix}.pkl")
    # print("Computing NDS...")
    # nds = calc_NDS(results)
    # adv_nds = calc_NDS(results, adversarial=True)
    # print("NDS: ", nds, " , Adversarial NDS: ", adv_nds)
    print(f"Starting run for {dataset}/{model}/{attack}{args.input_suffix}!")
    print("Creating table based on results...", flush=True)
    if args.save:
        if args.full:
            save_path = osp.join(args.save, f"{dataset}_{model}_{attack}{args.suffix}_full.db")
        else:
            save_path = osp.join(args.save, f"{dataset}_{model}_{attack}{args.suffix}.db")
    else:
        if args.full:
            save_path = osp.join(base_path, f"{dataset}_{model}_{attack}{args.suffix}_full.db")
        else:
            save_path = osp.join(base_path, f"{dataset}_{model}_{attack}{args.suffix}.db")

    if args.full:
        full = True
    else:
        full = False
    table = SummaryTable(path, save_path=save_path, mode=mode, innout_thresh=args.innout_thresh, full=full)
    # table.save(base_path, table_name=f"{dataset}_{model}_{attack}.pkl", box_table_name=f"box_{dataset}_{model}_{attack}.pkl")
    print("Done! Saved table!", flush=True)

    