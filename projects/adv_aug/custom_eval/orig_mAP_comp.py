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
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
import time
import os, glob, re
from pathlib import Path
from typing import Dict, Tuple

PATH_PREFIX = Path("/beegfs/krink/Projects") 

# Key = (model, dataset)
CONFIGS: Dict[Tuple[str, str], Path] = {
    ("cp", "nus"):
        Path("mmdetection3d/configs/centerpoint_attacks/centerpoint_nus_adv_red.py"),

    ("pn", "nus"):
        Path("mmdetection3d/configs/pillarnest/pillarnest_nus_adv_red.py"),

    ("pp", "nus"):
        Path("mmdetection3d/configs/pointpillars/pointpillars_nus_adv_red.py"),

    ("ff", "nus"):
        Path("mmdetection3d/configs/focalformer3d/FocalFormer3D_L_adv_red.py"),

    ("pp", "kit"):
        Path("mmdetection3d/configs/pointpillars/pointpillars_kitti_adv.py"),

    ("cp", "kit"):
        Path("mmdetection3d/configs/centerpoint_attacks/centerpoint_kitti_adv.py"),

    ("pn", "kit"):
        Path("mmdetection3d/configs/pillarnest/pillarnest_kitti_adv.py"),

    ("pp", "way"):
        Path("mmdetection3d/configs/pointpillars/pointpillars_waymo_adv_red.py"),

    ("cp", "way"):
        Path("mmdetection3d/configs/centerpoint_attacks/centerpoint_waymo_adv_red.py"),

    ("pn", "way"):
        Path("mmdetection3d/configs/pillarnest/pillarnest_waymo_adv_red.py"),

    ("ff", "way"):
        Path("mmdetection3d/configs/focalformer3d/FocalFormer3D_Waymo_L_adv_red.py"),
}


def resolve_config(model: str, dataset: str) -> str:
    key = (model.lower(), dataset.lower())
    try:
        ret = str(PATH_PREFIX / CONFIGS[key])
        print("Config path: ", ret)
        return ret
    except KeyError as e:
        valid = ", ".join(f"{m}/{d}" for (m, d) in sorted(CONFIGS))
        raise ValueError(f"Unknown combination {key}. Valid: {valid}") from e


def iter_results_multi(base_path, mode="auto"):
    root, ext = os.path.splitext(base_path)

    if mode == "single":
        yield from iter_results(base_path)
        return

    if mode == "multi":
        i = 0
        while True:
            path = f"{root}_{i}{ext}"
            if not os.path.exists(path):
                break
            yield from iter_results(path)
            i += 1
        return

    # auto (safe + efficient): glob once
    shards = sorted(
        glob.glob(f"{root}_[0-9]*{ext}"),
        key=lambda p: int(re.search(r"_(\d+)\.pkl$", p).group(1))
    )
    if shards:
        for p in shards:
            yield from iter_results(p)
    else:
        yield from iter_results(base_path)

def iter_results(file_path):
    """
    Generator that yields one sample at a time from mixed pickle formats:
    - Old format: a single list dumped once
    - New format: many single objects appended
    """
    # print(f"[iter_results] opening {file_path}", flush=True)
    with open(file_path, "rb") as f:
        try:
            t0 = time.time()
            first = pickle.load(f)
            # print(f"[iter_results] first object loaded in {time.time()-t0:.1f}s, type={type(first)}", flush = True)

            # Case 1: old format → list of samples
            if isinstance(first, list):
                for item in first:
                    yield item
            else:
                # Case 2: new / mixed format → first object is one sample
                yield first

            # Case 3: appended samples
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

        except EOFError:
            return

def to_cpu_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.ascontiguousarray(x)


def reorder_by_index(results, tokens, dataset):
    assert len(results) == len(tokens)
    assert len(results) == len(dataset)

    # Build mapping from token → prediction
    pred_by_token = {tok: pred for tok, pred in zip(tokens, results)}

    # Optional safety check
    assert len(pred_by_token) == len(tokens), "Duplicate tokens in predictions!"

    # Rebuild results in dataset order
    ordered_results = []
    missing = []

    for info in dataset.data_infos:
        token = info["token"]
        if token not in pred_by_token:
            missing.append(token)
            # create empty prediction (nuScenes requires a prediction entry)
            ordered_results.append({
                "pts_bbox": {
                    "boxes_3d": dataset.box_type_3d([]),
                    "scores_3d": torch.empty(0),
                    "labels_3d": torch.empty(0, dtype=torch.int64),
                }
            })
        else:
            ordered_results.append(pred_by_token[token])
    print("Missing tokens:", len(missing))
    return ordered_results

if __name__ == "__main__":
    # Parser arguments
    attack_list = ["iou_detachment", "iou_attachment", "iou_perturbation", "fgsm", "pgd", "lidattack", "det", "att", "per", "lid"]
    model_list = ["centerpoint", "pillarnest", "pointpillars", "cp", "pn", "pp"]
    dataset_list = ["kitti", "nuscenes", "waymo", "kit", "nus", "way"]

    parser = argparse.ArgumentParser(description='Evaluation Pipeline for Adversarial attacks')
    parser.add_argument('--model', default="pp", help='Model Name', type=str.lower)
    parser.add_argument('--dataset', default="nus", help='Dataset Name', type=str.lower)
    parser.add_argument('--attack', default="att", help='Attack Name', type=str.lower)
    parser.add_argument('--mode', default="auto", help="Are there multiple result files?", type=str.lower) # single, multi, auto
    parser.add_argument('--adv', action='store_true',help="Compute for Adversarial?")

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
    base_path = f"/beegfs/krink/Projects/master-thesis/model_results/reduced/{dataset}/{model}/{attack}"
    path = osp.join(base_path, "sample_results.pkl")
    results = []
    tokens = []
    print("Starting to read results...", flush=True)
    # aggregate results into correct format
    for res in iter_results_multi(path, mode):
        tokens.append(res["name"])
        if args.adv:
            results.append(res["adv_result"][0])
        else:
            results.append(res["result"][0])

    fixed = []
    for r in results:
        pb = r["pts_bbox"]
        boxes = pb["boxes_3d"]
        # Ensure boxes object is CPU
        if hasattr(boxes, "to"):
            boxes = boxes.to("cpu")
        scores = pb["scores_3d"].detach().cpu()
        labels = pb["labels_3d"].detach().cpu().to(torch.int64) 
        fixed.append({
            "pts_bbox": {
                "boxes_3d": boxes,
                "scores_3d": scores,
                "labels_3d": labels,
            }
        })

    results = fixed

    print("Finished reading results!", flush=True)

    # 1) Load the same config you normally use for KITTI val/test
    config = resolve_config(model, dataset)
    cfg = Config.fromfile(config)

    # 2) Build the VAL dataset (must match the split/order your predictions were produced on)
    ds = build_dataset(cfg.data.test)
    print("Starting to evaluate results...", flush=True)

    # 2) ensure no nesting + correct length
    print("Results length:", len(results))

    # Order by tokens
    if dataset == "nus":
        results = reorder_by_index(results, tokens, ds)

    # ---- NuScenes Patch ----
    if dataset == "nus" or dataset == "nuscenes":
        from nuscenes.eval.common import loaders

        subset_tokens = {info["token"] for info in ds.data_infos}

        _orig_load_gt = loaders.load_gt

        def load_gt_subset(nusc, eval_set, box_cls, verbose=False):
            gt = _orig_load_gt(nusc, eval_set, box_cls, verbose)
            # gt.boxes is a dict: sample_token -> List[DetectionBox]
            gt.boxes = {t: gt.boxes[t] for t in list(gt.boxes.keys()) if t in subset_tokens}
            return gt
        loaders.load_gt = load_gt_subset

    # 3) Run Evaluation (mAP tables)
    if dataset == "nus" or dataset == "kit":
        eval_results = ds.evaluate(results, metric='mAP')
    else:
        eval_results = ds.evaluate(results, metric='waymo')

    print("Finished to evaluate results:", eval_results, flush=True)
    if args.adv:
        save_path = f"/beegfs/krink/Projects/master-thesis/model_results/reduced/eval/{dataset}_{model}_{attack}.pkl"
    else:
        save_path = f"/beegfs/krink/Projects/master-thesis/model_results/reduced/eval/{dataset}_{model}_clean.pkl"

    os.makedirs("/beegfs/krink/Projects/master-thesis/model_results/reduced/eval", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(eval_results, f)
    print("Saved as pickle: ", save_path)


    