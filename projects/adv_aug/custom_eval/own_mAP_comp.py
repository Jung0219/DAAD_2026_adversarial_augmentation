from __future__ import annotations
import sqlite3
import pickle
import pandas as pd
import numpy as np
import torch
import os
from typing import Dict, Iterable, Optional, Tuple, Sequence, Union, List
import time

def ap_from_pr(rec, prec, recall_samples=101):
    """
    Compute AP by interpolated precision sampled at fixed recall points.
    - recall_samples=101 -> 0..1 step 0.01 (nuScenes/Waymo-like)
    - recall_samples=40  -> KITTI R40
    """
    if len(rec) == 0:
        return 0.0

    # precision envelope (monotonic decreasing)
    mprec = np.maximum.accumulate(prec[::-1])[::-1]

    # sample at fixed recall points
    if isinstance(recall_samples, int):
        rs = np.linspace(0.0, 1.0, recall_samples)
    else:
        rs = np.asarray(recall_samples, dtype=float)

    # for each recall sample r, precision = max precision where rec >= r
    ap = 0.0
    for r in rs:
        idx = np.where(rec >= r)[0]
        ap += (mprec[idx[0]] if idx.size > 0 else 0.0)
    ap /= len(rs)
    return float(ap)


def greedy_match_one_pred(cand_df, metric, thr, used_gt, better):
    """
    cand_df: candidates for ONE prediction (already filtered to same class)
    metric: 'dist_xy' or 'iou_3d' (or 'iou_bev')
    thr: threshold (dist <= thr) or (iou >= thr)
    used_gt: set of (sample_id, gt_id) already matched
    better: 'min' for dist, 'max' for iou
    returns: matched (True/False), matched_gt_id or None, optional weight (for APH)
    """
    if cand_df is None or len(cand_df) == 0:
        return False, None, None

    # Filter by threshold and unused GTs
    if better == "min":
        ok = cand_df[metric].notna() & (cand_df[metric] <= thr)
        c = cand_df[ok].copy()
        if len(c) == 0:
            return False, None, None
        c["is_used"] = list(zip(c["sample_id"], c["gt_id"]))
        c = c[~c["is_used"].isin(used_gt)]
        if len(c) == 0:
            return False, None, None
        # choose smallest distance
        row = c.loc[c[metric].idxmin()]
    else:
        ok = cand_df[metric].notna() & (cand_df[metric] >= thr)
        c = cand_df[ok].copy()
        if len(c) == 0:
            return False, None, None
        c["is_used"] = list(zip(c["sample_id"], c["gt_id"]))
        c = c[~c["is_used"].isin(used_gt)]
        if len(c) == 0:
            return False, None, None
        # choose largest IoU
        row = c.loc[c[metric].idxmax()]

    matched_key = (row["sample_id"], int(row["gt_id"]))
    return True, int(row["gt_id"]), row  # return full row to compute weight if needed


def eval_class_ap(pred_df, cand_df, gt_df, class_id, is_adv,
                  metric, thr, recall_samples=101,
                  heading_weight_fn=None):
    """
    pred_df columns: sample_id, is_adv, pred_id, class_id, score
    cand_df columns: sample_id, is_adv, pred_id, gt_id, dist_xy, iou_3d, (iou_bev), yaw_diff
    gt_df columns: sample_id, gt_id, class_id, ...
    heading_weight_fn: function(row)->weight in [0,1] for APH; if None compute normal AP.
    """
    # filter preds/gt by class + adv flag
    preds = pred_df[(pred_df["class_id"] == class_id) & (pred_df["is_adv"] == is_adv)].copy()
    gts   = gt_df[gt_df["class_id"] == class_id].copy()

    num_gt = len(gts)
    if num_gt == 0:
        return 0.0, {"num_gt": 0, "num_pred": len(preds)}

    # sort predictions globally by score desc
    preds.sort_values("score", ascending=False, inplace=True)

    # used GTs are per-sample (and per class implicitly)
    used_gt = set()

    tps = []
    fps = []
    tpw = []  # weighted TP for APH (optional)
    scores = []

    # Pre-index candidates by (sample_id,is_adv,pred_id) for speed
    # Ensure cand_df has only relevant is_adv + class candidates if you want, but not required.
    cand_keyed = cand_df[cand_df["is_adv"] == is_adv].set_index(["sample_id", "pred_id"])
    cand_keyed = (
        cand_df[cand_df["is_adv"] == is_adv]
        .set_index(["sample_id", "pred_id"])
        .sort_index()
    )

    better = "min" if metric == "dist_xy" else "max"

    for _, p in preds.iterrows():
        sample_id = p["sample_id"]
        pred_id   = int(p["pred_id"])
        scores.append(float(p["score"]))

        # candidates for this pred
        try:
            c = cand_keyed.loc[(sample_id, pred_id)].reset_index()
        except KeyError:
            c = None

        matched, gt_id, row = greedy_match_one_pred(c, metric, thr, used_gt, better)

        if matched:
            used_gt.add((sample_id, gt_id))
            tps.append(1)
            fps.append(0)
            if heading_weight_fn is not None:
                w = float(heading_weight_fn(row))
                tpw.append(w)
        else:
            tps.append(0)
            fps.append(1)
            if heading_weight_fn is not None:
                tpw.append(0.0)

    tps = np.asarray(tps, dtype=float)
    fps = np.asarray(fps, dtype=float)

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)

    rec = cum_tp / float(num_gt)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    ap = ap_from_pr(rec, prec, recall_samples=recall_samples)

    # If doing APH: replace precision with heading-weighted precision and compute APH similarly.
    # NOTE: Different implementations define APH slightly differently (recall vs weighted recall).
    # This version keeps recall based on true TP count, and weights precision by heading.
    if heading_weight_fn is not None:
        cum_tpw = np.cumsum(np.asarray(tpw, dtype=float))
        prec_h = cum_tpw / np.maximum(cum_tp + cum_fp, 1e-12)
        aph = ap_from_pr(rec, prec_h, recall_samples=recall_samples)
        return aph, {"num_gt": num_gt, "num_pred": len(preds)}

    return ap, {"num_gt": num_gt, "num_pred": len(preds)}


def nuscenes_map(pred_df, cand_df, gt_df, class_ids, is_adv,
                 dist_thresholds=(0.5, 1.0, 2.0, 4.0),
                 recall_samples=101):
    per_class = {}
    for cid in class_ids:
        if cid == -1:
            continue
        aps = []
        for thr in dist_thresholds:
            ap, _ = eval_class_ap(pred_df, cand_df, gt_df, cid, is_adv,
                                  metric="dist_xy", thr=thr,
                                  recall_samples=recall_samples)
            aps.append(ap)
        per_class[cid] = float(np.mean(aps)) if aps else 0.0
    mAP = float(np.mean(list(per_class.values()))) if per_class else 0.0
    # print_res("NuScenes", mAP, per_class)
    return mAP, per_class

def kitti_map(pred_df, cand_df, gt_df, class_ids, is_adv,
              iou_thresholds_by_class,  # dict {class_id: thr}
              metric="iou_3d",          # or "iou_bev"
              recall_samples=40):
    per_class = {}
    for cid in class_ids:
        if cid == -1:
            continue
        thr = float(iou_thresholds_by_class[cid])
        ap, _ = eval_class_ap(pred_df, cand_df, gt_df, cid, is_adv,
                              metric=metric, thr=thr,
                              recall_samples=recall_samples)
        per_class[cid] = ap
    mAP = float(np.mean(list(per_class.values()))) if per_class else 0.0
    # print_res("Kitti", mAP, per_class)
    return mAP, per_class

import math

def simple_heading_weight(row):
    # Replace with the exact Waymo definition you want.
    # This is a common smooth weight: 1 at 0 error, 0 at pi.
    yaw = row.get("orient_err")
    if yaw is None or (isinstance(yaw, float) and np.isnan(yaw)):
        return 0.0
    yaw = abs(float(yaw))
    yaw = min(yaw, math.pi)
    return max(0.0, 1.0 - yaw / math.pi)

def waymo_map_and_aph(pred_df, cand_df, gt_df, class_ids, is_adv,
                      iou_thresholds_by_class,
                      recall_samples=101):
    per_class_ap = {}
    per_class_aph = {}
    for cid in class_ids:
        if cid == -1:
            continue
        thr = float(iou_thresholds_by_class[cid])

        ap, _ = eval_class_ap(pred_df, cand_df, gt_df, cid, is_adv,
                              metric="iou_3d", thr=thr,
                              recall_samples=recall_samples,
                              heading_weight_fn=None)
        aph, _ = eval_class_ap(pred_df, cand_df, gt_df, cid, is_adv,
                               metric="iou_3d", thr=thr,
                               recall_samples=recall_samples,
                               heading_weight_fn=simple_heading_weight)

        per_class_ap[cid] = ap
        per_class_aph[cid] = aph

    mAP = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    mAPH = float(np.mean(list(per_class_aph.values()))) if per_class_aph else 0.0
    # print_res("Waymo", mAP, per_class_ap)
    # print_res("Waymo (heading)", mAPH, per_class_aph)
    return (mAP, per_class_ap), (mAPH, per_class_aph)

def print_res_all(dataset, mAP_clean, per_class_clean, mAP_adv, per_class_adv):
    print(f"=" * 60)
    print(f"{dataset} Style mAP/AP")
    print(f"-" * 60)
    print(f"mAP clean/adv: {mAP_clean:.3f}/{mAP_adv:.3f}")
    for ((class_clean, pcc), (class_adv, pca)) in zip(per_class_clean.items(), per_class_adv.items()):
        if class_clean != class_adv:
            print("classes are not aligned!")
            break
        print(f"    Class {class_clean} clean/adv: {pcc:.3}/{pca:.3}")
    print(f"=" * 60)

def print_res(dataset, mAP, per_class):
    print(f"=" * 60)
    print(f"{dataset} Style mAP/AP")
    print(f"-" * 60)
    print(f"mAP: {mAP:.3f}")
    for (class_clean, pcc) in per_class.items():
        print(f"    Class {class_clean}: {pcc:.3}")
    print(f"=" * 60)

def load_table(db_path, table_name):
    if not os.path.exists(db_path):
        print(db_path, table_name, " does not exist!!!")
        return None

    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception as e:
        print(f"Skipping {db_path} ({table_name}): {e}")
        return None
    

def normalize_pred_cand(pred_df, cand_df):
    # preds
    pred_df = pred_df.copy()
    pred_df["is_adv"] = pred_df["is_adv"].astype(int)
    pred_df["pred_id"] = pred_df["pred_id"].astype(int)
    pred_df["class_id"] = pred_df["class_id"].astype(int)
    pred_df["score"] = pred_df["score"].astype(float)

    # cands
    cand_df = cand_df.copy()
    cand_df["is_adv"] = cand_df["is_adv"].astype(int)
    cand_df["pred_id"] = cand_df["pred_id"].astype(int)
    cand_df["gt_id"] = cand_df["gt_id"].astype(int)
    if "dist_xy" in cand_df.columns:
        cand_df["dist_xy"] = pd.to_numeric(cand_df["dist_xy"], errors="coerce")
    if "iou_3d" in cand_df.columns:
        cand_df["iou_3d"] = pd.to_numeric(cand_df["iou_3d"], errors="coerce")
    if "yaw_diff" in cand_df.columns:
        cand_df["yaw_diff"] = pd.to_numeric(cand_df["yaw_diff"], errors="coerce")
    return pred_df, cand_df



if __name__ == "__main__":
    DATASETS = {
        "kit": "kit",
        "nus": "nus",
        "way": "way"
    }

    MODELS = {
        "pp": "pp",
        "cp": "cp",
        "pn": "pn",
        "ff": "ff"
    }

    ATTACKS = {
        # "att": "att",
        # "det": "det",
        # "per": "per",
        # "lid": "lid",
        "fgsm": "fgsm",
        "pgd": "pgd"
        # "box": "box" # Commented because no data yet!
    }

    results = []
    results_per_class = []

    for d_abbr, d_name in DATASETS.items():
        for m_abbr, m_name in MODELS.items():
            if d_abbr == "kit" and m_abbr == "ff":
                continue
            for a_abbr, a_name in ATTACKS.items():
                start = time.time()
                db_file = f"/beegfs/krink/Projects/master-thesis/model_results/red/tables/{d_abbr}_{m_abbr}_{a_abbr}_full.db"
                # ---- boxes table ----
                df_boxes = load_table(db_file, "boxes")
                if df_boxes is not None:
                    df_boxes["dataset"] = d_name
                    df_boxes["model"] = m_name
                    df_boxes["attack"] = a_name

                # ---- compute mAP/AP----
                pred_df = load_table(db_file, "pred_boxes")  # pred_boxes table
                cand_df = load_table(db_file, "pred_gt_candidates")  # pred_gt_candidates table
                # Check that tables are not empty
                if df_boxes is None or pred_df is None or cand_df is None:
                    print("One of the boxes is None! Skipping: ", db_file)
                    continue
                if len(df_boxes) == 0 or len(pred_df) == 0 or len(cand_df) == 0:
                    print("One of the boxes has no entries! Skipping: ", db_file)
                    continue

                gt_df = (
                    df_boxes[["sample_id", "gt_box_id", "class"]]
                    .drop_duplicates()
                    .rename(columns={"gt_box_id": "gt_id", "class": "class_id"})
                )
                gt_df["class_id"] = gt_df["class_id"].astype(int)
                gt_df["gt_id"] = gt_df["gt_id"].astype(int)

                # normalize preds/cands
                pred_df, cand_df = normalize_pred_cand(pred_df, cand_df)

                # join GT class onto candidates (robust)
                cand_df2 = cand_df.merge(
                    gt_df[["sample_id", "gt_id", "class_id"]],
                    on=["sample_id", "gt_id"],
                    how="inner",
                )

                class_ids = sorted(gt_df["class_id"].unique())
                # define thresholds based on datasets
                kitti_thr = {cid: 0.5 for cid in class_ids}  # replace per class
                waymo_thr = {cid: 0.5 for cid in class_ids}
                if d_abbr == "kit":
                    # Car has a iou threshold of 0.7
                    kitti_thr[2] = 0.7
                    waymo_thr[2] = 0.7
                else:
                    # Car has a iou threshold of 0.7
                    kitti_thr[0] = 0.7
                    waymo_thr[0] = 0.7

                #---- compute metrics ----
                # nuScenes (distance)
                n_mAP_clean, n_per_class_clean = nuscenes_map(pred_df, cand_df2, gt_df, class_ids, is_adv=0)
                n_mAP_adv,   n_per_class_adv   = nuscenes_map(pred_df, cand_df2, gt_df, class_ids, is_adv=1)

                # KITTI (IoU, R40)
                k_mAP_clean, k_per_class_clean = kitti_map(pred_df, cand_df2, gt_df, class_ids, 0, kitti_thr, metric="iou_3d")
                k_mAP_adv,   k_per_class_adv   = kitti_map(pred_df, cand_df2, gt_df, class_ids, 1, kitti_thr, metric="iou_3d")

                # Waymo (IoU AP + APH)
                (w_mAP_clean, w_per_class_ap_clean), (w_mAPH_clean, w_per_class_aph_clean) = \
                    waymo_map_and_aph(pred_df, cand_df2, gt_df, class_ids, 0, waymo_thr)
                (w_mAP_adv,   w_per_class_ap_adv),   (w_mAPH_adv,   w_per_class_aph_adv) = \
                    waymo_map_and_aph(pred_df, cand_df2, gt_df, class_ids, 1, waymo_thr)

                # ---- summary row ----
                results.append({
                    "dataset": d_name,
                    "dataset_abbr": d_abbr,
                    "model": m_name,
                    "model_abbr": m_abbr,
                    "attack": a_name,
                    "attack_abbr": a_abbr,

                    "nus_mAP_clean": n_mAP_clean,
                    "nus_mAP_adv": n_mAP_adv,

                    "kitti_mAP_clean": k_mAP_clean,
                    "kitti_mAP_adv": k_mAP_adv,

                    "waymo_mAP_clean": w_mAP_clean,
                    "waymo_mAP_adv": w_mAP_adv,
                    "waymo_mAPH_clean": w_mAPH_clean,
                    "waymo_mAPH_adv": w_mAPH_adv,

                    "num_gt": len(gt_df),
                    "num_pred_clean": int((pred_df["is_adv"] == 0).sum()),
                    "num_pred_adv": int((pred_df["is_adv"] == 1).sum()),
                })

                # ---- per-class rows ----
                for cid in class_ids:

                    results_per_class.append({
                        "dataset": d_name, "dataset_abbr": d_abbr,
                        "model": m_name, "model_abbr": m_abbr,
                        "attack": a_name, "attack_abbr": a_abbr,
                        "class_id": cid,

                        # nuScenes distance AP averaged over distance thresholds (per-class)
                        "nus_AP_clean": float(n_per_class_clean.get(cid, 0.0)),
                        "nus_AP_adv":   float(n_per_class_adv.get(cid, 0.0)),

                        # KITTI IoU AP (per-class)
                        "kitti_AP_clean": float(k_per_class_clean.get(cid, 0.0)),
                        "kitti_AP_adv":   float(k_per_class_adv.get(cid, 0.0)),

                        # Waymo IoU AP + APH (per-class)
                        "waymo_AP_clean":  float(w_per_class_ap_clean.get(cid, 0.0)),
                        "waymo_AP_adv":    float(w_per_class_ap_adv.get(cid, 0.0)),
                        "waymo_APH_clean": float(w_per_class_aph_clean.get(cid, 0.0)),
                        "waymo_APH_adv":   float(w_per_class_aph_adv.get(cid, 0.0)),
                    })

                print(f"--- {d_abbr}/{m_abbr}/{a_abbr} mAP computation finished after: {(start-time.time())/60}min ---")
                print(" Kitti mAP clean: ", k_mAP_clean, " Kitti mAP adv: ", k_mAP_adv)
                print(" nuScenes mAP clean: ", n_mAP_clean, " nuScenes mAP adv: ", n_mAP_adv)
                print(" Waymo mAP clean: ", w_mAP_clean, " Waymo mAP adv: ", w_mAP_adv)
                print(" Waymo mAPH clean: ", w_mAPH_clean, " Waymo mAPH adv: ", w_mAPH_adv)

    # DataFrames
    results_df = pd.DataFrame(results)
    per_class_df = pd.DataFrame(results_per_class)

    mAP_data = {
        "results_df": results_df,
        "per_class_df": per_class_df,
    }
    with open('/beegfs/krink/Projects/master-thesis/own_mAP_data.pickle', 'wb') as f:
        pickle.dump(mAP_data, f, protocol=pickle.HIGHEST_PROTOCOL)
