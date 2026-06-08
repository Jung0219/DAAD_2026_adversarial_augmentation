# nuScenes Validation Evaluation Report
**Project: Adversarial Augmentation (LiDAR-only Baselines)**
**Date: June 8, 2026**

This report summarizes the performance of the four baseline 3D object detection models on the standard nuScenes validation dataset (6,019 samples) using LiDAR-only modality.

---

## 1. Overall Performance Comparison

| Model | Modality | mAP ↑ | NDS ↑ | mATE (m) ↓ | mASE ↓ | mAOE (rad) ↓ | mAVE (m/s) ↓ | mAAE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PointPillars** | LiDAR | 39.83% | 53.01% | 0.4221 | 0.2786 | 0.4908 | 0.3157 | 0.1831 |
| **CenterPoint** | LiDAR | 56.93% | 65.22% | 0.2919 | 0.2563 | 0.3073 | 0.2810 | 0.1881 |
| **PillarNeSt** | LiDAR | 58.66% | 65.14% | 0.2932 | 0.2513 | 0.3162 | 0.3605 | 0.1974 |
| **FocalFormer3D** | LiDAR | **66.02%** | **70.64%** | **0.2796** | **0.2559** | **0.2912** | **0.2220** | **0.1882** |

> [!NOTE]
> * **mAP**: Mean Average Precision
> * **NDS**: nuScenes Detection Score (a consolidated metric combining mAP and error metrics)
> * **mATE / mASE / mAOE / mAVE / mAAE**: Mean Translation / Scale / Orientation / Velocity / Attribute Errors

---

## 2. Class-Wise Average Precision (AP) Breakdown (%)

| Object Class | PointPillars | CenterPoint | PillarNeSt | FocalFormer3D | Best Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Car** | 80.20% | 85.00% | 83.90% | **87.10%** | FocalFormer3D |
| **Truck** | 34.90% | 54.70% | 56.30% | **58.90%** | FocalFormer3D |
| **Bus** | 46.00% | 65.80% | 70.80% | **76.30%** | FocalFormer3D |
| **Trailer** | 26.10% | 34.60% | 35.80% | **44.50%** | FocalFormer3D |
| **Construction Vehicle** | 5.40% | 15.90% | 19.90% | **27.40%** | FocalFormer3D |
| **Pedestrian** | 72.10% | 84.50% | 81.90% | **87.10%** | FocalFormer3D |
| **Motorcycle** | 37.70% | 55.80% | 64.50% | **74.70%** | FocalFormer3D |
| **Bicycle** | 9.90% | 36.50% | 46.70% | **60.30%** | FocalFormer3D |
| **Traffic Cone** | 33.60% | 68.90% | 61.10% | **73.90%** | FocalFormer3D |
| **Barrier** | 52.30% | 67.60% | 65.60% | **70.00%** | FocalFormer3D |

---

## 3. Key Observations

1. **FocalFormer3D Dominance**: FocalFormer3D outperforms all other baselines across all 10 object classes, achieving a top mAP of **66.02%** and NDS of **70.64%**. It exhibits particularly strong performance on smaller/more complex classes like **Bicycles** (60.30% AP vs. 36.50% for CenterPoint) and **Motorcycles** (74.70% AP).
2. **PillarNeSt vs. CenterPoint**: PillarNeSt (mAP: 58.66%) slightly outperforms CenterPoint (mAP: 56.93%) overall, but their NDS scores are extremely close (**65.14%** vs. **65.22%**) due to CenterPoint having slightly lower velocity error (0.2810 vs. 0.3605).
3. **PointPillars Baseline**: PointPillars has the lowest overall performance (mAP: 39.83%, NDS: 53.01%), which is expected due to its simpler single-stage architecture.

---

## 4. Slurm Evaluation Logs Directory

All detailed metrics, evaluation prints, and prediction dumps are logged under:
* [projects/adv_aug/logs/](file:///beegfs/jung/mmdet3d_legacy/projects/adv_aug/logs/)
