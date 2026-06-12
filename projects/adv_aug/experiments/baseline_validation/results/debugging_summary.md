# Debugging Summary

## 1. What caused the original error
The original error `Exception: Error: Invalid box type: None` was caused by the models returning exactly zero predicted bounding boxes across the entire validation dataset. This happened because the `data/nuscenes` symlink was pointing to an incomplete/ongoing dataset extraction job, and the fallback directory (`data/nuscenes?`) contained only metadata (`.pkl` files) but not the actual LiDAR `.bin` sweep files. This caused mmcv's `LoadPointsFromFile` to silently fail and load empty point clouds, resulting in empty predictions and the subsequent evaluation crash.

## 2. What was changed
1. Waited for the extraction job `20591736` to finish extracting the full NuScenes dataset.
2. Corrected the dataset repository access point by pointing the `data/nuscenes` symlink directly to `/beegfs/jung/data/nuscenes/nuscenes`, ensuring both `.pkl` and LiDAR `.bin` files are correctly resolved.
3. Created `experiments/baseline_validation/run_experiment.sbatch` to output logs specifically to the baseline validation logs directory, ensuring a clean debugging workspace.

## 3. Which commands produced the final successful results
TBD

## 4. Final observed metrics for all four models
TBD

## 5. Any remaining concerns
TBD
