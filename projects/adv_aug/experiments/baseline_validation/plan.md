# Baseline Validation Agent Plan

## Goal

Reproduce baseline nuScenes metrics for the four models:

| Model         | Target mAP | Target NDS |
| ------------- | ---------: | ---------: |
| PointPillars  |       30.5 |       45.3 |
| CenterPoint   |       58.0 |       65.5 |
| PillarNeSt    |       64.3 |       70.4 |
| FocalFormer3D |       68.7 |       72.6 |

The task is complete only when all four models produce reasonable evaluation results close to the target metrics. (within +- 10)

Continue debugging until this goal has been reached.

---

## Current Issue

The test script launches correctly and appears to load the models, but evaluation fails with:

```text
Exception: Error: Invalid box type: None
```

The script currently used for testing is:

```bash
/beegfs/jung/DAAD_2026_adversarial_augmentation/projects/adv_aug/scripts/submit/run_experiment.sbatch
```

The main debugging target is to fix the evaluation pipeline so that valid nuScenes metrics are produced.

---

## Artifact Rule

Store all debugging artifacts under:

```bash
/beegfs/jung/DAAD_2026_adversarial_augmentation/experiments/baseline_validation
```

This includes logs, copied scripts, temporary scripts, result files, debug outputs, notes, and config snapshots.

Do not scatter temporary files throughout the main repository.

---

## Dataset Setup

Before running any tests, wait for the nuScenes extraction job to finish:

```text
JOBID: 20591736
NAME: extract-
STATUS: running
NODE: wn21053
```

Check the job:

```bash
squeue -j 20591736
```

When it disappears from `squeue`, confirm completion:

```bash
sacct -j 20591736 --format=JobID,JobName,State,ExitCode,Elapsed
```

Only continue if the job completed successfully.

The extracted dataset should be located at:

```bash
/beegfs/jung/data/nuscenes
```

Create the repository access point:

```bash
mkdir -p /beegfs/jung/DAAD_2026_adversarial_augmentation/data

ln -s /beegfs/jung/data/nuscenes \
  /beegfs/jung/DAAD_2026_adversarial_augmentation/data/nuscenes
```

Verify:

```bash
readlink -f /beegfs/jung/DAAD_2026_adversarial_augmentation/data/nuscenes
```

It should resolve to:

```bash
/beegfs/jung/data/nuscenes
```

Use this path as the dataset access point:

```bash
/beegfs/jung/DAAD_2026_adversarial_augmentation/data/nuscenes
```

---

## Execution Strategy

Start with one model first, preferably PointPillars.

Do not launch all four models blindly before the evaluation pipeline works for one model.

For each model:

1. Prepare an isolated output directory under `experiments/baseline_validation`.
2. Run the test job.
3. Wait for the SLURM job to finish.
4. Check the job status with `sacct`.
5. Inspect stdout and stderr logs.
6. Inspect generated result files.
7. If evaluation failed, debug and rerun.
8. If evaluation succeeded, record mAP and NDS.
9. Compare the observed metrics against the target metrics.

A submitted SLURM job is not success. Success means the job finishes, evaluation completes, and valid metrics are produced.

---

## Debugging Notes

For the current error:

```text
Invalid box type: None
```

Investigate:

1. Whether the result file contains `None` entries.
2. Whether predictions are being serialized correctly.
3. Whether the config, checkpoint, dataset split, and class mappings match.
4. Whether the nuScenes evaluation code receives valid prediction boxes.
5. Whether the issue comes from model output, result formatting, or evaluation conversion.

Make minimal changes. If core code is modified, document exactly what changed and why.

---

## Required Final Outputs

Create:

```bash
/beegfs/jung/DAAD_2026_adversarial_augmentation/experiments/baseline_validation/results/baseline_metrics.md
```

with:

```markdown
| Model | Observed mAP | Observed NDS | Target mAP | Target NDS | Status | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| PointPillars | TBD | TBD | 30.5 | 45.3 | Pending |  |
| CenterPoint | TBD | TBD | 58.0 | 65.5 | Pending |  |
| PillarNeSt | TBD | TBD | 64.3 | 70.4 | Pending |  |
| FocalFormer3D | TBD | TBD | 68.7 | 72.6 | Pending |  |
```

Also create a short debugging summary:

```bash
/beegfs/jung/DAAD_2026_adversarial_augmentation/experiments/baseline_validation/results/debugging_summary.md
```

Include:

1. What caused the original error.
2. What was changed.
3. Which commands produced the final successful results.
4. Final observed metrics for all four models.
5. Any remaining concerns.

---

## Completion Criteria

The task is complete only when:

1. nuScenes extraction is complete.
2. The dataset symlink is valid.
3. All four models run through evaluation.
4. All four models produce valid mAP and NDS.
5. The results are reasonably close to the target metrics.
6. The debugging summary and metrics table are saved under `experiments/baseline_validation`.

Continue debugging until these criteria are satisfied.
