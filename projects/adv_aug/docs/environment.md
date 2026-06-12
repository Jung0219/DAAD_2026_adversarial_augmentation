# Environment & Cluster Configuration

## 1. Active Codebase and Environment

| Property | Value |
|----------|-------|
| **Repo root** | `DAAD_2026_adversarial_augmentation` |
| **Project root** | `projects/adv_aug` |
| **Framework** | Legacy MMDetection3D-style codebase (mmcv ≤ 1.7.1) with project plugins |
| **Conda env** | `AA` |
| **Python** | 3.8 |
| **PyTorch** | 1.12.1 |
| **CUDA** | 11.8.0 (module) / 11.3 (built against) |
| **Dataset** | nuScenes under `data/nuscenes/` |

### Runtime Exports (required in all Slurm scripts)
```bash
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD=/home/btq48260/.miniforge/lib/libstdc++.so.6
```

### Cluster Modules to Load
```bash
module load GCC/11.3.0
module load CUDA/11.8.0
```

---

## 2. SLURM Cluster Configuration

### Two-Cluster Policy

| Cluster | Use For | Resources | Queue Time |
|---------|---------|-----------|------------|
| **PLEIADES** (local school HPC) | Framework verification, smoke tests, debugging | 5× A100 GPUs | Low |
| **CLAIX** (supercomputing) | Full-scale multi-epoch training runs | 10+ H100 GPUs | 1–2 days |

> Verify on PLEIADES first. Only move to CLAIX once the pipeline is confirmed working.

### PLEIADES Settings
```
Partition:  gpu
Account:    etechnik_gpu
```

### CLAIX Settings
```
Partition:  c23g
Account:    default
```

---

## 3. SBATCH File Templates

Use these as the starting point for any new submission script. Only the
`#SBATCH` header and environment setup are server-specific — the actual
python/bash commands go below.

### PLEIADES Template
```bash
#!/usr/bin/env bash
#SBATCH --job-name=<job_name>
#SBATCH --output=projects/adv_aug/runs/<EXP-XXX>/logs/<job_name>_%j.out
#SBATCH --error=projects/adv_aug/runs/<EXP-XXX>/logs/<job_name>_%j.err
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -e

cd /beegfs/jung/DAAD_2026_adversarial_augmentation

source ~/.bashrc
conda activate AA_legacy

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p projects/adv_aug/runs/<EXP-XXX>/logs

# --- your commands below ---
```

### CLAIX Template
```bash
#!/usr/bin/env bash
#SBATCH --job-name=<job_name>
#SBATCH --output=projects/adv_aug/runs/<EXP-XXX>/logs/<job_name>_%j.out
#SBATCH --error=projects/adv_aug/runs/<EXP-XXX>/logs/<job_name>_%j.err
#SBATCH --partition=c23g
#SBATCH --account=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e

cd /beegfs/jung/DAAD_2026_adversarial_augmentation

module load GCC/11.3.0 CUDA/11.8.0

source /home/btq48260/.miniforge/etc/profile.d/conda.sh
conda activate AA

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export LD_PRELOAD=/home/btq48260/.miniforge/lib/libstdc++.so.6

mkdir -p projects/adv_aug/runs/<EXP-XXX>/logs

# --- your commands below ---
```

### Key Differences

| Setting | PLEIADES | CLAIX |
|---------|----------|-------|
| `--partition` | `gpu` | `c23g` |
| `--account` | `etechnik_gpu` | `default` |
| `--mem` | not required | `32G` |
| Module load | not required | `GCC/11.3.0 CUDA/11.8.0` |
| Conda activate | `source ~/.bashrc` then `conda activate AA_legacy` | source miniforge conda.sh then `conda activate AA` |
| `LD_PRELOAD` | not required | required (`libstdc++.so.6`) |


### Training Schedule: `configs/_base_/schedules/schedule_2x.py`
```python
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=1000,
                 warmup_ratio=1.0/1000, step=[20, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)
```

# CLAIX Storage and Runtime Rules

This repository is used on RWTH CLAIX.

## Environment Model

* `login23-1` is a **login node**: use it only for editing, inspecting files, light checks, and submitting Slurm jobs.
* Real training/evaluation must run on **compute nodes** through Slurm.
* `/hpcwork/rwth2049` is **shared project storage**, not a compute node.
* Temporary compute-node storage such as `$TMPDIR`, `$TMP`, local SSD, or BeeOND is fast but disposable.

## Important Paths

```text
User ID: btq48260
Main login node: login23-1
Project storage: /hpcwork/rwth2049
Adversarial data: /hpcwork/rwth2049/adv_data
```

## Storage Constraint

The critical constraint is **file count**. The project is already using about 4.16 million files out of a 5.12 million soft limit.

Do not blindly unzip or generate millions of small files under `/hpcwork/rwth2049`.

## Required Workflow for Large Jobs

For large datasets, extracted archives, adversarial data generation, or many-small-file workloads:

```text
/hpcwork/rwth2049
  -> copy or selectively untar needed subset to $TMPDIR/BeeOND
  -> run computation from temp storage
  -> copy important outputs back to /hpcwork/rwth2049
  -> let temp data disappear after job ends
```

Important: anything left only in `$TMPDIR` or BeeOND may be deleted after the job.

## When Direct `/hpcwork` Reads Are Okay

Directly reading from `/hpcwork/rwth2049` is acceptable for:

* small tests
* config/debug checks
* baseline verification
* short Slurm jobs
* jobs that do not create many temporary files

Do not copy hundreds of GB for tiny tests.

## Rules for Agents

1. Never run heavy training or evaluation on `login23-1`.
2. Use Slurm for compute-heavy work.
3. Do not store datasets, checkpoints, or generated data in `/home/btq48260`.
4. Treat `/hpcwork/rwth2049` as permanent project storage.
5. Avoid creating many permanent tiny files under `/hpcwork/rwth2049`.
6. For large extraction or adversarial generation, use `$TMPDIR`/BeeOND during the job.
7. Always copy important outputs back to `/hpcwork/rwth2049` before the job exits.
8. Do not delete shared data under `/hpcwork/rwth2049` without explicit confirmation.
9. Prefer selective extraction over extracting full datasets.
10. For Waymo-scale data, do not stage the full ~2.2 TB dataset to one node. Extract only required subsets such as LiDAR if possible.

## Useful Commands

Check project quota:

```bash
r_quota -u rwth2049
```

Check personal quota:

```bash
r_quota -u $USER
```

Check top-level storage usage:

```bash
du -h --max-depth=1 /hpcwork/rwth2049 | sort -h
```

Check top-level file counts:

```bash
find /hpcwork/rwth2049 -maxdepth 1 -mindepth 1 -type d -exec sh -c 'echo -n "$1 "; find "$1" -type f | wc -l' sh {} \;
```