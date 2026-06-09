# AGENTS.md — Project Context & Index

Last updated: 2026-06-09. Read this file first. It tells you where everything is.

---

## 1. Project Goal

Develop and evaluate **runtime adversarial augmentation** for LiDAR-based 3D object
detection. The research question is whether gradient-guided, realism-constrained
point/voxel perturbations can improve robustness against occlusion, sparsity, and
local blockage without unacceptable clean-performance loss on nuScenes metrics
(mAP, NDS).

**Active phase: Phase 2 — Pipeline Familiarization and Minimal Gradient Sample.**
Baseline training/evaluation for all four models has been validated. Adversarial
training is still an implementation target.

**Two-server setup:** This project runs across two HPC clusters — use the right one for the right task:
- **PLEIADES** (local school HPC, 5× A100) — smoke tests, debugging, framework verification only.
- **CLAIX** (supercomputer, 10+ H100, queue ~1–2 days) — full-scale training runs only, after PLEIADES verification.

See [`docs/environment.md`](docs/environment.md) for partition names, accounts, and module setup.

---

## 2. Document Index

| Document | Location | What it contains |
|----------|----------|-----------------|
| **This file** | `AGENTS.md` | Entry point, pointers, practical rules |
| **Research plan** | [`docs/adversarial_augmentation_plan.md`](docs/adversarial_augmentation_plan.md) | Phase roadmap, status tracking, baseline metrics, attack code status |
| **Architecture reference** | [`docs/architecture.md`](docs/architecture.md) | Repo structure, PointPillars forward path, data pipeline, NuScenes config, voxelization autograd boundary, plugin registry, config chains, file quick reference |
| **Environment & cluster** | [`docs/environment.md`](docs/environment.md) | Conda env, runtime exports, SLURM cluster policy (PLEIADES vs CLAIX), training schedule, **sbatch templates for both servers** |
| **Scripts & commands** | [`docs/scripts.md`](docs/scripts.md) | Model/checkpoint table, all runnable sbatch and python commands |
| **Local setup** | [`docs/LOCAL_ASSETS.md`](docs/LOCAL_ASSETS.md) | Datasets, model weights, compiled extensions — what to restore after cloning |
| **Experiment guide & template** | [`experiments/README.md`](experiments/README.md) | Workflow SOP, SLURM policy, guardrails, report format, and blank plan template |
| **Eval report** | [`projects/adv_aug/docs/evaluation_report.md`](projects/adv_aug/docs/evaluation_report.md) | Full per-class nuScenes baseline validation results |

---

## 3. Practical Rules for Agents

1. **Read this file first.** Then pull only the doc you need — don't explore blindly.
2. Before editing any code, run `git status --short`. Do not revert existing local modifications unless explicitly asked.
3. Treat **Phase 2** as active. Do not claim adversarial training is implemented until there is a real integrated training path.
4. Start with PointPillars/CenterPoint for shared voxel-level logic. Inspect FocalFormer3D and PillarNeSt separately — they use project plugins.
5. Keep attack code **model-side**. CPU dataloader transforms are for normal augmentation; gradient-guided adversarial augmentation belongs in the forward/train path.
6. When running Slurm jobs, ensure checkpoint files exist and output directories under `projects/adv_aug/runs/.../logs` exist (or are created by the script).
7. Record any new experiment command, config, checkpoint, seed, output path, and metric in `docs/` or the plan file.
8. The NuScenes **mini** split requires generating info pkl files first — no pre-existing mini pkl files are available.
9. **Experiment planning workflow:** Always follow [`experiments/README.md`](experiments/README.md) for the workflow SOP and plan template. Create the spec doc first. **Do NOT start executing until the user explicitly confirms.**
10. **Writing sbatch scripts:** Always use the server-specific templates in [`docs/environment.md`](docs/environment.md) §3 as the starting point — do not guess at partition names, accounts, or environment setup.
