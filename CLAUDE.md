# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda activate lerobot
# lerobot source: /hqlab/workspace/zhaozy/lerobot/src  (injected via PYTHONPATH, not installed)
```

Scripts automatically set `PYTHONPATH=/hqlab/workspace/zhaozy/lerobot/src:src/`.
When running Python directly:
```bash
export PYTHONPATH="/hqlab/workspace/zhaozy/lerobot/src:src/:${PYTHONPATH:-}"
```

## Key Commands

**Start distillation training (4-GPU DDP):**
```bash
bash scripts/train_distill.sh
# Device/hyperparameter overrides:
TEACHER_DEVICE=cuda:1 STUDENT_DEVICES=2,3,4,5 bash scripts/train_distill.sh
bash scripts/train_distill.sh --batch_size 4 --steps 10000
# Ablations:
bash scripts/train_distill.sh --feature_distill false --logit_distill true   # logit only
bash scripts/train_distill.sh --feature_distill false --logit_distill false  # task loss only
```

**Run distill.py directly (debugging / single process):**
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --num_processes 4 --gpu_ids 2,3,4,5 \
  src/distill.py --config configs/distill_config.yaml
```

**Evaluate student model:**
```bash
bash scripts/eval_student.sh                          # auto-picks latest checkpoint
bash scripts/eval_student.sh --ckpt outputs/distill/checkpoints/last \
    --device cuda:2 --task libero_spatial --n_episodes 100
```

## Architecture Overview

This project distills XVLA (teacher, Florence2-based VLA) into SmolVLA (student, ~500M SmolVLM2-based VLA) on the LIBERO robot manipulation dataset, **without modifying lerobot source code**.

### Framework alignment

`src/distill.py` mirrors `lerobot/scripts/lerobot_train.py` structure exactly:

| lerobot_train.py | distill.py |
|---|---|
| `TrainPipelineConfig` | `DistillTrainPipelineConfig` (subclass, adds `DistillConfig` + `student_path`) |
| `update_policy(...)` | `update_distill(...)` |
| `@parser.wrap() def train(...)` | `@parser.wrap() def train_distill(...)` |
| `MetricsTracker` / `AverageMeter` | identical usage |
| `save_checkpoint` / `load_training_state` | identical usage; additionally saves `adapters.pt` |
| `EpisodeAwareSampler` + `cycle` | identical |

### Hardware layout
- Teacher (XVLA): `cuda:1` — frozen fp16, never wrapped in DDP, only loaded on main process
- Student (SmolVLA): `cuda:2,3,4,5` — 4-GPU DDP via `accelerator.prepare`, mixed precision

### Training data flow per step
1. `next(dl_iter)` → `preprocessor(batch)` for student; `teacher_preprocessor(batch)` on main process only
2. Student forward → task loss (`student.forward(batch)`)
3. Teacher forward (main process only, `no_grad`, `fp16 autocast`) with `MultiHookManager` capturing intermediate features
4. `accelerator.broadcast(teacher_feat, src=0)` — synchronizes teacher features to all DDP ranks
5. Compute feature MSE (via `DistillAdapters`) + logit KL divergence
6. `accelerator.backward(total_loss)` → grad clip → optimizer step

### Loss formula
```
total_loss = alpha_task × task_loss
           + alpha_distill × (alpha_feature × feat_loss + alpha_logit × kl_loss)
```
During warmup (default first 500 steps), only task loss is active.

### Feature alignment (`src/adapters.py`)
Trainable projections (Linear + LayerNorm) map student → teacher feature space. Optimized jointly with the student via a shared `AdamW` optimizer that holds both `student.parameters()` and `adapters.parameters()`.

| Adapter | Student dim | Teacher dim |
|---------|-------------|-------------|
| `VisionFeatureAdapter` | 576 (SigLIP) | 1024 (Florence2) |
| `ActionExpertFeatureAdapter` | 288 (lm_expert hidden) | 1024 (transformer hidden) |
| `ActionAdapter` | 7 (action_dim) | 20 (action_dim) |

When `in_dim == out_dim`, the adapter is `nn.Identity` (no extra parameters).

Sequence length mismatch between student and teacher tokens is handled by `align_seq_len(student_feat, teacher_feat)` in `adapters.py` — truncates or zero-pads to match teacher length.

### Hook system (`src/hooks.py`)
`FeatureHook` wraps `register_forward_hook` and stores `output.detach().clone()`. Always use `MultiHookManager` as a context manager to guarantee hook removal:

```python
with MultiHookManager({"key": module}) as hooks:
    model(batch)
    feat = hooks["key"].output
```

Hook attachment points:
| Key | Module path |
|-----|-------------|
| `student_vision` | `student.model.vlm_with_expert.vlm.model.vision_model` |
| `student_last` | `student.model.vlm_with_expert.lm_expert.layers[-1]` |
| `teacher_vision` | `teacher.model.vlm.vision_tower` |
| `teacher_last` | `teacher.model.transformer.blocks[-1]` |

If XVLA uses `.layers` instead of `.blocks`, update the teacher hook in `src/distill.py` `update_distill()`.

### Checkpoint format
Uses LeRobot's standard `save_checkpoint` layout under `output_dir/checkpoints/<step>/`:
```
outputs/distill/checkpoints/
├── 002000/
│   ├── pretrained_model/  {config.json, model.safetensors, train_config.json}
│   ├── training_state/    {optimizer_state.safetensors, scheduler_state.json, ...}
│   └── adapters.pt        ← DistillAdapters weights (extra, non-standard)
└── last -> 002000/        (symlink, updated each save)
```

### Inference / evaluation (`src/student_policy_wrapper.py`)
`StudentPolicyWrapper` subclasses `SmolVLAPolicy` unchanged (policy type = `"smolvla"`). Use `load_student_policy(ckpt_path, device)` to load for eval.

## Configuration (`configs/distill_config.yaml`)

Top-level keys map directly to `TrainPipelineConfig` / `DistillTrainPipelineConfig` fields. Key fields to verify before running:

- `student_path` — pretrained SmolVLA directory
- `distill.teacher_path` / `distill.teacher_device` — XVLA model path and GPU
- `dataset.root` / `dataset.repo_id` — local LIBERO dataset
- `output_dir` — where checkpoints and logs land
- `wandb.enable` — requires `wandb login` beforehand

All YAML values can be overridden from CLI as dotted paths, e.g. `--distill.alpha_task 0.8`.
