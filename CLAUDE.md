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
- Teacher (XVLA): `accelerator.device` per rank — frozen fp16, loaded on every rank independently (no DDP)
- Student (SmolVLA): `cuda:2,3,4,5` — 4-GPU DDP via `accelerator.prepare`, mixed precision

### Training data flow per step
1. `next(dl_iter)` → `preprocessor(batch)` for student; `teacher_preprocessor(batch)` on every rank
2. Student forward → task loss (`student.forward(batch)`); hooks capture `student_vision` from `connector`
3. Teacher forward (`no_grad`, `fp16 autocast`) — `forward_vlm()` returns `vlm_features` directly; `MultiHookManager` captures `teacher_last` from `transformer.blocks[-1]`; student expert features read from `vlm_with_expert._last_expert_output` (cached attribute set during student forward)
4. Compute feature MSE (via `DistillAdapters`) + logit KL divergence
5. `accelerator.backward(total_loss)` → grad clip → optimizer step

### Loss formula
```
total_loss = alpha_task × task_loss
           + alpha_distill × (alpha_feature × feat_loss + alpha_logit × kl_loss)
```
During warmup (`warmup_steps`, set to `0` in `distill_config.yaml` / `500` in `DistillConfig` dataclass), only task loss is active.

### Feature alignment (`src/adapters.py`)
Trainable projections (Linear + LayerNorm) map student → teacher feature space. Optimized jointly with the student via a shared `AdamW` optimizer that holds both `student.parameters()` and `adapters.parameters()`.

| Adapter | Student dim | Teacher dim |
|---------|-------------|-------------|
| `VisionFeatureAdapter` | `student_vision_dim` (default 960, connector output) | `teacher_vision_dim` (default 1024) |
| `ActionExpertFeatureAdapter` | `student_expert_dim` (default 480, `_last_expert_output`) | `teacher_expert_dim` (default 1024) |
| `ActionAdapter` | `student_action_dim` (default 7) | `teacher_action_dim` (default 20) |

Dims are driven by `distill_config.yaml` fields and passed to `DistillAdapters(...)`. When `in_dim == out_dim`, the adapter is `nn.Identity` (no extra parameters).

Sequence length mismatch between student and teacher tokens is handled by `align_seq_len(student_feat, teacher_feat)` in `adapters.py` — truncates or zero-pads to match teacher length.

### Hook system (`src/hooks.py`)
`FeatureHook` wraps `register_forward_hook` and stores `output.detach().clone()`. Always use `MultiHookManager` as a context manager to guarantee hook removal:

```python
with MultiHookManager({"key": module}) as hooks:
    model(batch)
    feat = hooks["key"].output
```

Hook attachment points (only two active hooks; the other two features are captured differently):
| Key | How captured |
|-----|-------------|
| `student_vision` | Hook on `student.model.vlm_with_expert.vlm.model.connector` |
| `student_last` | Read from `student.model.vlm_with_expert._last_expert_output` (cached attr set during student forward, no hook) |
| `teacher_vision` | `enc["vlm_features"]` returned directly by `teacher.model.forward_vlm(...)` (no hook) |
| `teacher_last` | Hook on `teacher.model.transformer.blocks[-1]` |

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
