# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda activate lerobot
# lerobot source: /hqlab/workspace/zhaozy/lerobot/src  (injected via PYTHONPATH, not installed)
```

The scripts automatically set `PYTHONPATH=/hqlab/workspace/zhaozy/lerobot/src:src/`.
When running Python directly, set it manually:
```bash
export PYTHONPATH="/hqlab/workspace/zhaozy/lerobot/src:src/:${PYTHONPATH:-}"
```

## Key Commands

**Start distillation training (4-GPU DDP):**
```bash
bash scripts/train_distill.sh
# With overrides:
bash scripts/train_distill.sh --teacher_device cuda:1 --student_devices 2,3,4,5 --alpha 0.5 --batch_size 8 --steps 30000 --accum_steps 4
# Feature distill only / task loss only:
bash scripts/train_distill.sh --feature_distill false --logit_distill true
bash scripts/train_distill.sh --feature_distill false --logit_distill false
```

**Evaluate student model:**
```bash
bash scripts/eval_student.sh                                      # auto-picks latest checkpoint
bash scripts/eval_student.sh --ckpt outputs/checkpoints/best --device cuda:2 --task libero_spatial --n_episodes 100
```

**Run distill.py directly (for debugging):**
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 src/distill.py --config configs/distill_config.yaml
```

## Architecture Overview

This project distills XVLA (teacher, ~large Florence2-based VLA) into SmolVLA (student, ~500M SmolVLM2-based VLA) on the LIBERO robot manipulation dataset, without modifying lerobot source code.

### Hardware layout
- Teacher (XVLA): `cuda:1` — frozen, fp16 inference only, never wrapped in DDP
- Student (SmolVLA): `cuda:2,3,4,5` — 4-GPU DDP, mixed precision training

### Training pipeline (`src/distill.py`)
1. Load teacher (frozen, fp16) and student (trainable, DDP-wrapped)
2. Build `DistillAdapters` (projection layers on student devices)
3. Per-step: student forward → register hooks → teacher forward (no_grad) → compute 3-part loss → AMP backward → gradient accumulation → optimizer step

### Loss formula
```
total_loss = alpha_task × MSE(student_action, GT_action)
           + alpha_distill × (alpha_feature × feat_loss + alpha_logit × kl_loss)
```
During warmup (default first 500 steps), only task loss is active.

### Feature alignment (`src/adapters.py`)
All adapters are trainable linear projections (Linear + LayerNorm) that map student → teacher dimension space. They are optimized jointly with the student.

| Adapter | Student dim | Teacher dim |
|---------|-------------|-------------|
| `VisionFeatureAdapter` | 576 (SigLIP embed_image) | 1024 (Florence2 vlm_features) |
| `ActionExpertFeatureAdapter` | 288 (lm_expert hidden) | 1024 (transformer hidden) |
| `ActionAdapter` | 7 (action_dim) | 20 (action_dim) |

When `in_dim == out_dim`, adapter degrades to identity (no extra parameters).

### Hook system (`src/hooks.py`)
`FeatureHook` attaches `register_forward_hook` to an `nn.Module` and stores `output.detach().clone()`. `MultiHookManager` is a context manager managing a dict of named hooks — always use as `with MultiHookManager({...}) as hooks:` to guarantee cleanup.

Hook attachment points:
| Key | Module path |
|-----|-------------|
| `student_vision` | `student.model.vlm_with_expert.vlm.model.vision_model` |
| `student_last` | `student.model.vlm_with_expert.lm_expert.layers[-1]` |
| `teacher_vision` | `teacher.model.vlm.vision_tower` |
| `teacher_last` | `teacher.model.transformer.blocks[-1]` |

If XVLA uses `.layers` instead of `.blocks`, change line 341 of `src/distill.py` or add `teacher.transformer_last_layer_attr` to `configs/distill_config.yaml`.

### Sequence length mismatch
Student and teacher token counts differ. `align_seq_len(student_feat, teacher_feat)` in `adapters.py` truncates or zero-pads the student feature to match teacher length before MSE computation.

### Checkpoint format
Saved via `model.save_pretrained()` (lerobot-compatible `from_pretrained` format):
```
outputs/checkpoints/
├── step_0002000/  {config.json, model.safetensors}
├── step_0004000/  ...
└── best/          {config.json, model.safetensors}  ← lowest total_loss seen
```
Keeps last 3 step checkpoints + `best/` (configurable via `keep_last_n_checkpoints`).

### Inference / evaluation wrapper (`src/student_policy_wrapper.py`)
`StudentPolicyWrapper` subclasses `SmolVLAPolicy` unchanged (name = `"smolvla"`). Use `load_student_policy(ckpt_path, device)` to load for eval. Eval script falls back to model-load-only validation if LIBERO gym is unavailable.

## Configuration (`configs/distill_config.yaml`)

All CLI flags override YAML values at runtime. Key fields to verify before running:
- `teacher.path` / `student.path` — pretrained model dirs
- `dataset.path` / `dataset.repo_id` — local LIBERO dataset
- `teacher.device` / `student.devices` — GPU assignment
- `wandb.enabled` — requires `wandb login` beforehand
