# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

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

Start distillation training (multi-GPU DDP):

```bash
bash scripts/train_distill.sh
# Device/hyperparameter overrides:
STUDENT_DEVICES=2,3,4,5 bash scripts/train_distill.sh
bash scripts/train_distill.sh --batch_size 4 --steps 10000
# Ablations:
bash scripts/train_distill.sh --feature_distill false --logit_distill true   # logit only
bash scripts/train_distill.sh --feature_distill false --logit_distill false  # task loss only
```

Run `distill.py` directly (debugging / single process):

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch --num_processes 4 --gpu_ids 2,3,4,5 \
  src/distill.py --config configs/distill_config.yaml
```

Evaluate student model:

```bash
bash scripts/eval_student.sh                          # auto-picks latest checkpoint
bash scripts/eval_student.sh --ckpt outputs/distill/checkpoints/last \
    --device cuda:2 --task libero_spatial --n_episodes 100
```

## Architecture Overview

This project distills XVLA (teacher, Florence2-based VLA) into SmolVLA (student, SmolVLM2-based VLA) on the LIBERO dataset, without modifying lerobot source code.

`src/distill.py` mirrors `lerobot/scripts/lerobot_train.py`:
- `TrainPipelineConfig` -> `DistillTrainPipelineConfig`
- `update_policy(...)` -> `update_distill(...)`
- identical metrics/checkpoint/sampler patterns, plus `adapters.pt` persistence

Hardware layout:
- Teacher (XVLA): loaded on each rank's `accelerator.device`, frozen fp16, no DDP
- Student (SmolVLA): all GPUs in `STUDENT_DEVICES`, DDP via `accelerator.prepare`

Loss:

```text
total_loss = alpha_task * task_loss
           + alpha_distill * (alpha_feature * feat_loss + alpha_logit * kl_loss)
```

Warmup (`warmup_steps`) applies only task loss.

## Feature Alignment and Hooks

- Trainable adapters in `src/adapters.py` align student features/actions to teacher space
- Sequence mismatch is handled by `align_seq_len(student_feat, teacher_feat)`
- Always use `MultiHookManager` (`src/hooks.py`) as a context manager so hooks are removed safely

Current feature sources:
- `student_vision`: hook on `student.model.vlm_with_expert.vlm.model.connector`
- `student_last`: read from `student.model.vlm_with_expert._last_expert_output`
- `teacher_vision`: `enc["vlm_features"]` from `teacher.model.forward_vlm(...)`
- `teacher_last`: hook on `teacher.model.transformer.blocks[-1]`

If XVLA exposes `.layers` instead of `.blocks`, update the teacher hook in `src/distill.py`.

## Configuration Checklist

Before training, verify `configs/distill_config.yaml`:
- `student_path`
- `distill.teacher_path`
- `dataset.root` / `dataset.repo_id`
- `output_dir`
- `wandb.enable` (run `wandb login` first if enabled)

All YAML fields can be overridden from CLI using dotted args, e.g.:

```bash
bash scripts/train_distill.sh --distill.alpha_task 0.8
```

## Checkpoints

Checkpoints follow LeRobot `save_checkpoint` layout under:

```text
outputs/distill/checkpoints/<step>/
```

Each checkpoint includes standard model/training state plus `adapters.pt`.
