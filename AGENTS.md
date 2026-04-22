# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the distillation pipeline: `distill.py` is the training entrypoint, `adapters.py` aligns teacher/student feature and action spaces, `hooks.py` captures intermediate tensors, and `student_policy_wrapper.py` exposes the student as a LeRobot policy. `configs/` stores runnable YAMLs for full runs and smoke tests. `scripts/` is the supported launch surface for train/eval/smoke flows. Keep regression coverage in `tests/`, especially `tests/test_action_alignment.py` and `tests/test_hooks.py`.

## Build, Test, and Development Commands
Use the `lerobot` conda environment and rely on the scripts to set `PYTHONPATH` for the sibling `../lerobot/src` checkout.

```bash
conda activate lerobot
bash scripts/smoke_test_distill.sh
bash scripts/train_distill.sh
bash scripts/eval_student.sh --ckpt outputs/distill/checkpoints/best
PYTHONPATH="../lerobot/src:src:${PYTHONPATH:-}" pytest -q tests/test_hooks.py tests/test_action_alignment.py
```

`scripts/smoke_test_distill.sh` is the fastest validation path. `scripts/train_distill.sh` launches multi-GPU training with `accelerate`; prefer it over calling `src/distill.py` directly except when debugging a single process.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and explicit type hints on public helpers. Keep changes localized to this repo; prefer adapting through `src/` rather than patching the sibling `lerobot` checkout. Keep config names aligned with CLI/YAML overrides such as `distill.alpha_vision_feature`.

## Testing Guidelines
Add or update `pytest` coverage with every behavior change. Name tests `test_<behavior>.py` and prefer small regression cases around tensor shapes, geometry conversion, hook capture, and normalization. Changes to action alignment belong in `tests/test_action_alignment.py`; hook and intermediate-feature changes belong in `tests/test_hooks.py`; run the targeted files before broader smoke validation.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects such as `Enable feature-only distill config and add hook regression tests`. Keep commit titles concise and behavior-focused. PRs should summarize the training impact, note config or environment assumptions, link related issues, and include concrete evidence such as pytest output, smoke-test results, or eval/checkpoint metrics.

## Environment & Configuration Tips
This project assumes a sibling `lerobot` checkout at `../lerobot/src`. Training scripts require CUDA and do not fall back to CPU. Verify `student_path`, `distill.teacher_path`, dataset roots, and `STUDENT_DEVICES` before long runs. Keep `distill_config.yaml` and `distill_smoke_test.yaml` semantically aligned when switching distillation strategy.
