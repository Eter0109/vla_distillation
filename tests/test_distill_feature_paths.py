from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
LEROBOT_SRC = ROOT.parent / "lerobot" / "src"
for path in (SRC_DIR, LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapters import DistillAdapters  # noqa: E402
from distill import (  # noqa: E402
    DistillConfig,
    apply_student_policy_overrides,
    configure_distill_branches_for_student,
    update_distill,
)


class _Accelerator:
    device = torch.device("cpu")
    is_main_process = False
    num_processes = 1
    sync_gradients = True

    def accumulate(self, *args):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class _StudentVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(connector=nn.Linear(3, 3))


class _StudentActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_with_expert = SimpleNamespace(vlm=_StudentVLM())
        self.action_out_proj = nn.Linear(2, 7)

    def sample_actions(self, *args, **kwargs):
        raise AssertionError("expert feature distill must reuse task forward, not sample_actions()")


class _StudentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _StudentActionModel()
        self.config = SimpleNamespace()

    def forward(self, batch):
        vision_in = batch["vision"]
        expert_in = batch["expert"]
        vision_out = self.model.vlm_with_expert.vlm.model.connector(vision_in)
        action_out = self.model.action_out_proj(expert_in)
        return (vision_out.mean() + action_out.mean()), {}


class _TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(norm=nn.LayerNorm(3))

    def _get_target_dtype(self):
        return torch.float32

    def generate_actions(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        hidden = torch.ones(batch_size, 3, 3)
        self.transformer.norm(hidden)
        return torch.zeros(batch_size, 3, 20)


class _TeacherPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TeacherModel()
        self.config = SimpleNamespace(num_denoising_steps=1)

    def _build_model_inputs(self, batch):
        return {
            "input_ids": batch["input_ids"],
            "image_input": None,
            "image_mask": None,
            "domain_id": None,
            "proprio": None,
        }


def test_student_policy_overrides_are_applied_before_loading():
    cfg = SimpleNamespace(
        distill=DistillConfig(
            student_train_expert_only=False,
            student_freeze_vision_encoder=False,
            student_use_cache=False,
        ),
        policy=SimpleNamespace(
            train_expert_only=True,
            freeze_vision_encoder=True,
            use_cache=True,
        ),
    )

    applied = apply_student_policy_overrides(cfg)

    assert applied == {
        "train_expert_only": False,
        "freeze_vision_encoder": False,
        "use_cache": False,
    }
    assert cfg.policy.train_expert_only is False
    assert cfg.policy.freeze_vision_encoder is False
    assert cfg.policy.use_cache is False


def test_real_vlm_distill_fails_if_overrides_leave_vlm_frozen():
    cfg = SimpleNamespace(distill=DistillConfig())
    cfg.distill.feature_distill = True
    cfg.distill.vision_feature_distill = True
    cfg.distill.student_train_expert_only = False
    cfg.distill.student_freeze_vision_encoder = False

    student = _StudentPolicy()
    for param in student.model.vlm_with_expert.vlm.parameters():
        param.requires_grad = False

    with pytest.raises(RuntimeError, match="student VLM has no trainable parameters"):
        configure_distill_branches_for_student(cfg, student)


def test_expert_feature_distill_reuses_task_forward_without_student_sample_actions():
    cfg = SimpleNamespace(distill=DistillConfig(), grad_accum_steps=1)
    cfg.distill.feature_distill = True
    cfg.distill.vision_feature_distill = False
    cfg.distill.expert_feature_distill = True
    cfg.distill.logit_distill = False
    cfg.distill.warmup_steps = 0
    cfg.distill.distill_ramp_steps = 0
    cfg.distill.student_expert_dim = 2
    cfg.distill.teacher_expert_dim = 3

    student = _StudentPolicy()
    teacher = _TeacherPolicy()
    adapters = DistillAdapters(
        student_expert_dim=2,
        teacher_expert_dim=3,
        enable_vision_distill=False,
        enable_expert_distill=True,
        enable_logit_distill=False,
    )
    optimizer = torch.optim.SGD(
        list(student.parameters()) + list(adapters.parameters()),
        lr=0.01,
    )
    student_batch = {
        "vision": torch.randn(2, 4, 3),
        "expert": torch.randn(2, 3, 2),
    }
    teacher_batch = {"input_ids": torch.ones(2, 1, dtype=torch.long)}

    _, output = update_distill(
        train_metrics=SimpleNamespace(),
        student=student,
        teacher=teacher,
        student_batch=student_batch,
        teacher_batch=teacher_batch,
        adapters=adapters,
        optimizer=optimizer,
        grad_clip_norm=0.0,
        micro_step=0,
        optimizer_step=0,
        cfg=cfg,
        accelerator=_Accelerator(),
    )

    assert "loss_expert_feat" in output
    assert output["loss_distill"] > 0
