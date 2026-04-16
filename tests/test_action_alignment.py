from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adapters import (
    DistillAdapters,
    normalize_action_to_student_training_scale,
    xvla_libero_action20_to_action7,
    xvla_teacher_action20_to_student_rel7,
)


def _import_lerobot_xvla_rotation_step():
    lerobot_src = ROOT.parent / "lerobot" / "src"
    if not lerobot_src.exists():
        pytest.skip(f"lerobot source not found at expected path: {lerobot_src}")

    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    try:
        from lerobot.policies.xvla.processor_xvla import XVLARotation6DToAxisAngleProcessorStep
        from lerobot.types import TransitionKey
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Unable to import lerobot XVLA processor: {exc}")

    return XVLARotation6DToAxisAngleProcessorStep, TransitionKey


def _rot6d_from_matrix(rot: torch.Tensor) -> torch.Tensor:
    return torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)


def _rotz(angle_rad: float) -> torch.Tensor:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def test_xvla_libero_alignment_matches_lerobot_postprocessor():
    step_cls, transition_key = _import_lerobot_xvla_rotation_step()
    step = step_cls()

    torch.manual_seed(42)
    teacher_action = torch.randn(2, 6, 20, dtype=torch.float32)

    expected = step({transition_key.ACTION: teacher_action.reshape(-1, 20)})[transition_key.ACTION]
    expected = expected.reshape(2, 6, 7)
    actual = xvla_libero_action20_to_action7(teacher_action)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_teacher_abs20_to_student_rel7_geometry():
    state = torch.zeros(1, 20, dtype=torch.float32)
    action = torch.zeros(1, 20, dtype=torch.float32)

    # current pose
    state[:, :3] = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    state[:, 3:9] = _rot6d_from_matrix(torch.eye(3, dtype=torch.float32).unsqueeze(0))

    # target pose
    action[:, :3] = torch.tensor([[1.025, 1.975, 3.01]], dtype=torch.float32)
    action[:, 3:9] = _rot6d_from_matrix(_rotz(0.25).unsqueeze(0))
    action[:, 9:10] = torch.tensor([[0.75]], dtype=torch.float32)

    rel = xvla_teacher_action20_to_student_rel7(action, teacher_state=state)
    expected = torch.tensor([[0.5, -0.5, 0.2, 0.0, 0.0, 0.5, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(rel, expected, rtol=0.0, atol=2e-4)


def test_teacher_abs20_to_student_rel7_geometry_with_state8_axis_angle():
    state = torch.zeros(1, 8, dtype=torch.float32)
    action = torch.zeros(1, 20, dtype=torch.float32)

    # current pose from 8D state: pos(3) + axis-angle(3) + extra(2)
    state[:, :3] = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    state[:, 3:6] = torch.tensor([[0.0, 0.0, math.pi / 4.0]], dtype=torch.float32)
    state[:, 6:] = torch.tensor([[0.12, -0.34]], dtype=torch.float32)

    # target pose from teacher action: pos + rot6d + gripper
    action[:, :3] = torch.tensor([[1.025, 1.975, 3.01]], dtype=torch.float32)
    action[:, 3:9] = _rot6d_from_matrix(_rotz(math.pi / 4.0 + 0.2).unsqueeze(0))
    action[:, 9:10] = torch.tensor([[0.75]], dtype=torch.float32)

    rel = xvla_teacher_action20_to_student_rel7(action, teacher_state=state)
    expected = torch.tensor([[0.5, -0.5, 0.2, 0.0, 0.0, 0.4, 1.0]], dtype=torch.float32)
    torch.testing.assert_close(rel, expected, rtol=0.0, atol=2e-4)


def test_teacher_abs20_to_student_rel7_supports_bt_and_bd():
    torch.manual_seed(0)
    teacher_action_bt = torch.randn(3, 4, 20, dtype=torch.float32)
    teacher_state_bd = torch.randn(3, 20, dtype=torch.float32)
    teacher_action_bd = torch.randn(5, 20, dtype=torch.float32)
    teacher_state_bd_2 = torch.randn(5, 20, dtype=torch.float32)

    out_bt = xvla_teacher_action20_to_student_rel7(teacher_action_bt, teacher_state=teacher_state_bd)
    out_bd = xvla_teacher_action20_to_student_rel7(teacher_action_bd, teacher_state=teacher_state_bd_2)

    assert out_bt.shape == (3, 4, 7)
    assert out_bd.shape == (5, 7)


def test_teacher_abs20_to_student_rel7_supports_state8_bt_and_bd():
    torch.manual_seed(1)
    teacher_action_bt = torch.randn(3, 4, 20, dtype=torch.float32)
    teacher_state_bd = torch.randn(3, 8, dtype=torch.float32)
    teacher_action_bd = torch.randn(5, 20, dtype=torch.float32)
    teacher_state_bd_2 = torch.randn(5, 8, dtype=torch.float32)

    out_bt = xvla_teacher_action20_to_student_rel7(teacher_action_bt, teacher_state=teacher_state_bd)
    out_bd = xvla_teacher_action20_to_student_rel7(teacher_action_bd, teacher_state=teacher_state_bd_2)

    assert out_bt.shape == (3, 4, 7)
    assert out_bd.shape == (5, 7)


def test_teacher_target_is_normalized_to_student_scale():
    stats = {
        "mean": torch.tensor([0.1, -0.2, 0.3, 0.0, 0.0, 0.2, -0.4], dtype=torch.float32),
        "std": torch.tensor([0.5, 0.25, 0.2, 0.3, 0.3, 0.4, 0.6], dtype=torch.float32),
    }
    adapters = DistillAdapters(
        enable_vision_distill=False,
        enable_expert_distill=False,
        enable_logit_distill=True,
        student_action_dim=7,
        teacher_action_dim=20,
        action_align_mode="teacher_abs20_to_student_rel7",
        student_action_stats=stats,
        student_action_norm_mode="MEAN_STD",
    )

    state = torch.zeros(1, 20, dtype=torch.float32)
    action = torch.zeros(1, 20, dtype=torch.float32)
    state[:, 3:9] = _rot6d_from_matrix(torch.eye(3, dtype=torch.float32).unsqueeze(0))
    action[:, :3] = torch.tensor([[0.6, 0.2, -0.1]], dtype=torch.float32)
    action[:, 3:9] = _rot6d_from_matrix(_rotz(math.pi / 3.0).unsqueeze(0))
    action[:, 9:10] = torch.tensor([[0.25]], dtype=torch.float32)

    raw_target = xvla_teacher_action20_to_student_rel7(action, teacher_state=state)
    expected = normalize_action_to_student_training_scale(
        raw_target,
        stats=stats,
        norm_mode="MEAN_STD",
    )
    actual = adapters.adapt_teacher_action(action, teacher_state=state)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_distill_adapter_outputs_aligned_7d_for_regression_backward():
    stats = {
        "mean": torch.zeros(7, dtype=torch.float32),
        "std": torch.ones(7, dtype=torch.float32),
    }
    adapters = DistillAdapters(
        enable_vision_distill=False,
        enable_expert_distill=False,
        enable_logit_distill=True,
        student_action_dim=7,
        teacher_action_dim=20,
        action_align_mode="teacher_abs20_to_student_rel7",
        student_action_stats=stats,
        student_action_norm_mode="MEAN_STD",
    )

    student_action = torch.randn(2, 8, 7, dtype=torch.float32, requires_grad=True)
    teacher_action = torch.randn(2, 8, 20, dtype=torch.float32)
    teacher_state = torch.randn(2, 20, dtype=torch.float32)

    s_aligned = adapters.adapt_student_action(student_action)
    t_aligned = adapters.adapt_teacher_action(teacher_action, teacher_state=teacher_state)

    assert s_aligned.shape == (2, 8, 7)
    assert t_aligned.shape == (2, 8, 7)

    loss = F.mse_loss(s_aligned, t_aligned.detach())
    loss.backward()

    assert student_action.grad is not None
    assert torch.isfinite(student_action.grad).all()


def test_compat_mode_xvla_libero_20to7_still_available():
    adapters = DistillAdapters(
        enable_vision_distill=False,
        enable_expert_distill=False,
        enable_logit_distill=True,
        student_action_dim=7,
        teacher_action_dim=20,
        action_align_mode="xvla_libero_20to7",
    )

    student_action = torch.randn(2, 8, 7, dtype=torch.float32, requires_grad=True)
    teacher_action = torch.randn(2, 8, 20, dtype=torch.float32)

    s_aligned = adapters.adapt_student_action(student_action)
    t_aligned = adapters.adapt_teacher_action(teacher_action)

    assert s_aligned.shape == (2, 8, 7)
    assert t_aligned.shape == (2, 8, 7)

    temperature = 2.0
    loss = F.kl_div(
        F.log_softmax(s_aligned / temperature, dim=-1),
        F.softmax(t_aligned.detach() / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    loss.backward()

    assert student_action.grad is not None
    assert torch.isfinite(student_action.grad).all()
