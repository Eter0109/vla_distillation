from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adapters import DistillAdapters, xvla_libero_action20_to_action7


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


def test_alignment_matches_lerobot_xvla_libero_postprocessor():
    step_cls, transition_key = _import_lerobot_xvla_rotation_step()
    step = step_cls()

    torch.manual_seed(42)
    teacher_action = torch.randn(2, 6, 20, dtype=torch.float32)

    expected = step({transition_key.ACTION: teacher_action.reshape(-1, 20)})[transition_key.ACTION]
    expected = expected.reshape(2, 6, 7)
    actual = xvla_libero_action20_to_action7(teacher_action)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_alignment_supports_bt_and_bd_inputs():
    torch.manual_seed(0)
    teacher_bt = torch.randn(3, 4, 20, dtype=torch.float32)
    teacher_bd = torch.randn(5, 20, dtype=torch.float32)

    out_bt = xvla_libero_action20_to_action7(teacher_bt)
    out_bd = xvla_libero_action20_to_action7(teacher_bd)

    assert out_bt.shape == (3, 4, 7)
    assert out_bd.shape == (5, 7)
    assert set(torch.unique(out_bt[..., -1]).tolist()).issubset({-1.0, 1.0})
    assert set(torch.unique(out_bd[..., -1]).tolist()).issubset({-1.0, 1.0})


def test_distill_adapter_outputs_aligned_7d_for_kl():
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
