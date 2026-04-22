from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hooks import FeatureHook, HookSpec, MultiHookManager


def _import_soft_prompted_transformer():
    lerobot_src = ROOT.parent / "lerobot" / "src"
    if not lerobot_src.exists():
        pytest.skip(f"lerobot source not found at expected path: {lerobot_src}")

    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))

    try:
        from lerobot.policies.xvla.soft_transformer import SoftPromptedTransformer
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Unable to import XVLA SoftPromptedTransformer: {exc}")

    return SoftPromptedTransformer


def test_feature_hook_capture_input_preserves_gradients():
    layer = nn.Linear(4, 3)
    x = torch.randn(2, 5, 4, requires_grad=True)

    with MultiHookManager({
        "pre": HookSpec(module=layer, detach=False, clone=False, capture="input")
    }) as hooks:
        y = layer(x)
        captured = hooks["pre"].output

    assert captured is x
    assert captured.requires_grad
    y.sum().backward()
    assert x.grad is not None


def test_feature_hook_keeps_last_forward_value():
    layer = nn.Linear(4, 3)
    hook = FeatureHook(layer, detach=False, clone=False, capture="input")

    x1 = torch.randn(1, 4, requires_grad=True)
    x2 = torch.randn(1, 4, requires_grad=True)
    layer(x1)
    layer(x2)
    hook.close()

    assert hook.output is x2


def test_xvla_norm_hook_returns_action_tokens_only():
    SoftPromptedTransformer = _import_soft_prompted_transformer()
    model = SoftPromptedTransformer(
        hidden_size=16,
        multi_modal_input_size=8,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_domains=3,
        dim_action=7,
        dim_propio=5,
        dim_time=4,
        len_soft_prompts=2,
        max_len_seq=32,
        use_hetero_proj=False,
    )

    batch_size = 2
    num_actions = 3
    with MultiHookManager({"teacher_expert": HookSpec(module=model.norm)}) as hooks:
        output = model(
            domain_id=torch.zeros(batch_size, dtype=torch.long),
            vlm_features=torch.randn(batch_size, 4, 8),
            aux_visual_inputs=torch.randn(batch_size, 2, 8),
            action_with_noise=torch.randn(batch_size, num_actions, 7),
            proprio=torch.randn(batch_size, 5),
            t=torch.ones(batch_size),
        )
        captured = hooks["teacher_expert"].output

    assert output.shape == (batch_size, num_actions, 7)
    assert captured is not None
    assert captured.shape == (batch_size, num_actions, 16)
