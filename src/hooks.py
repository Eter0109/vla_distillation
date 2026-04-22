"""
前向钩子工具（src/hooks.py）

用于在模型前向传播时捕获中间层特征，
供蒸馏损失计算使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Any


class FeatureHook:
    """注册在 nn.Module 上的前向钩子，捕获模块输出。

    用法::

        hook = FeatureHook(some_module)
        model(input)
        feat = hook.output   # Tensor 或 Tuple[Tensor]
        hook.close()         # 移除钩子，避免内存泄漏
    """

    def __init__(
        self,
        module: nn.Module,
        detach: bool = True,
        clone: bool = True,
        capture: str = "output",
    ):
        self.output: Any = None
        self.detach = detach
        self.clone = clone
        self.capture = capture
        if capture == "output":
            self._handle = module.register_forward_hook(self._hook_fn)
        elif capture == "input":
            self._handle = module.register_forward_pre_hook(self._pre_hook_fn)
        else:
            raise ValueError(f"Unsupported hook capture mode: {capture}")

    def _store(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if self.detach:
            tensor = tensor.detach()
        if self.clone:
            tensor = tensor.clone()
        return tensor

    def _hook_fn(self, module: nn.Module, input: Any, output: Any) -> None:
        # 若输出为 tuple，取第一个元素（通常为隐层状态）
        if isinstance(output, tuple):
            self.output = self._store(output[0])
        elif isinstance(output, torch.Tensor):
            self.output = self._store(output)
        else:
            # HuggingFace ModelOutput (dataclass)：取 last_hidden_state 或第一个 Tensor 字段
            tensor = getattr(output, "last_hidden_state", None)
            if tensor is None:
                # 遍历字段取第一个 Tensor
                for v in output.values():
                    if isinstance(v, torch.Tensor):
                        tensor = v
                        break
            self.output = self._store(tensor)

    def _pre_hook_fn(self, module: nn.Module, input: Any) -> None:
        tensor = None
        if isinstance(input, tuple):
            tensor = input[0] if input else None
        elif isinstance(input, torch.Tensor):
            tensor = input
        self.output = self._store(tensor)

    def close(self) -> None:
        """移除钩子，释放资源。"""
        self._handle.remove()
        # 不清空 output，让调用方在 with 块外仍可读取最后一次捕获的特征

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MultiHookManager:
    """同时管理多个 FeatureHook 的上下文管理器。

    用法::

        with MultiHookManager({
            "teacher_vision": teacher_vision_module,
            "teacher_last": teacher_last_layer,
            "student_vision": student_vision_module,
            "student_last": student_last_layer,
        }) as hooks:
            # 前向传播
            teacher_out = teacher(batch)
            student_out = student(batch)
            # 读取中间特征
            tv = hooks["teacher_vision"].output
    """

    def __init__(self, name_to_module: dict[str, nn.Module | "HookSpec"]):
        self._hooks: dict[str, FeatureHook] = {
            name: FeatureHook(
                spec.module,
                detach=spec.detach,
                clone=spec.clone,
                capture=spec.capture,
            )
            for name, spec in {
                name: (value if isinstance(value, HookSpec) else HookSpec(module=value))
                for name, value in name_to_module.items()
            }.items()
        }

    def __getitem__(self, name: str) -> FeatureHook:
        return self._hooks[name]

    def close_all(self) -> None:
        for hook in self._hooks.values():
            hook.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close_all()


@dataclass(frozen=True)
class HookSpec:
    module: nn.Module
    detach: bool = True
    clone: bool = True
    capture: str = "output"
