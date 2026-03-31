"""
前向钩子工具（src/hooks.py）

用于在模型前向传播时捕获中间层特征，
供蒸馏损失计算使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any


class FeatureHook:
    """注册在 nn.Module 上的前向钩子，捕获模块输出（detach + clone）。

    用法::

        hook = FeatureHook(some_module)
        model(input)
        feat = hook.output   # Tensor 或 Tuple[Tensor]
        hook.close()         # 移除钩子，避免内存泄漏
    """

    def __init__(self, module: nn.Module):
        self.output: Any = None
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, input: Any, output: Any) -> None:
        # 若输出为 tuple，取第一个元素（通常为隐层状态）
        if isinstance(output, tuple):
            self.output = output[0].detach().clone()
        else:
            self.output = output.detach().clone()

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

    def __init__(self, name_to_module: dict[str, nn.Module]):
        self._hooks: dict[str, FeatureHook] = {
            name: FeatureHook(module) for name, module in name_to_module.items()
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
