"""
学生模型 Policy Wrapper（src/student_policy_wrapper.py）

将训练好的 SmolVLA 学生模型封装为 lerobot-eval 可识别的 Policy 接口，
可直接传递给 lerobot.scripts.lerobot_eval 进行 LIBERO 任务评测。

使用方法：
    from student_policy_wrapper import load_student_policy
    policy = load_student_policy("/path/to/checkpoint", device="cuda:0")
    # 然后传给 lerobot_eval 的 policy 参数
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

LEROBOT_ROOT = "/hqlab/workspace/zhaozy/lerobot/src"
if LEROBOT_ROOT not in sys.path:
    sys.path.insert(0, LEROBOT_ROOT)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class StudentPolicyWrapper(SmolVLAPolicy):
    """SmolVLA 学生模型的 lerobot-eval 兼容包装器。

    继承自 SmolVLAPolicy，行为完全与父类相同。
    此包装器主要用于从检查点目录加载蒸馏后的权重，
    并确保与 lerobot 评测接口兼容。
    """

    name = "smolvla"  # lerobot-eval 通过 name 识别 policy 类型

    @classmethod
    def from_distilled_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
    ) -> "StudentPolicyWrapper":
        """从蒸馏训练检查点加载模型。

        Args:
            checkpoint_path: 检查点目录（包含 config.json + model.safetensors）
            device: 推理设备
        Returns:
            加载好权重并移动到 device 的 StudentPolicyWrapper 实例
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点目录不存在: {checkpoint_path}")
        if not (checkpoint_path / "config.json").exists():
            raise FileNotFoundError(f"检查点目录缺少 config.json: {checkpoint_path}")
        if not (checkpoint_path / "model.safetensors").exists():
            raise FileNotFoundError(f"检查点目录缺少 model.safetensors: {checkpoint_path}")

        # 使用 PreTrainedPolicy.from_pretrained 加载（lerobot 标准接口）
        policy = cls.from_pretrained(str(checkpoint_path))
        policy = policy.to(device)
        policy.eval()
        return policy


def load_student_policy(
    checkpoint_path: str | Path,
    device: str = "cuda:0",
) -> StudentPolicyWrapper:
    """便捷函数：加载蒸馏后的学生模型。

    Args:
        checkpoint_path: 检查点目录路径（step_XXXXXXX 或 best）
        device: 推理设备，如 "cuda:0"
    Returns:
        StudentPolicyWrapper 实例，可直接用于 lerobot-eval
    """
    return StudentPolicyWrapper.from_distilled_checkpoint(checkpoint_path, device)
