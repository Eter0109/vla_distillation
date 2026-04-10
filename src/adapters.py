"""
适配层模块（src/adapters.py）

将教师模型（XVLA）与学生模型（SmolVLA）的中间特征维度对齐。
视觉/Expert 特征使用可训练线性投影；动作蒸馏使用与 LIBERO 评测一致
的确定性 20→7 对齐，不引入额外可训练参数。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

try:
    from lerobot.policies.xvla.utils import rotate6d_to_axis_angle as _rotate6d_to_axis_angle_np
except Exception:
    def _mat2quat(rmat):
        mat = np.asarray(rmat).astype(np.float32)[:3, :3]
        m00 = mat[0, 0]
        m01 = mat[0, 1]
        m02 = mat[0, 2]
        m10 = mat[1, 0]
        m11 = mat[1, 1]
        m12 = mat[1, 2]
        m20 = mat[2, 0]
        m21 = mat[2, 1]
        m22 = mat[2, 2]
        k = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        k /= 3.0
        w, v = np.linalg.eigh(k)
        inds = np.array([3, 0, 1, 2])
        q1 = v[inds, np.argmax(w)]
        if q1[0] < 0.0:
            np.negative(q1, q1)
        inds = np.array([1, 2, 3, 0])
        return q1[inds]

    def _quat2axisangle(quat):
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def _rotate6d_to_axis_angle_np(r6d):
        flag = 0
        if len(r6d.shape) == 1:
            r6d = r6d[None, ...]
            flag = 1

        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]

        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + 1e-6)
        b3 = np.cross(b1, b2, axis=-1)

        rotation_matrix = np.stack([b1, b2, b3], axis=-1)
        axis_angle_list = []
        for i in range(rotation_matrix.shape[0]):
            quat = _mat2quat(rotation_matrix[i])
            axis_angle_list.append(_quat2axisangle(quat))

        axis_angle_array = np.stack(axis_angle_list, axis=0)
        if flag == 1:
            axis_angle_array = axis_angle_array[0]
        return axis_angle_array

_SUPPORTED_ACTION_ALIGN_MODES = {"xvla_libero_20to7"}


def _rotate6d_to_axis_angle_torch(rotation_6d: torch.Tensor) -> torch.Tensor:
    """将 6D 旋转表示转换为 axis-angle（3D）。

    对齐 `lerobot.policies.xvla.utils.rotate6d_to_axis_angle` 的实现路径，
    保持与 XVLA LIBERO 评测一致的数值语义。
    """
    rotation_np = rotation_6d.detach().to(dtype=torch.float32).cpu().numpy()
    axis_angle_np = _rotate6d_to_axis_angle_np(rotation_np)
    return torch.from_numpy(axis_angle_np).to(device=rotation_6d.device, dtype=rotation_6d.dtype)


def xvla_libero_action20_to_action7(action: torch.Tensor) -> torch.Tensor:
    """将 XVLA 动作从 20D（含 6D 旋转）对齐为 LIBERO 评测使用的 7D。

    支持输入形状:
      - (B, D)
      - (B, T, D)
      - 任意 (..., D)（最后一维为动作维）

    规则与 `XVLARotation6DToAxisAngleProcessorStep` 一致:
      1) eef = action[..., :3]
      2) rot6d = action[..., 3:9] -> axis_angle(3)
      3) gripper = action[..., 9:10]，阈值化为 {-1, 1}
      4) 拼接为 7 维动作
    """
    if action.ndim < 2:
        raise ValueError(f"Expected action ndim >= 2, got shape={tuple(action.shape)}")
    if action.shape[-1] < 10:
        raise ValueError(
            f"Expected action last dim >= 10 for XVLA LIBERO conversion, got {action.shape[-1]}"
        )

    flat = action.reshape(-1, action.shape[-1])
    target_eef = flat[:, :3]
    rotation_6d = flat[:, 3:9]
    target_act = flat[:, 9:10]

    target_axis = _rotate6d_to_axis_angle_torch(rotation_6d)
    target_act = torch.where(
        target_act > 0.5,
        torch.ones_like(target_act),
        -torch.ones_like(target_act),
    )
    action_7d = torch.cat([target_eef, target_axis, target_act], dim=-1)
    return action_7d.reshape(*action.shape[:-1], 7)


class FeatureProjector(nn.Module):
    """通用线性投影适配层：将 in_dim 映射到 out_dim。
    当 in_dim == out_dim 时退化为恒等映射（不引入额外参数）。
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if in_dim != out_dim:
            # 带LayerNorm的线性投影，提升蒸馏稳定性
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=True),
                nn.LayerNorm(out_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim) 任意形状张量，最后一维为特征维
        Returns:
            (..., out_dim)
        """
        return self.proj(x)


class VisionFeatureAdapter(FeatureProjector):
    """视觉特征适配层。
    将学生模型视觉 connector 输出对齐到教师 `forward_vlm()` 返回的 `vlm_features` 维度。

    维度说明：
        学生（SmolVLA connector 输出）：960维
        教师（XVLA forward_vlm 返回 vlm_features）：1024维
    """

    def __init__(self, student_dim: int = 960, teacher_dim: int = 1024):
        super().__init__(in_dim=student_dim, out_dim=teacher_dim)


class ActionExpertFeatureAdapter(FeatureProjector):
    """Action Expert 特征适配层。
    将学生模型 lm_expert 最终输出维度对齐到教师 transformer block hidden 维度。

    维度说明：
        学生（SmolVLA _last_expert_output）：480维
        教师（XVLA transformer hidden）：1024维
    """

    def __init__(self, student_dim: int = 480, teacher_dim: int = 1024):
        super().__init__(in_dim=student_dim, out_dim=teacher_dim)


class ActionAdapter(nn.Module):
    """动作维度适配层（确定性对齐，不可训练）。

    不再执行 7→20 线性映射。当前仅支持与 XVLA LIBERO 评测一致的
    `xvla_libero_20to7` 规则，将动作最终对齐到 7D 蒸馏空间。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 7,
        align_mode: str = "xvla_libero_20to7",
    ):
        super().__init__()
        if align_mode not in _SUPPORTED_ACTION_ALIGN_MODES:
            raise ValueError(
                f"Unsupported action align mode: {align_mode}. "
                f"Supported modes: {sorted(_SUPPORTED_ACTION_ALIGN_MODES)}"
            )
        if out_dim != 7:
            raise ValueError(f"{align_mode} currently requires out_dim=7, got {out_dim}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.align_mode = align_mode

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.in_dim:
            raise ValueError(
                f"ActionAdapter expected last dim={self.in_dim}, got {action.shape[-1]}"
            )
        if self.in_dim == self.out_dim:
            return action
        if self.align_mode == "xvla_libero_20to7":
            return xvla_libero_action20_to_action7(action)
        raise RuntimeError(f"Unexpected align mode: {self.align_mode}")


class DistillAdapters(nn.Module):
    """蒸馏适配层集合，统一管理所有对齐层。

    可根据配置文件的开关（enable_vision_distill / enable_expert_distill /
    enable_logit_distill）决定哪些适配器需要参与梯度计算。
    """

    def __init__(
        self,
        student_vision_dim: int = 960,
        teacher_vision_dim: int = 1024,
        student_expert_dim: int = 480,
        teacher_expert_dim: int = 1024,
        student_action_dim: int = 7,
        teacher_action_dim: int = 20,
        action_align_mode: str = "xvla_libero_20to7",
        enable_vision_distill: bool = True,
        enable_expert_distill: bool = False,
        enable_logit_distill: bool = True,
    ):
        super().__init__()
        self.enable_vision_distill = enable_vision_distill
        self.enable_expert_distill = enable_expert_distill
        self.enable_logit_distill = enable_logit_distill

        if enable_vision_distill:
            self.vision_adapter = VisionFeatureAdapter(student_vision_dim, teacher_vision_dim)
        if enable_expert_distill:
            self.expert_adapter = ActionExpertFeatureAdapter(student_expert_dim, teacher_expert_dim)

        if enable_logit_distill:
            self.student_action_adapter = ActionAdapter(
                in_dim=student_action_dim,
                out_dim=7,
                align_mode=action_align_mode,
            )
            self.teacher_action_adapter = ActionAdapter(
                in_dim=teacher_action_dim,
                out_dim=7,
                align_mode=action_align_mode,
            )
            self.aligned_action_dim = 7

    def adapt_vision(self, student_feat: torch.Tensor) -> torch.Tensor:
        """将学生视觉特征投影到教师维度。"""
        assert self.enable_vision_distill, "vision distill未启用"
        return self.vision_adapter(student_feat)

    def adapt_expert(self, student_feat: torch.Tensor) -> torch.Tensor:
        """将学生 lm_expert 特征投影到教师维度。"""
        assert self.enable_expert_distill, "expert distill未启用"
        return self.expert_adapter(student_feat)

    def adapt_action(self, student_action: torch.Tensor) -> torch.Tensor:
        """向后兼容别名：等价于 adapt_student_action。"""
        assert self.enable_logit_distill, "logit distill未启用"
        return self.student_action_adapter(student_action)

    def adapt_student_action(self, student_action: torch.Tensor) -> torch.Tensor:
        """将学生动作对齐到蒸馏动作空间（7D）。"""
        assert self.enable_logit_distill, "logit distill未启用"
        return self.student_action_adapter(student_action)

    def adapt_teacher_action(self, teacher_action: torch.Tensor) -> torch.Tensor:
        """将教师动作对齐到蒸馏动作空间（7D）。"""
        assert self.enable_logit_distill, "logit distill未启用"
        return self.teacher_action_adapter(teacher_action)


def align_seq_len(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """对齐序列长度（第1维），截断或零填充学生特征使其与教师长度一致。

    Args:
        student_feat: (B, L_s, D)
        teacher_feat: (B, L_t, D)
    Returns:
        student_feat: (B, L_t, D)
    """
    L_s = student_feat.shape[1]
    L_t = teacher_feat.shape[1]
    if L_s == L_t:
        return student_feat
    if L_s > L_t:
        return student_feat[:, :L_t, :]
    # L_s < L_t：零填充
    pad = student_feat.new_zeros(student_feat.shape[0], L_t - L_s, student_feat.shape[2])
    return torch.cat([student_feat, pad], dim=1)
