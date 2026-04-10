"""
适配层模块（src/adapters.py）

将教师模型（XVLA）与学生模型（SmolVLA）的中间特征维度对齐。
视觉/Expert 特征使用可训练线性投影；动作蒸馏使用确定性 teacher→student
语义对齐，不引入额外可训练参数。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mat2quat(rmat: np.ndarray) -> np.ndarray:
    """将 3x3 旋转矩阵转为四元数（xyzw）。"""
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


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """将四元数（xyzw）转换为 axis-angle。"""
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)

    den = np.sqrt(max(np.float32(0.0), 1.0 - quat[3] * quat[3]))
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * np.float32(2.0) * np.float32(math.acos(float(quat[3])))) / den


try:
    from lerobot.policies.xvla.utils import rotate6d_to_axis_angle as _rotate6d_to_axis_angle_np
except Exception:
    def _rotate6d_to_axis_angle_np(r6d: np.ndarray) -> np.ndarray:
        """兼容 fallback：与 XVLA rotate6d_to_axis_angle 语义一致。"""
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


_SUPPORTED_ACTION_ALIGN_MODES = {
    "teacher_abs20_to_student_rel7",
    "xvla_libero_20to7",
}


def _rotate6d_to_axis_angle_torch(rotation_6d: torch.Tensor) -> torch.Tensor:
    """将 6D 旋转表示转换为 axis-angle（3D）。"""
    rotation_np = rotation_6d.detach().to(dtype=torch.float32).cpu().numpy()
    axis_angle_np = _rotate6d_to_axis_angle_np(rotation_np)
    return torch.from_numpy(axis_angle_np).to(device=rotation_6d.device, dtype=rotation_6d.dtype)


def _rotate6d_to_matrix_torch(rotation_6d: torch.Tensor) -> torch.Tensor:
    """将 6D 旋转表示转换为旋转矩阵（... , 3, 3）。"""
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected last dim=6 for rotation_6d, got {rotation_6d.shape[-1]}")
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _rotation_matrix_to_axis_angle_torch(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """将旋转矩阵（..., 3, 3）转换为 axis-angle。"""
    if rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(
            "rotation_matrix must have shape (..., 3, 3), "
            f"got {tuple(rotation_matrix.shape)}"
        )
    flat = rotation_matrix.reshape(-1, 3, 3)
    mats = flat.detach().to(dtype=torch.float32).cpu().numpy()
    axis_angle = np.stack([_quat2axisangle(_mat2quat(m)) for m in mats], axis=0)
    return torch.from_numpy(axis_angle).to(device=rotation_matrix.device, dtype=rotation_matrix.dtype)


def _axis_angle_to_rotation_matrix_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    """将 axis-angle（..., 3）转换为旋转矩阵（..., 3, 3）。"""
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected last dim=3 for axis_angle, got {axis_angle.shape[-1]}")

    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / torch.clamp(theta, min=1e-8)

    kx = axis[..., 0]
    ky = axis[..., 1]
    kz = axis[..., 2]
    zeros = torch.zeros_like(kx)

    k = torch.stack(
        [
            torch.stack([zeros, -kz, ky], dim=-1),
            torch.stack([kz, zeros, -kx], dim=-1),
            torch.stack([-ky, kx, zeros], dim=-1),
        ],
        dim=-2,
    )

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    sin_t = torch.sin(theta)[..., 0][..., None, None]
    cos_t = torch.cos(theta)[..., 0][..., None, None]
    return eye + sin_t * k + (1.0 - cos_t) * (k @ k)


def _validate_action_input(action: torch.Tensor, min_dim: int, *, name: str) -> None:
    if action.ndim < 2:
        raise ValueError(f"Expected {name} ndim >= 2, got shape={tuple(action.shape)}")
    if action.shape[-1] < min_dim:
        raise ValueError(
            f"Expected {name} last dim >= {min_dim}, got {action.shape[-1]}"
        )


def xvla_action20_to_abs7(
    action: torch.Tensor,
    *,
    binarize_gripper: bool = False,
) -> torch.Tensor:
    """将 XVLA 单臂动作从 20D 转成 abs-7D（pos + axis-angle + gripper）。"""
    _validate_action_input(action, min_dim=10, name="action")
    flat = action.reshape(-1, action.shape[-1])
    target_eef = flat[:, :3]
    rotation_6d = flat[:, 3:9]
    gripper = flat[:, 9:10]
    target_axis = _rotate6d_to_axis_angle_torch(rotation_6d)
    if binarize_gripper:
        gripper = torch.where(
            gripper > 0.5,
            torch.ones_like(gripper),
            -torch.ones_like(gripper),
        )
    action_7d = torch.cat([target_eef, target_axis, gripper], dim=-1)
    return action_7d.reshape(*action.shape[:-1], 7)


def xvla_libero_action20_to_action7(action: torch.Tensor) -> torch.Tensor:
    """与 XVLA LIBERO 评测一致的 20→7 动作转换（gripper 二值化）。"""
    return xvla_action20_to_abs7(action, binarize_gripper=True)


def _prepare_teacher_state_for_action(
    teacher_state: torch.Tensor,
    teacher_action: torch.Tensor,
) -> torch.Tensor:
    """将 teacher state 对齐到 teacher action 的 batch/time 结构。"""
    _validate_action_input(teacher_action, min_dim=10, name="teacher_action")
    _validate_action_input(teacher_state, min_dim=6, name="teacher_state")
    teacher_state = teacher_state.to(device=teacher_action.device, dtype=teacher_action.dtype)

    # 训练中常见形状是 (B, D) 或 (B, T_obs, D)。对于后者取当前时刻（最后一帧）。
    if teacher_state.ndim > 2:
        teacher_state = teacher_state[:, -1, :]

    if teacher_action.ndim == 2:
        if teacher_state.ndim != 2:
            raise ValueError(
                "For teacher_action shape (B, D), expected teacher_state shape (B, D), "
                f"got {tuple(teacher_state.shape)}"
            )
        return teacher_state

    if teacher_action.ndim == 3:
        if teacher_state.ndim != 2:
            raise ValueError(
                "For teacher_action shape (B, T, D), expected teacher_state shape (B, D) "
                f"or (B, T_obs, D). Got {tuple(teacher_state.shape)}"
            )
        if teacher_state.shape[0] != teacher_action.shape[0]:
            raise ValueError(
                "Batch size mismatch between teacher_action and teacher_state: "
                f"{teacher_action.shape[0]} vs {teacher_state.shape[0]}"
            )
        return teacher_state.unsqueeze(1).expand(-1, teacher_action.shape[1], -1)

    raise ValueError(
        f"Unsupported teacher_action ndim={teacher_action.ndim}. "
        "Only (B, D) and (B, T, D) are supported for action distillation."
    )


def _extract_current_pose_from_teacher_state(teacher_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """从 teacher state 提取当前位置与当前旋转矩阵。

    支持两类 state 语义：
      - state_dim >= 9: [pos(3), rot6d(6), ...]
      - 6 <= state_dim < 9: [pos(3), axis-angle(3), ...]（例如 LIBERO 常见 8D）
    """
    if teacher_state.shape[-1] < 6:
        raise ValueError(
            "teacher_state must contain at least position(3)+orientation(3), "
            f"got last dim={teacher_state.shape[-1]}"
        )

    current_pos = teacher_state[:, :3]
    if teacher_state.shape[-1] >= 9:
        current_rot = _rotate6d_to_matrix_torch(teacher_state[:, 3:9])
    else:
        current_rot = _axis_angle_to_rotation_matrix_torch(teacher_state[:, 3:6])
    return current_pos, current_rot


def xvla_teacher_action20_to_student_rel7(
    teacher_action: torch.Tensor,
    *,
    teacher_state: torch.Tensor,
) -> torch.Tensor:
    """将 XVLA teacher 20D 动作对齐到 SmolVLA 语义 7D 相对动作空间。

    规则（teacher_abs20_to_student_rel7）：
      1) 从 teacher action 取 target_pos/target_rot6d/gripper（前10维）
      2) 从 teacher state 取 current pose：
         - state_dim>=9: pos+rot6d
         - 6<=state_dim<9: pos+axis-angle
      3) delta_pos = target_pos - current_pos
      4) delta_rot = axis_angle(R_target * R_current^{-1})
      5) gripper: [0, 1] -> [-1, 1] 并 clamp
    """
    aligned_state = _prepare_teacher_state_for_action(teacher_state, teacher_action)

    action_flat = teacher_action.reshape(-1, teacher_action.shape[-1])
    state_flat = aligned_state.reshape(-1, aligned_state.shape[-1])

    target_pos = action_flat[:, :3]
    target_rot6d = action_flat[:, 3:9]
    target_gripper = action_flat[:, 9:10]

    current_pos, current_rot = _extract_current_pose_from_teacher_state(state_flat)

    delta_pos = target_pos - current_pos
    target_rot = _rotate6d_to_matrix_torch(target_rot6d)
    delta_rot_mat = torch.matmul(target_rot, current_rot.transpose(-1, -2))
    delta_rot = _rotation_matrix_to_axis_angle_torch(delta_rot_mat)

    mapped_gripper = torch.clamp(target_gripper * 2.0 - 1.0, min=-1.0, max=1.0)
    rel7 = torch.cat([delta_pos, delta_rot, mapped_gripper], dim=-1)
    return rel7.reshape(*teacher_action.shape[:-1], 7)


def _canonical_norm_mode(norm_mode: Any) -> str:
    mode = getattr(norm_mode, "value", norm_mode)
    mode = str(mode).strip().lower()
    if mode in {"identity", "none"}:
        return "identity"
    if mode in {"mean_std", "meanstd"}:
        return "mean_std"
    if mode in {"min_max", "minmax"}:
        return "min_max"
    if mode in {"quantiles", "quantile"}:
        return "quantiles"
    if mode in {"quantile10", "quantile_10"}:
        return "quantile10"
    return mode


def _get_stat_tensor(
    stats: dict[str, Any],
    key: str,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if key not in stats:
        raise ValueError(f"Missing required action stat '{key}'")
    tensor = torch.as_tensor(stats[key], device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        tensor = tensor.repeat(reference.shape[-1])
    if tensor.shape[-1] < reference.shape[-1]:
        raise ValueError(
            f"Stat '{key}' dim={tensor.shape[-1]} is smaller than action dim={reference.shape[-1]}"
        )
    return tensor[..., : reference.shape[-1]]


def normalize_action_to_student_training_scale(
    action_7d: torch.Tensor,
    *,
    stats: dict[str, Any] | None,
    norm_mode: Any,
    eps: float = 1e-8,
) -> torch.Tensor:
    """将 action 映射到 student 训练尺度（按 student 的 action normalization 规则）。"""
    mode = _canonical_norm_mode(norm_mode)
    if mode == "identity":
        return action_7d
    if stats is None:
        raise ValueError(
            f"Action normalization mode '{norm_mode}' requires action stats, but stats is None."
        )

    if mode == "mean_std":
        mean = _get_stat_tensor(stats, "mean", reference=action_7d)
        std = _get_stat_tensor(stats, "std", reference=action_7d)
        return (action_7d - mean) / (std + eps)

    if mode == "min_max":
        min_val = _get_stat_tensor(stats, "min", reference=action_7d)
        max_val = _get_stat_tensor(stats, "max", reference=action_7d)
        denom = torch.where(
            (max_val - min_val) == 0,
            torch.tensor(eps, device=action_7d.device, dtype=action_7d.dtype),
            max_val - min_val,
        )
        return 2.0 * (action_7d - min_val) / denom - 1.0

    if mode == "quantiles":
        q01 = _get_stat_tensor(stats, "q01", reference=action_7d)
        q99 = _get_stat_tensor(stats, "q99", reference=action_7d)
        denom = torch.where(
            (q99 - q01) == 0,
            torch.tensor(eps, device=action_7d.device, dtype=action_7d.dtype),
            q99 - q01,
        )
        return 2.0 * (action_7d - q01) / denom - 1.0

    if mode == "quantile10":
        q10 = _get_stat_tensor(stats, "q10", reference=action_7d)
        q90 = _get_stat_tensor(stats, "q90", reference=action_7d)
        denom = torch.where(
            (q90 - q10) == 0,
            torch.tensor(eps, device=action_7d.device, dtype=action_7d.dtype),
            q90 - q10,
        )
        return 2.0 * (action_7d - q10) / denom - 1.0

    raise ValueError(
        f"Unsupported action normalization mode: {norm_mode}. "
        "Expected one of IDENTITY/MEAN_STD/MIN_MAX/QUANTILES/QUANTILE10."
    )


class FeatureProjector(nn.Module):
    """通用线性投影适配层：将 in_dim 映射到 out_dim。

    当 in_dim == out_dim 时退化为恒等映射（不引入额外参数）。
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if in_dim != out_dim:
            # 带 LayerNorm 的线性投影，提升蒸馏稳定性
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=True),
                nn.LayerNorm(out_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VisionFeatureAdapter(FeatureProjector):
    """视觉特征适配层。"""

    def __init__(self, student_dim: int = 960, teacher_dim: int = 1024):
        super().__init__(in_dim=student_dim, out_dim=teacher_dim)


class ActionExpertFeatureAdapter(FeatureProjector):
    """Action Expert 特征适配层。"""

    def __init__(self, student_dim: int = 480, teacher_dim: int = 1024):
        super().__init__(in_dim=student_dim, out_dim=teacher_dim)


class ActionAdapter(nn.Module):
    """动作维度适配层（确定性对齐，不可训练）。

    - student 侧：保持 7D 蒸馏空间（默认 identity）
    - teacher 侧：
      - 默认 `teacher_abs20_to_student_rel7`
      - 兼容 `xvla_libero_20to7`
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 7,
        *,
        role: str,
        align_mode: str = "teacher_abs20_to_student_rel7",
        student_action_stats: dict[str, Any] | None = None,
        student_action_norm_mode: Any = "IDENTITY",
    ):
        super().__init__()
        if role not in {"student", "teacher"}:
            raise ValueError(f"Unsupported role: {role}. Expected 'student' or 'teacher'.")
        if align_mode not in _SUPPORTED_ACTION_ALIGN_MODES:
            raise ValueError(
                f"Unsupported action align mode: {align_mode}. "
                f"Supported modes: {sorted(_SUPPORTED_ACTION_ALIGN_MODES)}"
            )
        if out_dim != 7:
            raise ValueError(f"Action distill space must be 7D, got out_dim={out_dim}")

        self.role = role
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.align_mode = align_mode
        self.student_action_stats = student_action_stats
        self.student_action_norm_mode = student_action_norm_mode

    def forward(
        self,
        action: torch.Tensor,
        *,
        teacher_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action.shape[-1] != self.in_dim:
            raise ValueError(
                f"ActionAdapter expected last dim={self.in_dim}, got {action.shape[-1]}"
            )

        if self.role == "student":
            return action[..., : self.out_dim]

        if self.align_mode == "xvla_libero_20to7":
            return xvla_libero_action20_to_action7(action)

        if self.align_mode == "teacher_abs20_to_student_rel7":
            if teacher_state is None:
                raise ValueError(
                    "teacher_state is required for align_mode='teacher_abs20_to_student_rel7'."
                )
            aligned = xvla_teacher_action20_to_student_rel7(action, teacher_state=teacher_state)
            return normalize_action_to_student_training_scale(
                aligned,
                stats=self.student_action_stats,
                norm_mode=self.student_action_norm_mode,
            )

        raise RuntimeError(f"Unexpected align mode: {self.align_mode}")


class DistillAdapters(nn.Module):
    """蒸馏适配层集合，统一管理所有对齐层。"""

    def __init__(
        self,
        student_vision_dim: int = 960,
        teacher_vision_dim: int = 1024,
        student_expert_dim: int = 480,
        teacher_expert_dim: int = 1024,
        student_action_dim: int = 7,
        teacher_action_dim: int = 20,
        action_align_mode: str = "teacher_abs20_to_student_rel7",
        student_action_stats: dict[str, Any] | None = None,
        student_action_norm_mode: Any = "IDENTITY",
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
                role="student",
                align_mode=action_align_mode,
                student_action_stats=student_action_stats,
                student_action_norm_mode=student_action_norm_mode,
            )
            self.teacher_action_adapter = ActionAdapter(
                in_dim=teacher_action_dim,
                out_dim=7,
                role="teacher",
                align_mode=action_align_mode,
                student_action_stats=student_action_stats,
                student_action_norm_mode=student_action_norm_mode,
            )
            self.aligned_action_dim = 7

    def adapt_vision(self, student_feat: torch.Tensor) -> torch.Tensor:
        assert self.enable_vision_distill, "vision distill未启用"
        return self.vision_adapter(student_feat)

    def adapt_expert(self, student_feat: torch.Tensor) -> torch.Tensor:
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

    def adapt_teacher_action(
        self,
        teacher_action: torch.Tensor,
        *,
        teacher_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将教师动作对齐到蒸馏动作空间（7D）。"""
        assert self.enable_logit_distill, "logit distill未启用"
        return self.teacher_action_adapter(teacher_action, teacher_state=teacher_state)


def align_seq_len(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """对齐序列长度（第1维），截断或零填充学生特征使其与教师长度一致。"""
    l_s = student_feat.shape[1]
    l_t = teacher_feat.shape[1]
    if l_s == l_t:
        return student_feat
    if l_s > l_t:
        return student_feat[:, :l_t, :]
    pad = student_feat.new_zeros(student_feat.shape[0], l_t - l_s, student_feat.shape[2])
    return torch.cat([student_feat, pad], dim=1)
