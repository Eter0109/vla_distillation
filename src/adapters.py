"""
适配层模块（src/adapters.py）

将教师模型（XVLA）与学生模型（SmolVLA）的中间特征维度对齐。
所有适配层均为可训练的线性投影，放在学生模型训练侧。
"""

import torch
import torch.nn as nn


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
    将学生模型 VLM 视觉编码器的输出维度对齐到教师模型 VLM 的维度。

    维度说明：
        学生（SmolVLM2-500M embed_image输出）：576维（text_config.hidden_size）
        教师（Florence2 forward_vlm vlm_features）：1024维（projection_dim/hidden_size）
    """

    def __init__(self, student_dim: int = 576, teacher_dim: int = 1024):
        super().__init__(in_dim=student_dim, out_dim=teacher_dim)


class ActionExpertFeatureAdapter(FeatureProjector):
    """Action Expert 特征适配层。
    将学生模型 lm_expert 最后一层输出维度对齐到教师 transformer 隐层维度。

    维度说明：
        学生（SmolVLA expert_hidden_size = 576 * 0.5 = 288）
        教师（XVLA SoftPromptedTransformer hidden_size = 1024）
    """

    def __init__(self, student_dim: int = 288, teacher_dim: int = 1024):
        super().__init__(in_dim=student_dim, out_dim=teacher_dim)


class ActionAdapter(FeatureProjector):
    """动作维度适配层。
    将学生模型动作输出维度对齐到教师模型动作维度，用于 logit 蒸馏（KL 散度）。

    维度说明：
        学生（SmolVLA action_feature.shape[0] = 7，max_action_dim = 32 padding后）
        教师（XVLA action_feature.shape[0] = 20，dim_action）

    注意：
        SmolVLA forward 返回的是流匹配损失张量，不是显式 logit。
        蒸馏时对齐的是预测动作 chunk（B, T, action_dim）而非概率分布。
        KL散度将在 chunk 最后一维 softmax 后计算，因此需要维度匹配。
    """

    def __init__(self, student_dim: int = 7, teacher_dim: int = 20):
        super().__init__(in_dim=student_dim, out_dim=teacher_dim)


class DistillAdapters(nn.Module):
    """蒸馏适配层集合，统一管理所有对齐层。

    可根据配置文件的开关（enable_feature_distill / enable_logit_distill）
    决定哪些适配器需要参与梯度计算。
    """

    def __init__(
        self,
        student_vision_dim: int = 576,
        teacher_vision_dim: int = 1024,
        student_expert_dim: int = 288,
        teacher_expert_dim: int = 1024,
        student_action_dim: int = 7,
        teacher_action_dim: int = 20,
        enable_feature_distill: bool = True,
        enable_logit_distill: bool = True,
    ):
        super().__init__()
        self.enable_feature_distill = enable_feature_distill
        self.enable_logit_distill = enable_logit_distill

        if enable_feature_distill:
            self.vision_adapter = VisionFeatureAdapter(student_vision_dim, teacher_vision_dim)
            self.expert_adapter = ActionExpertFeatureAdapter(student_expert_dim, teacher_expert_dim)

        if enable_logit_distill:
            self.action_adapter = ActionAdapter(student_action_dim, teacher_action_dim)

    def adapt_vision(self, student_feat: torch.Tensor) -> torch.Tensor:
        """将学生视觉特征投影到教师维度。"""
        assert self.enable_feature_distill, "feature distill未启用"
        return self.vision_adapter(student_feat)

    def adapt_expert(self, student_feat: torch.Tensor) -> torch.Tensor:
        """将学生 lm_expert 特征投影到教师维度。"""
        assert self.enable_feature_distill, "feature distill未启用"
        return self.expert_adapter(student_feat)

    def adapt_action(self, student_action: torch.Tensor) -> torch.Tensor:
        """将学生动作预测投影到教师动作维度用于 KL 散度计算。"""
        assert self.enable_logit_distill, "logit distill未启用"
        return self.action_adapter(student_action)


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
