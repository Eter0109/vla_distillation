from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # 修正导入路径
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from typing import Tuple, Dict

class DistillSmolVLA(SmolVLAPolicy):
    """带蒸馏和LoRA的SmolVLA模型，适配LeRobot训练框架"""
    def __init__(self, config, teacher=None):
        super().__init__(config)
        self.teacher = teacher  # 教师模型（XVLA）
        self.teacher.eval()  # 固定教师模型

        # 1. 维度对齐层：适配SmolVLA和XVLA的特征维度差异
        # SmolVLM2-500M的视觉特征维度是512，XVLA是768（可根据实际调整）
        self.vis_proj = nn.Linear(512, 768)  # 视觉特征投影
        self.attn_proj = nn.Linear(
            self.model.lm_expert.config.hidden_size,  # SmolVLA专家层维度
            768  # XVLA注意力维度
        )

        # 2. 应用LoRA（仅训练LoRA参数，冻结主干）
        self.apply_lora()

        # 3. 蒸馏损失权重
        self.act_loss_coeff = 1.0    # 动作损失权重
        self.vis_loss_coeff = 0.3    # 视觉特征损失权重
        self.attn_loss_coeff = 0.2   # 注意力损失权重

    def apply_lora(self):
        """给SmolVLA的关键层添加LoRA，适配SmolVLMWithExpertModel结构"""
        # 目标层：匹配smolvlm_with_expert.py中的attention层命名
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=[
                "q_proj", "v_proj",  # 注意力层
                "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"  # FFN层（可选）
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 仅对lm_expert和VLM的顶层应用LoRA（符合SmolVLA训练策略）
        self.model.lm_expert = get_peft_model(self.model.lm_expert, lora_config)
        # 若不冻结VLM，可对VLM顶层应用LoRA
        # self.model.vlm.model.text_model.layers[-1] = get_peft_model(...)

        # 打印可训练参数
        self.model.lm_expert.print_trainable_parameters()

    def extract_features(self, batch) -> Dict[str, torch.Tensor]:
        """提取学生模型的中间特征（用于蒸馏）"""
        # 1. 视觉特征（来自SmolVLM的vision_model）
        image_input = batch["observation.image"]  # (B, 3, H, W)
        vis_feat = self.model.vlm.model.vision_model(image_input).last_hidden_state  # (B, T, 512)
        vis_feat_proj = self.vis_proj(vis_feat)  # 对齐到768维

        # 2. 注意力特征（来自lm_expert的最后一层）
        # 先执行VLM前向获取文本/视觉融合特征
        vlm_embeds = self.model.vlm.model.connector(vis_feat)  # 视觉特征投影
        lang_tokens = batch.get("observation.language_tokens", None)
        if lang_tokens is not None:
            lang_embeds = self.model.vlm.model.text_model.get_input_embeddings()(lang_tokens)
            input_embeds = torch.cat([lang_embeds, vlm_embeds], dim=1)
        else:
            input_embeds = vlm_embeds

        # 执行lm_expert前向，提取最后一层注意力输出
        attn_feat = None
        for layer_idx, layer in enumerate(self.model.lm_expert.layers):
            input_embeds = layer(input_embeds)[0]
            if layer_idx == len(self.model.lm_expert.layers) - 1:
                attn_feat = self.attn_proj(input_embeds)  # 对齐到768维

        return {
            "vis_feat": vis_feat_proj,
            "attn_feat": attn_feat,
        }

    def forward_teacher(self, batch) -> Dict[str, torch.Tensor]:
        """执行教师模型（XVLA）前向，提取特征（无梯度）"""
        with torch.no_grad():
            # XVLA forward（参考modeling_xvla.py的输入格式）
            xvla_output = self.teacher(
                input_ids=batch.get("input_ids", None),
                image_input=batch["observation.image"],
                image_mask=batch.get("image_mask", None),
                domain_id=batch.get("domain_id", torch.zeros(len(batch), dtype=torch.long).to(batch["observation.image"].device)),
                proprio=batch.get("observation.state", None),
                action=batch.get("action", None),
            )
            # 提取XVLA的视觉特征和注意力特征
            teacher_vis_feat = self.teacher.forward_vlm(
                batch.get("input_ids", None),
                batch["observation.image"],
                batch.get("image_mask", None)
            )["vlm_features"]  # (B, T, 768)
            teacher_attn_feat = self.teacher.transformer.blocks[-1](xvla_output["x"])[0]  # 最后一层transformer输出

        return {
            "action": xvla_output["pred_action"],
            "vis_feat": teacher_vis_feat,
            "attn_feat": teacher_attn_feat,
        }

    def forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        重载forward，返回：(总损失, 损失分解字典)
        适配LeRobot的训练逻辑（参考tdmpc/modeling_tdmpc.py的输出格式）
        """
        # 1. 学生模型基础前向（动作预测）
        student_action, student_aux = super().forward(batch)

        # 2. 提取学生/教师特征
        student_feat = self.extract_features(batch)
        teacher_feat = self.forward_teacher(batch)

        # 3. 计算蒸馏损失
        # 3.1 动作损失（MSE）
        loss_act = F.mse_loss(student_action, teacher_feat["action"])

        # 3.2 视觉特征损失（MSE，对齐维度）
        loss_vis = F.mse_loss(
            student_feat["vis_feat"],
            teacher_feat["vis_feat"][:, :student_feat["vis_feat"].shape[1], :]  # 序列长度对齐
        )

        # 3.3 注意力特征损失（KL散度）
        loss_attn = F.kl_div(
            student_feat["attn_feat"].log_softmax(dim=-1),
            teacher_feat["attn_feat"][:, :student_feat["attn_feat"].shape[1], :].softmax(dim=-1),
            reduction="batchmean"
        )

        # 4. 总损失
        total_loss = (
            self.act_loss_coeff * loss_act
            + self.vis_loss_coeff * loss_vis
            + self.attn_loss_coeff * loss_attn
        )

        # 5. 损失日志（适配LeRobot的info格式）
        info = {
            "loss/total": total_loss.item(),
            "loss/action": loss_act.item(),
            "loss/vision": loss_vis.item(),
            "loss/attention": loss_attn.item(),
            "action/student_mean": student_action.mean().item(),
            "action/teacher_mean": teacher_feat["action"].mean().item(),
        }

        return total_loss, info

    def select_action(self, batch) -> torch.Tensor:
        """重载select_action，仅返回动作（推理时用）"""
        # 屏蔽蒸馏逻辑，仅执行基础动作预测
        action, _ = super().forward(batch)
        return action