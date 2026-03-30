import torch
import torch.nn.functional as F
# 修正1：导入正确路径的 get_policy_class
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
# 修正2：导入自定义的 FeatureHook（需确保 hooks.py 在同目录）
from hooks import FeatureHook
# 假设 DistillSmolVLA 是自定义的 SmolVLA 蒸馏类，需确保该文件在同目录
from distill_policy import DistillSmolVLA

# ======================
# 0. 基础配置（适配 LeRobot 数据集/设备）
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_id = "lerobot/pusht"  # 替换为你的数据集
output_dir = "./outputs/distill_smolvla"

# ======================
# 1. 加载模型（严格适配 LeRobot 接口）
# ======================

# 学生模型：SmolVLA（自定义蒸馏版）
# Step1: 获取 SmolVLA 配置和默认参数
smolvla_cfg = make_policy_config("smolvla")
# Step2: 加载数据集元信息（用于初始化模型输入/输出特征）
dataset_meta = LeRobotDatasetMetadata(dataset_id)
features = dataset_meta.features
# Step3: 初始化学生模型（DistillSmolVLA 需继承 PreTrainedPolicy）
student = DistillSmolVLA(smolvla_cfg)
# 加载预训练权重（替换为你的权重路径）
student.load_state_dict(torch.load("your_pretrained_smolvla.pt", map_location=device), strict=False)
student.to(device)
student.train()  # 训练模式

# 教师模型：XVLA（冻结参数）
# Step1: 获取 XVLA 预训练模型（LeRobot 标准加载方式）
xvla_cls = get_policy_class("xvla")
xvla_cfg = make_policy_config("xvla")
teacher = xvla_cls.from_pretrained("lerobot/xvla")  # 或本地路径
teacher.to(device)
teacher.eval()  # 评估模式
# 冻结所有参数
for p in teacher.parameters():
    p.requires_grad = False

# ======================
# 2. 注册 Hook（提取中间特征）
# ======================
# 注意：需根据 SmolVLA/XVLA 实际模型结构调整 Hook 位置
# 视觉特征 Hook
hook_s_vis = FeatureHook(student.model.vlm.model.vision_model)
hook_t_vis = FeatureHook(teacher.model.vlm.vision_tower)

# 注意力层 Hook
hook_s_attn = FeatureHook(student.model.lm_expert.layers[-1].self_attn)
hook_t_attn = FeatureHook(teacher.model.transformer.layers[-1].self_attn)

# ======================
# 3. 数据集与优化器（LeRobot 原生接口）
# ======================
# 构建数据集（适配 SmolVLA/XVLA 的 delta_timestamps 要求）
delta_timestamps = {
    "observation.image": [-0.1, 0.0],  # 根据模型需求调整
    "observation.state": [-0.1, 0.0],
    "action": [0.0],
}
dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)

# 优化器
optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

# ======================
# 4. 蒸馏训练循环
# ======================
training_steps = 5000  # 训练步数
log_freq = 10  # 日志打印频率
step = 0
done = False

while not done:
    for batch in dataloader:
        # 数据预处理：移到设备 + 适配模型输入
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        
        optimizer.zero_grad()

        # 学生模型前向传播（获取动作输出）
        s_act = student.select_action(batch)  # LeRobot 标准推理接口

        # 教师模型前向传播（无梯度）
        with torch.no_grad():
            t_act = teacher.select_action(batch)

        # ==================
        # 分层蒸馏损失计算
        # ==================
        # 1. 动作蒸馏损失（MSE）
        loss_act = F.mse_loss(s_act, t_act)

        # 2. 视觉特征蒸馏损失（MSE，需确保维度匹配）
        s_vis = hook_s_vis.output.last_hidden_state
        t_vis = hook_t_vis.output
        # 适配维度（示例：若学生特征维度≠教师，加投影层）
        if hasattr(student, "proj"):
            s_vis = student.proj(s_vis)
        else:
            # 动态创建投影层（仅首次执行）
            proj = nn.Linear(s_vis.shape[-1], t_vis.shape[-1], device=device)
            student.proj = proj
            s_vis = student.proj(s_vis)
        loss_vis = F.mse_loss(s_vis, t_vis)

        # 3. 注意力蒸馏损失（KL散度）
        s_attn = hook_s_attn.output
        t_attn = hook_t_attn.output
        # 确保注意力维度匹配 + 归一化
        loss_attn = F.kl_div(
            s_attn.log_softmax(dim=-1),
            t_attn.softmax(dim=-1),
            reduction="batchmean"
        )

        # 总损失（权重可调整）
        loss = loss_act + 0.3 * loss_vis + 0.2 * loss_attn

        # 反向传播 + 优化
        loss.backward()
        optimizer.step()

        # 日志打印
        if step % log_freq == 0:
            print(f"Step: {step} | Total Loss: {loss.item():.4f} | "
                  f"Act Loss: {loss_act.item():.4f} | Vis Loss: {loss_vis.item():.4f} | "
                  f"Attn Loss: {loss_attn.item():.4f}")

        step += 1
        if step >= training_steps:
            done = True
            break

# ======================
# 5. 保存模型
# ======================
student.save_pretrained(output_dir)
print(f"蒸馏完成，模型已保存至 {output_dir}")