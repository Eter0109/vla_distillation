# VLA Knowledge Distillation: XVLA → SmolVLA

将大型 XVLA 教师模型的知识蒸馏到轻量 SmolVLA 学生模型，基于 LIBERO 机器人数据集。

---

## 项目结构

```
vla_distillation/
├── configs/
│   └── distill_config.yaml      # 全局配置（数据集、模型路径、蒸馏超参等）
├── scripts/
│   ├── train_distill.sh         # 启动分布式蒸馏训练
│   └── eval_student.sh          # 评测学生模型
├── src/
│   ├── distill.py               # 蒸馏训练主逻辑（DDP + AMP + 梯度累积）
│   ├── adapters.py              # 特征对齐投影层（禁止修改 lerobot 源码）
│   ├── hooks.py                 # Forward Hook 工具（捕获中间层特征）
│   └── student_policy_wrapper.py # 将学生模型封装为 lerobot Policy 接口
├── outputs/
│   └── checkpoints/             # 训练检查点（自动管理，保留最近3个+best）
└── logs/
    └── train.log                # 训练日志
```

---

## 环境要求

```bash
conda activate lerobot
# lerobot 仓库路径: /hqlab/workspace/zhaozy/lerobot
# 该路径已在脚本中通过 PYTHONPATH 注入，无需额外安装
```

**硬件默认分配：**
- 教师模型（XVLA）：`cuda:1`（单卡，冻结，fp16 推理）
- 学生模型（SmolVLA）：`cuda:2,3,4,5`（4卡 DDP，混合精度训练）

---

## 快速开始

### 1. 检查配置

编辑 `configs/distill_config.yaml`，确认以下关键路径正确：

```yaml
dataset:
  path: /hqlab/workspace/zhaozy/data/libero
  repo_id: libero_local

teacher:
  path: /hqlab/workspace/zhaozy/models/xvla-libero-model
  device: cuda:1

student:
  path: /hqlab/workspace/zhaozy/models/smolvla_libero
  devices: [2, 3, 4, 5]
```

### 2. 启动训练

```bash
# 默认配置（教师 cuda:1，学生 cuda:2-5）
bash scripts/train_distill.sh

# 自定义设备和超参
TEACHER_DEVICE=5 STUDENT_DEVICES=4 \
bash scripts/train_distill.sh \
--batch_size 1 \
--steps 10

# 仅任务损失（无蒸馏）
bash scripts/train_distill.sh --feature_distill false --logit_distill false

# 仅 logit 蒸馏（无特征蒸馏）
bash scripts/train_distill.sh --feature_distill false --logit_distill true
```

### 3. 评测学生模型

```bash
# 自动选取最新检查点
bash scripts/eval_student.sh

# 指定检查点
bash scripts/eval_student.sh --ckpt outputs/checkpoints/step_0030000

# 指定 GPU、任务类型、episode 数
bash scripts/eval_student.sh \
    --ckpt outputs/checkpoints/best \
    --device cuda:2 \
    --task libero_spatial \
    --n_episodes 100 \
    --batch_size 10
```

---

## 技术细节

### 蒸馏损失

```
total_loss = alpha_task × MSE(student_action, GT_action)
           + alpha_distill × (alpha_feature × feat_loss + alpha_logit × kl_loss)
```

- **任务损失**：学生预测动作与真实动作的 MSE（flow matching loss）
- **特征蒸馏**：学生/教师中间层特征经投影后做 MSE（视觉特征 + action expert 最后层）
- **Logit 蒸馏**：学生/教师预测动作的 KL 散度，带温度系数 T

预热阶段（默认前 500 步）仅使用任务损失，蒸馏损失权重为 0。

### 特征对齐

| 组件 | 学生维度 | 教师维度 | 适配层 |
|------|---------|---------|--------|
| 视觉特征 (SigLIP → Florence2) | 576 | 1024 | `VisionFeatureAdapter` |
| Action Expert 最后层 | 288 | 1024 | `ActionExpertFeatureAdapter` |
| 预测动作 (action_dim) | 7 | 20 | `ActionAdapter` |

所有对齐层定义在 `src/adapters.py`，**不修改 lerobot 源码**。

### Hook 挂载点

| 位置 | 模块路径 |
|------|---------|
| 学生视觉 | `student.model.vlm_with_expert.vlm.model.vision_model` |
| 学生 expert 最后层 | `student.model.vlm_with_expert.lm_expert.layers[-1]` |
| 教师视觉 | `teacher.model.vlm.vision_tower` |
| 教师 transformer 最后层 | `teacher.model.transformer.blocks[-1]` |

### DDP + AMP + 梯度累积

```
torchrun --nproc_per_node=4 src/distill.py --config configs/distill_config.yaml
```

- 学生模型用 `DistributedDataParallel` 包装，`find_unused_parameters=True`
- 混合精度：`GradScaler` + `autocast(dtype=torch.float16)`
- 梯度累积：默认 4 步累积一次，等效 batch size = 8 × 4 × 4 = 128
- 梯度裁剪：`max_norm=10.0`
- LR 调度：线性预热（1000步）+ 余弦衰减

---

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `alpha_task` | 1.0 | 任务损失权重 |
| `alpha_distill` | 0.5 | 蒸馏损失总权重 |
| `alpha_logit` | 0.6 | Logit 蒸馏子权重 |
| `alpha_feature` | 0.4 | 特征蒸馏子权重 |
| `warmup_steps` | 500 | 仅任务损失的预热步数 |
| `temperature` | 2.0 | KL 蒸馏温度系数 |
| `total_steps` | 30000 | 总训练步数 |
| `batch_size` | 8 | 每卡 batch size |
| `accum_steps` | 4 | 梯度累积步数 |
| `learning_rate` | 1e-4 | 初始学习率 |
| `use_gradient_checkpointing` | true | 是否启用梯度检查点 |
| `enable_feature_distill` | true | 是否启用特征蒸馏 |
| `enable_logit_distill` | true | 是否启用 logit 蒸馏 |
| `keep_last_n_checkpoints` | 3 | 保留最近 N 个检查点 |

---

## 检查点格式

检查点通过 `model.save_pretrained()` 保存，与 lerobot `from_pretrained` 兼容：

```
outputs/checkpoints/
├── step_0002000/
│   ├── config.json
│   └── model.safetensors
├── step_0004000/
│   ├── config.json
│   └── model.safetensors
└── best/
    ├── config.json
    └── model.safetensors
```

---

## 在代码中加载学生模型

```python
from student_policy_wrapper import load_student_policy

policy = load_student_policy(
    ckpt_path="outputs/checkpoints/best",
    device="cuda:0",
)
# policy 是标准 SmolVLAPolicy，可直接用于 lerobot eval 流程
```

---

## 注意事项

- **不修改 lerobot 源码**：所有新增逻辑（适配层、hook、训练脚本）均在 `vla_distillation/` 下
- 教师模型在推理时全程 `torch.no_grad()` + `fp16 autocast`，不参与反向传播
- 序列长度不同时（教师 vs 学生 token 数），`align_seq_len()` 自动截断/零填充学生特征
- XVLA `transformer.blocks` 若实际为 `.layers`，可在 `configs/distill_config.yaml` 中通过 `teacher.transformer_last_layer_attr` 字段调整（或直接修改 `distill.py` 第 341 行）
