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
- 教师模型（XVLA）：每个 rank 的 `accelerator.device`（冻结，fp16 推理，随学生卡自动分配）
- 学生模型（SmolVLA）：`STUDENT_DEVICES`（默认 `0,1,2,3`，DDP，混合精度训练）

---

## 快速开始

### 1. 检查配置

编辑 `configs/distill_config.yaml`，确认以下关键路径正确：

```yaml
dataset:
  root: /path/to/datasets/libero
  repo_id: libero

distill:
  teacher_path: /path/to/models/xvla-libero-model
  teacher_dtype: float16

student_path: /path/to/models/smolvla_libero
output_dir: outputs/distill
```

### 2. 启动训练

```bash
# 默认配置（学生 cuda:0-3，教师随学生卡自动分配）
bash scripts/train_distill.sh

# 自定义设备和超参
STUDENT_DEVICES=0,1 \
bash scripts/train_distill.sh \
--batch_size 1 \
--steps 10

# 仅任务损失（无蒸馏）
bash scripts/train_distill.sh --feature_distill false --logit_distill false

# 仅 logit 蒸馏（无特征蒸馏）
bash scripts/train_distill.sh --feature_distill false --logit_distill true

# 从指定 checkpoint 完整恢复训练
bash scripts/train_distill.sh --resume true --checkpoint_path outputs/distill/checkpoints/000200

# 不显式指定 checkpoint_path 时，默认恢复 output_dir/checkpoints/last
bash scripts/train_distill.sh --resume true
```

### 3. 评测学生模型

```bash
# 自动选取最新检查点
bash scripts/eval_student.sh

# 指定检查点
bash scripts/eval_student.sh --ckpt outputs/distill/checkpoints/step_0030000

# 指定 GPU、任务类型、episode 数
bash scripts/eval_student.sh \
    --ckpt outputs/distill/checkpoints/best \
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
           + alpha_distill × distill_scale × (
                 alpha_vision_feature × vision_feat_loss
               + alpha_expert_feature × expert_feat_loss
               + alpha_logit × action_distill_loss
             )
```

- **任务损失**：学生预测动作与真实动作的 MSE（flow matching loss）
- **视觉蒸馏**：学生/教师视觉特征经投影后做 MSE，梯度直接回传到学生视觉主干
- **Expert 蒸馏**：当前默认关闭；在 teacher 真实决策路径 hidden state 对齐前，不再使用 dummy target
- **动作蒸馏（logit_distill 分支）**：蒸馏空间固定为 student 7D。默认先将 teacher `20D` 转成
  `abs7`，再结合 `teacher 当前 state` 做 `abs7→rel7`（`Δpos + Δrot(axis-angle) + gripper`），最后按
  student 的 action normalization 统计映射到训练尺度；损失默认 `MSE`，可切 `SmoothL1`/`KL`

`distill/loss_task` 记录的是单个 micro-batch 的原始值，曲线会天然较抖；分析训练趋势时请结合 `distill/loss_task_avg` / `distill/loss_task_ema`，并优先按 `distill/optimizer_step` 对齐不同实验。

预热阶段（`warmup_steps`，按 optimizer step 计）仅使用任务损失；预热结束后，蒸馏权重会在 `distill_ramp_steps` 内逐步爬升。

### 特征对齐

| 组件 | 学生维度 | 教师维度 | 适配层 |
|------|---------|---------|--------|
| 视觉特征 (SigLIP → Florence2) | 576 | 1024 | `VisionFeatureAdapter` |
| Action Expert 最后层 | 288 | 1024 | `ActionExpertFeatureAdapter` |
| 预测动作（蒸馏空间） | 7 | 7（teacher 20D 对齐到 student 7D 语义） | `ActionAdapter` |

所有对齐层定义在 `src/adapters.py`，**不修改 lerobot 源码**。

### Hook 挂载点

| 位置 | 模块路径 |
|------|---------|
| 学生视觉 | `student.model.vlm_with_expert.vlm.model.connector` |
| 学生 expert 最后层 | `student.model.vlm_with_expert.lm_expert.layers[-1]` |
| 教师视觉 | `teacher.model.vlm.vision_tower` |
| 教师 transformer 最后层 | `teacher.model.transformer.blocks[-1]` |

### DDP + AMP + 梯度累积

```
torchrun --nproc_per_node=4 src/distill.py --config configs/distill_config.yaml
```

- 学生模型用 `DistributedDataParallel` 包装，`find_unused_parameters=True`
- 混合精度：`GradScaler` + `autocast(dtype=torch.float16)`
- 梯度累积：日志中的 `step` 是 micro step，真实参数更新请看 `optimizer_step`
- 梯度裁剪：`max_norm=10.0`
- LR 调度：线性预热（1000步）+ 余弦衰减

---

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `alpha_task` | 1.0 | 任务损失权重 |
| `alpha_distill` | 0.2 | 蒸馏损失总权重 |
| `alpha_vision_feature` | 0.2 | 视觉蒸馏子权重 |
| `alpha_expert_feature` | 0.0 | Expert 蒸馏子权重（当前默认关闭） |
| `alpha_logit` | 0.0 | Logit 蒸馏子权重（当前默认关闭） |
| `warmup_steps` | 250 | 仅任务损失的预热步数，按 optimizer step 计 |
| `distill_ramp_steps` | 500 | 蒸馏权重从 0 逐步升到 1 的 optimizer step 数 |
| `temperature` | 2.0 | 仅在 `action_distill_loss=kl` 时生效 |
| `action_distill_loss` | `mse` | 动作蒸馏损失类型：`mse` / `smooth_l1` / `kl` |
| `action_distill_horizon` | `1` | 动作蒸馏使用的 chunk 前几步；`1` 表示仅蒸馏首步，`<=0` 表示使用 teacher/student 共有的整段 chunk |
| `action_align_mode` | `teacher_abs20_to_student_rel7` | 动作对齐模式（默认 teacher 20D 对齐到 student 7D 语义；保留 `xvla_libero_20to7` 兼容模式） |
| `teacher_transformer_last_layer_attr` | `blocks` | 教师 transformer 最后一层容器名，默认解析 `teacher.model.transformer.blocks[-1]` |
| `total_steps` | 30000 | 总训练步数 |
| `batch_size` | 8 | 每卡 batch size |
| `accum_steps` | 4 | 梯度累积步数 |
| `learning_rate` | 1e-4 | 初始学习率 |
| `use_gradient_checkpointing` | true | 是否启用梯度检查点 |
| `vision_feature_distill` | true | 是否启用视觉蒸馏 |
| `expert_feature_distill` | false | 是否启用 expert 蒸馏 |
| `logit_distill` | false | 是否启用 logit 蒸馏 |
| `keep_last_n_checkpoints` | 3 | 保留最近 N 个检查点 |

---

## 检查点格式

检查点通过 `model.save_pretrained()` 保存，与 lerobot `from_pretrained` 兼容：

```
outputs/distill/checkpoints/
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
    ckpt_path="outputs/distill/checkpoints/best",
    device="cuda:0",
)
# policy 是标准 SmolVLAPolicy，可直接用于 lerobot eval 流程
```

---

## 注意事项

- **不修改 lerobot 源码**：所有新增逻辑（适配层、hook、训练脚本）均在 `vla_distillation/` 下
- 教师模型在推理时全程 `torch.no_grad()` + `fp16 autocast`，不参与反向传播
- 序列长度不同时（教师 vs 学生 token 数），`align_seq_len()` 自动截断/零填充学生特征
- XVLA `transformer.blocks` 若实际为 `.layers`，可在 `configs/distill_config.yaml` 中通过 `distill.teacher_transformer_last_layer_attr` 字段调整
- `resume=true` 时会从 checkpoint 根目录下的 `pretrained_model/`、`training_state/` 和 `adapters.pt` 一起恢复；`checkpoint_path` 应指向如 `outputs/distill/checkpoints/000200` 的目录，而不是其 `pretrained_model/` 子目录
- `scripts/eval_student.sh` 在缺少 gym/libero wrapper 时只做模型加载验证，不代表真实 LIBERO 成功率
