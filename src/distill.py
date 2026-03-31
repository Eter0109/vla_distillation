"""
蒸馏训练核心逻辑（src/distill.py）

功能：
- 教师模型（XVLA）单卡半精度推理，冻结参数
- 学生模型（SmolVLA）多卡 DDP 训练，混合精度 + 梯度检查点 + 梯度累积
- 多层特征蒸馏 + 输出层 Logit 蒸馏（开关可配）
- 损失 = alpha_task * MSE(学生动作, GT动作) + alpha_distill * 蒸馏损失
- 支持通过命令行覆盖 YAML 配置项
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import yaml

# 添加 lerobot 到路径
LEROBOT_ROOT = "/hqlab/workspace/zhaozy/lerobot/src"
if LEROBOT_ROOT not in sys.path:
    sys.path.insert(0, LEROBOT_ROOT)

PROJECT_SRC = Path(__file__).parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from adapters import DistillAdapters, align_seq_len
from hooks import MultiHookManager


# =============================================================================
# 工具函数
# =============================================================================

def setup_logging(log_file: str, rank: int) -> logging.Logger:
    """配置日志：主进程写文件+控制台，其他进程仅控制台。"""
    logger = logging.getLogger("distill")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    if rank == 0:
        os.makedirs(Path(log_file).parent, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_config(config_path: str) -> dict:
    """从 YAML 文件加载配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，支持覆盖 YAML 配置。"""
    parser = argparse.ArgumentParser(description="VLA 蒸馏训练 SmolVLA <- XVLA")
    parser.add_argument("--config", type=str, default="configs/distill_config.yaml", help="YAML 配置文件路径")

    # 命令行覆盖项
    parser.add_argument("--teacher_device", type=str, default=None, help="教师模型部署卡，如 cuda:1")
    parser.add_argument("--student_devices", type=str, default=None, help="学生训练卡，如 2,3,4,5")
    parser.add_argument("--alpha", type=float, default=None, help="蒸馏损失总权重 alpha_distill")
    parser.add_argument("--batch_size", type=int, default=None, help="每卡 batch size")
    parser.add_argument("--steps", type=int, default=None, help="总训练步数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--accum_steps", type=int, default=None, help="梯度累积步数")
    parser.add_argument("--grad_ckpt", type=lambda x: x.lower() == "true", default=None, help="是否启用梯度检查点")
    parser.add_argument("--feature_distill", type=lambda x: x.lower() == "true", default=None, help="是否启用特征蒸馏")
    parser.add_argument("--logit_distill", type=lambda x: x.lower() == "true", default=None, help="是否启用 logit 蒸馏")
    parser.add_argument("--dataset_path", type=str, default=None, help="本地数据集路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--student_cuda_offset", type=int, default=0, help="学生 local_rank 相对于 CUDA_VISIBLE_DEVICES 的偏移（教师占据前 N 个槽位时使用）")
    # DDP 通信参数（由 torchrun 自动注入）
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """将命令行参数覆盖到配置字典中。"""
    if args.teacher_device:
        cfg["teacher"]["device"] = args.teacher_device
    if args.student_devices:
        devices = [int(d) for d in args.student_devices.split(",")]
        cfg["student"]["devices"] = devices
    if args.alpha is not None:
        cfg["distillation"]["alpha_distill"] = args.alpha
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.steps is not None:
        cfg["training"]["total_steps"] = args.steps
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.accum_steps is not None:
        cfg["training"]["accum_steps"] = args.accum_steps
    if args.grad_ckpt is not None:
        cfg["student"]["use_gradient_checkpointing"] = args.grad_ckpt
    if args.feature_distill is not None:
        cfg["distillation"]["enable_feature_distill"] = args.feature_distill
    if args.logit_distill is not None:
        cfg["distillation"]["enable_logit_distill"] = args.logit_distill
    if args.dataset_path:
        cfg["dataset"]["path"] = args.dataset_path
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
        cfg["paths"]["checkpoint_dir"] = str(Path(args.output_dir) / "checkpoints")
    if hasattr(args, "student_cuda_offset") and args.student_cuda_offset:
        cfg["student"]["cuda_offset"] = args.student_cuda_offset
    return cfg


def validate_paths(cfg: dict, logger: logging.Logger) -> None:
    """检查关键路径是否存在。"""
    errors = []
    teacher_path = cfg["teacher"]["path"]
    student_path = cfg["student"]["path"]
    dataset_path = cfg["dataset"]["path"]

    if not Path(teacher_path).exists():
        errors.append(f"教师模型路径不存在: {teacher_path}")
    if not Path(student_path).exists():
        errors.append(f"学生模型路径不存在: {student_path}")
    if not Path(dataset_path).exists():
        errors.append(f"数据集路径不存在: {dataset_path}")

    if errors:
        for e in errors:
            logger.error(e)
        raise FileNotFoundError("\n".join(errors))


def setup_ddp(local_rank: int, student_cuda_offset: int = 0) -> int:
    """初始化 DDP 进程组，返回全局 rank。"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank + student_cuda_offset)
    return rank, world_size


def cleanup_ddp():
    """销毁 DDP 进程组。"""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# 模型加载
# =============================================================================

def load_teacher(cfg: dict, rank: int, logger: logging.Logger) -> XVLAPolicy | None:
    """加载教师模型到指定卡；为避免 OOM，仅 rank0 持有教师。"""
    if rank != 0:
        logger.info("非主 rank 跳过教师模型加载，仅计算 task loss")
        return None

    teacher_path = cfg["teacher"]["path"]
    teacher_device = cfg["teacher"]["device"]
    teacher_dtype_name = str(cfg["teacher"].get("dtype", "float16")).lower()
    teacher_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(teacher_dtype_name, torch.float16)

    logger.info(f"加载教师模型: {teacher_path} -> {teacher_device} ({teacher_dtype_name})")
    teacher = XVLAPolicy.from_pretrained(teacher_path, dtype=teacher_dtype)
    teacher = teacher.to(device=teacher_device, dtype=teacher_dtype)

    # 冻结所有参数
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    logger.info(f"教师模型参数已冻结，部署在 {teacher_device}")
    return teacher


def load_student(cfg: dict, local_rank: int, logger: logging.Logger) -> SmolVLAPolicy:
    """加载学生模型，移动到当前卡，（可选）启用梯度检查点。"""
    student_path = cfg["student"]["path"]
    # 支持教师模型占据 cuda:0 的情况，学生卡从 offset 开始
    cuda_offset = cfg["student"].get("cuda_offset", 0)
    student_device = f"cuda:{local_rank + cuda_offset}"

    logger.info(f"加载学生模型: {student_path} -> {student_device}")
    student = SmolVLAPolicy.from_pretrained(student_path)
    student = student.to(student_device)

    # 启用梯度检查点（节省显存）
    if cfg["student"].get("use_gradient_checkpointing", False):
        vlm_model = student.model.vlm_with_expert
        if hasattr(vlm_model.vlm, "gradient_checkpointing_enable"):
            vlm_model.vlm.gradient_checkpointing_enable()
            logger.info("已启用学生 VLM 梯度检查点")
        if hasattr(vlm_model.lm_expert, "gradient_checkpointing_enable"):
            vlm_model.lm_expert.gradient_checkpointing_enable()
            logger.info("已启用学生 lm_expert 梯度检查点")

    student.train()
    return student


# =============================================================================
# 数据集
# =============================================================================

def build_dataloader(cfg: dict, rank: int, world_size: int) -> DataLoader:
    """构建分布式 DataLoader。"""
    dataset_path = cfg["dataset"]["path"]
    repo_id = cfg["dataset"]["repo_id"]
    delta_timestamps = cfg["dataloader"]["delta_timestamps"]

    # 当前 lerobot 版本不支持 local_files_only，直接按本地 root + repo_id 加载
    try:
        _ = LeRobotDatasetMetadata(repo_id, root=dataset_path)
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_path,
            delta_timestamps=delta_timestamps,
        )
    except Exception:
        # 回退：直接用 path 作为 repo_id
        dataset = LeRobotDataset(
            repo_id=dataset_path,
            delta_timestamps=delta_timestamps,
        )

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=cfg["dataloader"].get("pin_memory", True),
        drop_last=True,
    )
    return loader, sampler


# =============================================================================
# 损失计算
# =============================================================================

def compute_distill_loss(
    student: SmolVLAPolicy,
    teacher: XVLAPolicy | None,
    batch: dict,
    adapters: DistillAdapters,
    cfg: dict,
    teacher_device: str,
    student_device: str,
    step: int,
    logger: logging.Logger,
) -> tuple[torch.Tensor, dict]:
    """计算完整蒸馏损失。

    Returns:
        total_loss: 标量张量（在学生设备上）
        log_info: 各项损失的数值字典
    """
    distill_cfg = cfg["distillation"]
    alpha_task = distill_cfg["alpha_task"]
    alpha_distill = distill_cfg["alpha_distill"]
    alpha_logit = distill_cfg.get("alpha_logit", 0.6)
    alpha_feature = distill_cfg.get("alpha_feature", 0.4)
    warmup_steps = distill_cfg.get("warmup_steps", 0)
    temperature = distill_cfg.get("temperature", 2.0)
    enable_logit = distill_cfg.get("enable_logit_distill", True)
    enable_feature = distill_cfg.get("enable_feature_distill", True)

    # 是否在预热阶段（仅使用任务损失）
    in_warmup = step < warmup_steps

    # -------------------------------------------------------------------
    # 1. 学生模型前向（任务损失）
    # -------------------------------------------------------------------
    # 将 batch 移动到学生设备
    student_batch = {k: v.to(student_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # 注册学生中间层 hook
    hook_modules = {}
    # 在 DDP 包装下，通过 student.module 访问原始模型属性
    student_module = student.module if isinstance(student, DDP) else student
    if enable_feature and not in_warmup:
        vlm_expert = student_module.model.vlm_with_expert
        hook_modules["student_vision"] = vlm_expert.vlm.model.vision_model
        hook_modules["student_last"] = vlm_expert.lm_expert.layers[-1]

    with MultiHookManager(hook_modules) as student_hooks:
        # 学生 forward：返回 (task_loss, loss_dict)
        student_task_loss, student_loss_dict = student.forward(student_batch)

    # -------------------------------------------------------------------
    # 2. 获取学生预测动作（用于 logit 蒸馏）
    # -------------------------------------------------------------------
    student_pred_action = None
    if enable_logit and not in_warmup:
        with torch.no_grad():
            # sample_actions 返回 (B, chunk_size, action_dim)
            images, img_masks = student_module.prepare_images(student_batch)
            state = student_module.prepare_state(student_batch)
            lang_tokens = student_batch[OBS_LANGUAGE_TOKENS]
            lang_masks = student_batch[OBS_LANGUAGE_ATTENTION_MASK]
            student_pred_action = student_module.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state
            ).detach()
            # 截断到真实动作维度
            real_action_dim = cfg["student"]["action_dim"]
            student_pred_action = student_pred_action[:, :, :real_action_dim]

    # -------------------------------------------------------------------
    # 3. 教师模型前向（仅推理，AMP 半精度）
    # -------------------------------------------------------------------
    teacher_vision_feat = None
    teacher_last_feat = None
    teacher_pred_action = None

    if not in_warmup and teacher is not None and (enable_logit or enable_feature):
        teacher_batch = {
            k: v.to(teacher_device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        teacher_hook_modules = {}
        if enable_feature:
            teacher_hook_modules["teacher_vision"] = teacher.model.vlm.vision_tower
            teacher_hook_modules["teacher_last"] = teacher.model.transformer.blocks[-1]

        with MultiHookManager(teacher_hook_modules) as teacher_hooks:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                # 获取教师 VLM 特征
                teacher_inputs = teacher._build_model_inputs(teacher_batch)
                enc = teacher.model.forward_vlm(
                    input_ids=teacher_inputs["input_ids"],
                    pixel_values=teacher_inputs["image_input"],
                    image_mask=teacher_inputs["image_mask"],
                )
                teacher_vlm_feat = enc["vlm_features"]  # (B, L_t, D_teacher_vlm)

                # 获取教师预测动作
                if enable_logit:
                    teacher_pred_action = teacher.model.generate_actions(
                        input_ids=teacher_inputs["input_ids"],
                        image_input=teacher_inputs["image_input"],
                        image_mask=teacher_inputs["image_mask"],
                        domain_id=teacher_inputs["domain_id"],
                        proprio=teacher_inputs["proprio"],
                        steps=teacher.config.num_denoising_steps,
                    )  # (B, chunk_size, action_dim)

            # 捕获教师中间层特征
            if enable_feature:
                teacher_vision_feat = teacher_hooks["teacher_vision"].output
                teacher_last_feat = teacher_hooks["teacher_last"].output

        # 立即释放教师临时特征，清理显存
        del teacher_inputs
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # 4. 蒸馏损失
    # -------------------------------------------------------------------
    distill_loss = torch.tensor(0.0, device=student_device)
    log_info = {}

    if not in_warmup:
        # 4a. 视觉特征蒸馏
        if enable_feature and teacher_vision_feat is not None:
            sv = student_hooks["student_vision"].output  # (B, L_s, D_student)
            tv = teacher_vision_feat.to(student_device).float()  # (B, L_t, D_teacher)

            # 序列长度对齐
            sv = align_seq_len(sv.float(), tv)
            # 维度对齐（投影到教师维度）
            sv_proj = adapters.adapt_vision(sv)
            loss_vis = F.mse_loss(sv_proj, tv)
            distill_loss = distill_loss + alpha_feature * loss_vis
            log_info["loss_vision_feat"] = loss_vis.item()

            # 释放
            del tv, sv, sv_proj
            torch.cuda.empty_cache()

        # 4b. Action Expert 最后层特征蒸馏
        if enable_feature and teacher_last_feat is not None:
            se = student_hooks["student_last"].output.float()  # (B, L_s, D_student_exp)
            te = teacher_last_feat.to(student_device).float()    # (B, L_t, D_teacher)

            se = align_seq_len(se, te)
            se_proj = adapters.adapt_expert(se)
            loss_expert = F.mse_loss(se_proj, te)
            distill_loss = distill_loss + alpha_feature * loss_expert
            log_info["loss_expert_feat"] = loss_expert.item()

            del te, se, se_proj
            torch.cuda.empty_cache()

        # 4c. 输出层 Logit（动作）KL 散度蒸馏
        if enable_logit and teacher_pred_action is not None and student_pred_action is not None:
            t_action = teacher_pred_action.to(student_device).float()  # (B, T_t, D_t)
            s_action = student_pred_action.float()                     # (B, T_s, D_s)

            # 对齐 chunk 维度
            T_min = min(t_action.shape[1], s_action.shape[1])
            t_action = t_action[:, :T_min, :]
            s_action = s_action[:, :T_min, :]

            # 将学生动作投影到教师动作维度
            s_action_proj = adapters.adapt_action(s_action)  # (B, T_min, D_t)

            # KL 散度（在动作维度上）
            T_kl = temperature
            log_p_s = F.log_softmax(s_action_proj / T_kl, dim=-1)
            p_t = F.softmax(t_action / T_kl, dim=-1)
            loss_kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T_kl ** 2)
            distill_loss = distill_loss + alpha_logit * loss_kl
            log_info["loss_kl_action"] = loss_kl.item()

            del t_action, s_action, s_action_proj
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # 5. 总损失
    # -------------------------------------------------------------------
    in_warmup_flag = 1.0 if in_warmup else 0.0
    total_loss = alpha_task * student_task_loss + (1.0 - in_warmup_flag) * alpha_distill * distill_loss

    log_info["loss_task"] = student_task_loss.item()
    log_info["loss_distill"] = distill_loss.item()
    log_info["teacher_enabled"] = teacher is not None and not in_warmup

    return total_loss, log_info


# =============================================================================
# 检查点管理
# =============================================================================

def save_checkpoint(student: SmolVLAPolicy, step: int, cfg: dict, is_best: bool = False) -> None:
    """保存学生模型检查点（仅主进程）。"""
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_path = ckpt_dir / f"step_{step:07d}"
    # 从 DDP 包装中取原始模型
    model = student.module if isinstance(student, DDP) else student
    model.save_pretrained(str(save_path))

    if is_best:
        best_path = ckpt_dir / "best"
        model.save_pretrained(str(best_path))

    # 清理旧检查点（保留最近 N 个 + best）
    keep_n = cfg["training"].get("keep_last_n_checkpoints", 3)
    ckpt_dirs = sorted(
        [p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    while len(ckpt_dirs) > keep_n:
        oldest = ckpt_dirs.pop(0)
        import shutil
        shutil.rmtree(oldest)


# =============================================================================
# 主训练循环
# =============================================================================

def train(cfg: dict, local_rank: int, rank: int, world_size: int) -> None:
    """主训练函数。"""
    is_main = rank == 0
    log_file = cfg["paths"].get("log_file", "logs/train.log")
    logger = setup_logging(log_file, rank)

    if is_main:
        logger.info("=" * 60)
        logger.info("VLA 蒸馏训练启动")
        logger.info(f"教师设备: {cfg['teacher']['device']}")
        logger.info(f"学生设备数: {world_size}，local_rank={local_rank}")
        logger.info(f"特征蒸馏: {cfg['distillation']['enable_feature_distill']}")
        logger.info(f"Logit蒸馏: {cfg['distillation']['enable_logit_distill']}")
        logger.info("=" * 60)

    validate_paths(cfg, logger)
    torch.manual_seed(cfg["training"].get("seed", 42) + rank)

    student_device = f"cuda:{local_rank}"
    teacher_device = cfg["teacher"]["device"]

    # 若学生 GPU 有偏移（教师占用 cuda:0 等），重新计算学生设备
    cuda_offset = cfg["student"].get("cuda_offset", 0)
    if cuda_offset > 0:
        student_device = f"cuda:{local_rank + cuda_offset}"

    # -------------------------------------------------------------------
    # 加载模型
    # -------------------------------------------------------------------
    teacher = load_teacher(cfg, rank, logger)
    student = load_student(cfg, local_rank, logger)

    # 用 DDP 包装学生模型
    student_ddp = DDP(
        student,
        device_ids=[int(student_device.split(":")[-1])],
        output_device=int(student_device.split(":")[-1]),
        find_unused_parameters=True,  # SmolVLA 有部分冻结参数
    )

    # -------------------------------------------------------------------
    # 适配层（在学生设备上）
    # -------------------------------------------------------------------
    distill_cfg = cfg["distillation"]
    adapters = DistillAdapters(
        student_vision_dim=cfg["student"]["vlm_feat_dim"],
        teacher_vision_dim=cfg["teacher"]["vlm_feat_dim"],
        student_expert_dim=cfg["student"]["expert_hidden_size"],
        teacher_expert_dim=cfg["teacher"]["transformer_hidden_size"],
        student_action_dim=cfg["student"]["action_dim"],
        teacher_action_dim=cfg["teacher"]["action_dim"],
        enable_feature_distill=distill_cfg["enable_feature_distill"],
        enable_logit_distill=distill_cfg["enable_logit_distill"],
    ).to(student_device)

    # -------------------------------------------------------------------
    # 优化器 & 调度器
    # -------------------------------------------------------------------
    train_cfg = cfg["training"]
    # 学生模型可训练参数 + 适配层参数一起优化
    optim_params = [
        {"params": [p for p in student.parameters() if p.requires_grad]},
        {"params": list(adapters.parameters())},
    ]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg.get("optimizer_betas", [0.9, 0.95])),
        eps=train_cfg.get("optimizer_eps", 1e-8),
        weight_decay=train_cfg.get("weight_decay", 1e-6),
    )

    total_steps = train_cfg["total_steps"]
    warmup_steps_sched = train_cfg.get("scheduler_warmup_steps", 1000)
    decay_steps = train_cfg.get("scheduler_decay_steps", 25000)
    decay_lr = train_cfg.get("scheduler_decay_lr", 2.5e-6)

    def lr_lambda(current_step: int) -> float:
        """余弦衰减 + 线性预热。"""
        base_lr = train_cfg["learning_rate"]
        if current_step < warmup_steps_sched:
            return float(current_step) / max(1, warmup_steps_sched)
        progress = (current_step - warmup_steps_sched) / max(1, decay_steps - warmup_steps_sched)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(decay_lr / base_lr, cosine_factor)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 混合精度
    scaler = torch.amp.GradScaler("cuda")

    # -------------------------------------------------------------------
    # 数据集
    # -------------------------------------------------------------------
    loader, sampler = build_dataloader(cfg, rank, world_size)

    # -------------------------------------------------------------------
    # W&B（仅主进程）
    # -------------------------------------------------------------------
    wandb_run = None
    if is_main and cfg.get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wb_cfg = cfg["wandb"]
            os.makedirs(wb_cfg.get("log_dir", "logs/wandb"), exist_ok=True)
            wandb_run = wandb.init(
                project=wb_cfg.get("project", "vla_distillation"),
                name=wb_cfg.get("run_name", "smolvla_from_xvla"),
                dir=wb_cfg.get("log_dir", "logs/wandb"),
                config=cfg,
            )
            logger.info("W&B 已启用")
        except Exception as e:
            logger.warning(f"W&B 初始化失败: {e}，跳过")

    # -------------------------------------------------------------------
    # 训练循环
    # -------------------------------------------------------------------
    accum_steps = train_cfg.get("accum_steps", 1)
    log_every = train_cfg.get("log_every_steps", 50)
    save_every = train_cfg.get("save_every_steps", 2000)
    grad_clip = train_cfg.get("grad_clip_norm", 10.0)

    step = 0
    best_loss = float("inf")
    data_iter = iter(loader)

    optimizer.zero_grad()
    t_start = time.time()

    while step < total_steps:
        sampler.set_epoch(step // len(loader) if len(loader) > 0 else 0)

        # 从迭代器取一个 batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # 梯度累积内循环
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            loss, log_info = compute_distill_loss(
                student=student_ddp,
                teacher=teacher,
                batch=batch,
                adapters=adapters,
                cfg=cfg,
                teacher_device=teacher_device,
                student_device=student_device,
                step=step,
                logger=logger,
            )
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        # 梯度累积：每 accum_steps 步更新一次
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad]
                + list(adapters.parameters()),
                grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        step += 1

        # -------------------------------------------------------------------
        # 日志
        # -------------------------------------------------------------------
        if is_main and step % log_every == 0:
            elapsed = time.time() - t_start
            lr_now = optimizer.param_groups[0]["lr"]
            mem_used = torch.cuda.max_memory_allocated(student_device) / 1024 ** 3
            logger.info(
                f"step={step}/{total_steps} | "
                f"loss={log_info['loss_total']:.4f} "
                f"(task={log_info['loss_task']:.4f} distill={log_info['loss_distill']:.4f}) | "
                f"lr={lr_now:.2e} | mem={mem_used:.2f}GB | "
                f"elapsed={elapsed:.0f}s | warmup={log_info['in_warmup']}"
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "step": step,
                        "loss/total": log_info["loss_total"],
                        "loss/task": log_info["loss_task"],
                        "loss/distill": log_info["loss_distill"],
                        **{f"loss/{k}": v for k, v in log_info.items() if k.startswith("loss_") and k not in ("loss_task", "loss_distill", "loss_total")},
                        "lr": lr_now,
                        "gpu_mem_gb": mem_used,
                    }
                )

        # -------------------------------------------------------------------
        # 检查点
        # -------------------------------------------------------------------
        if is_main and step % save_every == 0:
            cur_loss = log_info["loss_total"]
            is_best = cur_loss < best_loss
            if is_best:
                best_loss = cur_loss
            save_checkpoint(student_ddp, step, cfg, is_best=is_best)
            logger.info(f"已保存检查点 step={step}，is_best={is_best}")

    # 训练结束保存最终检查点
    if is_main:
        save_checkpoint(student_ddp, step, cfg, is_best=False)
        logger.info(f"训练完成！共 {step} 步，最终检查点已保存。")
        if wandb_run:
            wandb_run.finish()

    cleanup_ddp()


# =============================================================================
# 入口
# =============================================================================

def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # 初始化 DDP
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    student_cuda_offset = cfg.get("student", {}).get("cuda_offset", 0)
    rank, world_size = setup_ddp(local_rank, student_cuda_offset)

    train(cfg, local_rank, rank, world_size)


if __name__ == "__main__":
    main()
