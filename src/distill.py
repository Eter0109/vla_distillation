#!/usr/bin/env python
"""
VLA 蒸馏训练脚本（src/distill.py）

结构完全对齐 lerobot/scripts/lerobot_train.py：
- 配置：DistillTrainPipelineConfig 继承 TrainPipelineConfig + @parser.wrap()
- 更新函数：update_distill 对应 update_policy
- 主函数：train_distill 对应 train，结构、命名与 lerobot_train.py 一致
- 工具：MetricsTracker / AverageMeter / WandBLogger / save_checkpoint

教师模型（XVLA）：
  - 仅主进程加载，fp16 冻结推理，部署在 teacher_device（默认 cuda:1）
  - 中间特征通过 accelerator.broadcast 同步到所有 DDP 进程

学生模型（SmolVLA）：
  - accelerator.prepare 封装 DDP + AMP
  - 与 DistillAdapters 联合优化
"""

import dataclasses
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

# LeRobot 核心模块
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
try:
    from lerobot.rl.wandb_utils import WandBLogger
except ImportError:
    WandBLogger = None
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    init_logging,
    inside_slurm,
)

# 自定义蒸馏模块
from adapters import DistillAdapters, align_seq_len
from hooks import MultiHookManager
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


# =============================================================================
# 配置扩展
# =============================================================================

@dataclass
class DistillConfig:
    """蒸馏超参，作为 DistillTrainPipelineConfig 的子字段。"""
    # 损失权重
    alpha_task: float = 1.0
    alpha_distill: float = 0.5
    alpha_feature: float = 0.4
    alpha_logit: float = 0.6
    # 蒸馏开关
    feature_distill: bool = True
    logit_distill: bool = True
    # 预热：warmup 期间只用 task loss
    warmup_steps: int = 500
    temperature: float = 2.0
    # 教师模型
    teacher_path: str = ""
    teacher_device: str = "cuda:0"
    teacher_dtype: str = "float16"
    # 适配层维度
    student_vision_dim: int = 960
    teacher_vision_dim: int = 1024
    student_expert_dim: int = 480
    teacher_expert_dim: int = 1024
    student_action_dim: int = 7
    teacher_action_dim: int = 20


@dataclass
class DistillTrainPipelineConfig(TrainPipelineConfig):
    """在 LeRobot TrainPipelineConfig 基础上追加蒸馏配置。"""
    distill: DistillConfig = field(default_factory=DistillConfig)
    # 学生模型路径（同 policy.pretrained_path，单独保留以兼容现有 YAML）
    student_path: str = ""
    # 梯度累积步数
    grad_accum_steps: int = 1

    def validate(self) -> None:
        # 跳过父类对 policy 的强制检验（蒸馏场景下 policy 由 student_path 控制）
        import datetime as dt
        if not self.job_name:
            self.job_name = "distill_smolvla"
        if not self.output_dir:
            now = dt.datetime.now()
            self.output_dir = Path("outputs/train") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
        if self.resume and self.output_dir and not Path(self.output_dir).is_dir():
            raise NotADirectoryError(f"resume=True 但 output_dir 不存在: {self.output_dir}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        # 不走 policy 路径加载
        return []


# =============================================================================
# 模型加载
# =============================================================================

def load_teacher(cfg: DistillTrainPipelineConfig, accelerator: Accelerator) -> PreTrainedPolicy | None:
    """加载教师模型（各 rank 本地加载），fp16 冻结部署在当前 rank device。"""
    dc = cfg.distill
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    teacher_dtype = dtype_map[dc.teacher_dtype.lower()]

    logging.info(colored(f"加载教师模型: {dc.teacher_path} -> {accelerator.device}", "blue"))
    teacher_cls = get_policy_class("xvla")
    teacher = teacher_cls.from_pretrained(dc.teacher_path)
    teacher = teacher.to(device=accelerator.device, dtype=teacher_dtype)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    logging.info(colored(f"教师模型已就绪: {accelerator.device} ({teacher_dtype})", "green"))
    return teacher


def load_student(cfg: DistillTrainPipelineConfig) -> PreTrainedPolicy:
    """加载学生模型（SmolVLA），启用梯度检查点。"""
    path = cfg.student_path
    logging.info(colored(f"加载学生模型: {path}", "blue"))
    student_cls = get_policy_class("smolvla")
    student = student_cls.from_pretrained(path)

    vlm = getattr(student.model.vlm_with_expert, "vlm", None)
    lm_expert = getattr(student.model.vlm_with_expert, "lm_expert", None)
    if vlm is not None and hasattr(vlm, "gradient_checkpointing_enable"):
        vlm.gradient_checkpointing_enable()
        logging.info("学生 VLM 梯度检查点已启用")
    if lm_expert is not None and hasattr(lm_expert, "gradient_checkpointing_enable"):
        lm_expert.gradient_checkpointing_enable()
        logging.info("学生 lm_expert 梯度检查点已启用")

    student.train()
    return student


# =============================================================================
# 单步蒸馏更新（对应 lerobot_train.update_policy）
# =============================================================================

def update_distill(
    train_metrics: MetricsTracker,
    student: PreTrainedPolicy,
    teacher: PreTrainedPolicy | None,
    student_batch: dict,
    teacher_batch: dict | None,
    adapters: DistillAdapters,
    optimizer: Optimizer,
    grad_clip_norm: float,
    step: int,
    cfg: DistillTrainPipelineConfig,
    accelerator: Accelerator,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict]:
    """
    执行一步蒸馏前向+反向+优化（梯度累积由 accelerator.accumulate 管理）。

    Returns:
        更新后的 MetricsTracker，以及包含各分项 loss 的 output_dict。
    """
    t0 = time.perf_counter()
    dc = cfg.distill
    in_warmup = step < dc.warmup_steps
    log = {}

    student.train()

    student_vision_module = None
    student_model = student.module if hasattr(student, "module") else student
    if dc.feature_distill:
        student_vision_module = student_model.model.vlm_with_expert.vlm.model.connector

    with accelerator.accumulate(student, adapters):
        with MultiHookManager({"student_vision": student_vision_module} if student_vision_module else {}) as s_hooks:
            # ── 1. 学生任务损失（同时捕获 student features）──────────────────────
            with accelerator.autocast():
                task_loss, student_out = student.forward(student_batch)
                task_loss = task_loss.float()

            # ── 2. 蒸馏损失 ─────────────────────────────────────────────────
            distill_loss = torch.zeros(1, device=accelerator.device)

            if not in_warmup and (dc.feature_distill or dc.logit_distill):
                t_vision = None
                t_last = None
                t_action = None

                if teacher is not None and teacher_batch is not None:
                    t_hook_modules = {}
                    if dc.feature_distill:
                        t_hook_modules = {
                            "teacher_last": teacher.model.transformer.blocks[-1],
                        }
                    with MultiHookManager(t_hook_modules) as t_hooks, \
                         torch.no_grad(), \
                         torch.amp.autocast("cuda", dtype=torch.float16):
                        t_inputs = teacher._build_model_inputs(teacher_batch)
                        enc = teacher.model.forward_vlm(
                            input_ids=t_inputs["input_ids"],
                            pixel_values=t_inputs["image_input"],
                            image_mask=t_inputs["image_mask"],
                        )
                        if dc.feature_distill:
                            t_vision = enc["vlm_features"].float()
                        if dc.logit_distill:
                            t_action = teacher.model.generate_actions(
                                input_ids=t_inputs["input_ids"],
                                image_input=t_inputs["image_input"],
                                image_mask=t_inputs["image_mask"],
                                domain_id=t_inputs["domain_id"],
                                proprio=t_inputs["proprio"],
                                steps=teacher.config.num_denoising_steps,
                            ).float()
                        if dc.feature_distill:
                            t_last = t_hooks["teacher_last"].output.float()

                if dc.feature_distill:
                    dev = accelerator.device
                    vlm_with_expert = student_model.model.vlm_with_expert
                    se_cached = vlm_with_expert._last_expert_output
                    sv = s_hooks["student_vision"].output.float()
                    if step == 0 and accelerator.is_main_process:
                        logging.info(
                            f"[DEBUG] feature shapes: "
                            f"student_vision={tuple(sv.shape)} teacher_vision={tuple(t_vision.shape) if t_vision is not None else None} "
                            f"student_expert={tuple(se_cached.shape) if se_cached is not None else None} "
                            f"teacher_last={tuple(t_last.shape) if t_last is not None else None}"
                        )
                    if t_vision is None or se_cached is None:
                        logging.warning("feature_distill=True 但 t_vision 或 se_cached 为 None，跳过特征蒸馏本步")
                    else:
                        sv = align_seq_len(sv, t_vision)
                        sv_proj = adapters.adapt_vision(sv)
                        loss_vis = F.mse_loss(sv_proj, t_vision.detach())
                        distill_loss = distill_loss + dc.alpha_feature * loss_vis
                        log["loss_vision_feat"] = loss_vis.item()

                        se = se_cached.to(dev).float()
                        se = align_seq_len(se, t_last)
                        se_proj = adapters.adapt_expert(se)
                        loss_exp = F.mse_loss(se_proj, t_last.detach())
                        distill_loss = distill_loss + dc.alpha_feature * loss_exp
                        log["loss_expert_feat"] = loss_exp.item()

                if dc.logit_distill:
                    with torch.no_grad(), accelerator.autocast():
                        images, img_masks = student_model.prepare_images(student_batch)
                        state = student_model.prepare_state(student_batch)
                        lang_tokens = student_batch[f"{OBS_LANGUAGE_TOKENS}"]
                        lang_masks = student_batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
                        student_pred_action = student_model.model.sample_actions(
                            images, img_masks, lang_tokens, lang_masks, state
                        ).detach()
                        student_pred_action = student_pred_action[..., : dc.student_action_dim]
                        if step == 0 and accelerator.is_main_process:
                            logging.info(
                                f"[DEBUG] action shapes: "
                                f"student_action={tuple(student_pred_action.shape)} teacher_action={tuple(t_action.shape) if t_action is not None else None}"
                            )

                    s_act = student_pred_action.float()
                    min_T = min(s_act.shape[1], t_action.shape[1])
                    s_act = s_act[:, :min_T, :]
                    t_act = t_action[:, :min_T, : dc.teacher_action_dim]
                    s_act_proj = adapters.adapt_action(s_act)
                    T = dc.temperature
                    loss_kl = F.kl_div(
                        F.log_softmax(s_act_proj / T, dim=-1),
                        F.softmax(t_act.detach() / T, dim=-1),
                        reduction="batchmean",
                    ) * (T ** 2)
                    distill_loss = distill_loss + dc.alpha_logit * loss_kl
                    log["loss_kl_action"] = loss_kl.item()

            # ── 3. 总损失 ────────────────────────────────────────────────────
            total_loss = dc.alpha_task * task_loss + dc.alpha_distill * distill_loss

            # ── 4. 反向 ─────────────────────────────────────────────────────
            accelerator.backward(total_loss)

            # 手动同步 adapter 梯度（adapters 未包装 DDP，梯度不会自动 all-reduce）
            if accelerator.sync_gradients and accelerator.num_processes > 1:
                for p in adapters.parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

            # ── 5. 梯度裁剪 + 优化器步进 ──────────────────────────────────
            if grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(
                    list(student.parameters()) + list(adapters.parameters()),
                    grad_clip_norm,
                )
            else:
                grad_norm = torch.tensor(0.0)

            optimizer.step()
            optimizer.zero_grad()

    if lr_scheduler is not None and accelerator.sync_gradients:
        lr_scheduler.step()

    log.update({
        "loss_task": task_loss.item(),
        "loss_distill": distill_loss.item(),
        "in_warmup": float(in_warmup),
    })

    train_metrics.loss = total_loss.item()
    train_metrics.grad_norm = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - t0
    return train_metrics, log


# =============================================================================
# 主训练函数（对齐 lerobot_train.train）
# =============================================================================

@parser.wrap()
def train_distill(cfg: DistillTrainPipelineConfig, accelerator: Accelerator | None = None):
    """蒸馏训练主函数，结构对齐 lerobot_train.train。"""
    cfg.validate()

    if accelerator is None:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=cfg.grad_accum_steps,
        )

    init_logging(accelerator=accelerator)
    is_main = accelerator.is_main_process

    if is_main:
        logging.info(pformat(dataclasses.asdict(cfg)))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    if cfg.policy is None:
        cfg.policy = PreTrainedConfig.from_pretrained(cfg.student_path)
        cfg.policy.pretrained_path = Path(cfg.student_path)
        if is_main:
            logging.info(f"Loaded policy config from student_path: {cfg.student_path}")

    # WandB（仅主进程）
    if cfg.wandb.enable and cfg.wandb.project and is_main and WandBLogger is not None and cfg.policy is not None:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main:
            logging.info(colored("Logs 仅保存本地。", "yellow", attrs=["bold"]))

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── 数据集 ───────────────────────────────────────────────────────────────
    if is_main:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not is_main:
        dataset = make_dataset(cfg)

    # ── 学生模型 ─────────────────────────────────────────────────────────────
    if is_main:
        logging.info("Loading student policy")
    student = load_student(cfg)

    # 预处理器
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=student.config,
        pretrained_path=Path(cfg.student_path),
        dataset_stats=dataset.meta.stats,
    )

    # ── 教师模型（各 rank 本地加载）────────────────────────────────────────────
    accelerator.wait_for_everyone()
    teacher = load_teacher(cfg, accelerator)

    # 教师预处理器（各 rank 本地创建）
    teacher_preprocessor = None
    if teacher is not None:
        teacher_preprocessor, _ = make_pre_post_processors(
            policy_cfg=teacher.config,
            pretrained_path=Path(cfg.distill.teacher_path),
            dataset_stats=dataset.meta.stats,
        )

    # ── 适配层 ───────────────────────────────────────────────────────────────
    dc = cfg.distill
    adapters = DistillAdapters(
        student_vision_dim=dc.student_vision_dim,
        teacher_vision_dim=dc.teacher_vision_dim,
        student_expert_dim=dc.student_expert_dim,
        teacher_expert_dim=dc.teacher_expert_dim,
        student_action_dim=dc.student_action_dim,
        teacher_action_dim=dc.teacher_action_dim,
        enable_feature_distill=dc.feature_distill,
        enable_logit_distill=dc.logit_distill,
    ).to(device)

    # ── 优化器 & 调度器 ──────────────────────────────────────────────────────
    if is_main:
        logging.info("Creating optimizer and scheduler")

    all_params = [
        {"params": [p for p in student.parameters() if p.requires_grad]},
        {"params": list(adapters.parameters())},
    ]

    # 使用 student policy 自带的 training preset（若存在）
    if cfg.use_policy_training_preset and hasattr(student, "get_optimizer_preset"):
        from lerobot.optim.factory import make_optimizer_and_scheduler
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, student)
        # 追加 adapters 参数组
        for p in adapters.parameters():
            optimizer.add_param_group({"params": p})
    else:
        import math
        from torch.optim.lr_scheduler import LambdaLR
        opt_cfg = cfg.optimizer
        lr = opt_cfg.lr if opt_cfg else 1e-4
        wd = opt_cfg.weight_decay if opt_cfg else 1e-6
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=wd)

        warmup = cfg.scheduler.warmup_steps if cfg.scheduler else 1000
        total = cfg.steps

        def _lr_lambda(s):
            if s < warmup:
                return s / max(1, warmup)
            p = (s - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * p))

        lr_scheduler = LambdaLR(optimizer, _lr_lambda)

    # ── 断点恢复 ─────────────────────────────────────────────────────────────
    step = 0
    if cfg.resume and cfg.checkpoint_path is not None:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )
        adapters_ckpt = Path(cfg.checkpoint_path) / "adapters.pt"
        if adapters_ckpt.exists():
            adapters.load_state_dict(torch.load(adapters_ckpt, map_location="cpu"))
            logging.info(colored(f"已从 {adapters_ckpt} 恢复 adapters 权重", "green"))
        else:
            logging.warning(f"resume=True 但未找到 {adapters_ckpt}，adapters 将从头初始化")

    num_learnable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in student.parameters())
    if is_main:
        logging.info(colored(f"Output dir: {cfg.output_dir}", "yellow", attrs=["bold"]))
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        effective_bs = cfg.batch_size * accelerator.num_processes * cfg.grad_accum_steps
        logging.info(f"Effective batch size: {cfg.batch_size} x {accelerator.num_processes} x {cfg.grad_accum_steps} = {effective_bs}")
        logging.info(f"{num_learnable=} ({format_big_number(num_learnable)})")
        logging.info(f"{num_total=} ({format_big_number(num_total)})")

    # ── DataLoader ───────────────────────────────────────────────────────────
    if hasattr(student.config, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=student.config.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # ── Accelerator prepare ──────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    student, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        student, optimizer, dataloader, lr_scheduler
    )
    # Do NOT wrap adapters in DDP — its parameters are already in the optimizer
    # and DDP would shadow the custom methods (adapt_vision, adapt_expert, etc.).
    dl_iter = cycle(dataloader)

    # ── 指标跟踪 ─────────────────────────────────────────────────────────────
    train_metrics = {
        "loss": AverageMeter("loss", ":.4f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Distill Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            colored(
                f"开始蒸馏训练，共 {cfg.steps} 步，"
                f"feature_distill={dc.feature_distill}，"
                f"logit_distill={dc.logit_distill}",
                "green",
                attrs=["bold"],
            )
        )

    # ── 训练主循环 ───────────────────────────────────────────────────────────
    for _ in range(step, cfg.steps):
        t_data = time.perf_counter()
        raw_batch = next(dl_iter)
        student_batch = preprocessor(raw_batch)
        teacher_batch = teacher_preprocessor(raw_batch) if teacher_preprocessor else None
        train_tracker.dataloading_s = time.perf_counter() - t_data

        train_tracker, output_dict = update_distill(
            train_metrics=train_tracker,
            student=student,
            teacher=teacher,
            student_batch=student_batch,
            teacher_batch=teacher_batch,
            adapters=adapters,
            optimizer=optimizer,
            grad_clip_norm=cfg.optimizer.grad_clip_norm if cfg.optimizer else 10.0,
            step=step,
            cfg=cfg,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        if is_main:
            progbar.update(1)
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        # 日志
        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log = train_tracker.to_dict()
                if output_dict:
                    wandb_log.update({f"distill/{k}": v for k, v in output_dict.items()})
                wandb_logger.log_dict(wandb_log, step)
            train_tracker.reset_averages()

        # 保存检查点
        if cfg.save_checkpoint and is_saving_step:
            if is_main:
                logging.info(f"Checkpoint after step {step}")
                ckpt_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=ckpt_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(student),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                # 额外保存 adapters 权重（与学生 checkpoint 同目录）
                adapters_path = ckpt_dir / "adapters.pt"
                torch.save(accelerator.unwrap_model(adapters).state_dict(), adapters_path)
                update_last_checkpoint(ckpt_dir)
                if wandb_logger:
                    wandb_logger.log_policy(ckpt_dir)
            accelerator.wait_for_everyone()

    # ── 收尾 ─────────────────────────────────────────────────────────────────
    if is_main:
        progbar.close()
        logging.info("蒸馏训练完成。")

    accelerator.wait_for_everyone()
    accelerator.end_training()


# =============================================================================
# 入口
# =============================================================================

def main():
    register_third_party_plugins()
    train_distill()


if __name__ == "__main__":
    main()
