#!/usr/bin/env python
"""
VLA 蒸馏训练脚本（src/distill.py）

结构完全对齐 lerobot/scripts/lerobot_train.py：
- 配置：DistillTrainPipelineConfig 继承 TrainPipelineConfig + @parser.wrap()
- 更新函数：update_distill 对应 update_policy
- 主函数：train_distill 对应 train，结构、命名与 lerobot_train.py 一致
- 工具：MetricsTracker / AverageMeter / WandBLogger / save_checkpoint

教师模型（XVLA）：
  - 各 rank 本地加载，fp16 冻结推理，部署在对应 rank 的 accelerator.device
  - 中间特征通过 accelerator.broadcast 同步到所有 DDP 进程

学生模型（SmolVLA）：
  - accelerator.prepare 封装 DDP + AMP
  - 与 DistillAdapters 联合优化
"""

import dataclasses
import logging
import math
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
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
try:
    from lerobot.rl.wandb_utils import WandBLogger
except ImportError:
    try:
        from lerobot.common.wandb_utils import WandBLogger
    except ImportError:
        WandBLogger = None
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
try:
    from lerobot.utils.train_utils import (
        get_step_checkpoint_dir,
        get_step_identifier,
        load_training_state,
        save_checkpoint,
        update_last_checkpoint,
    )
except ImportError:
    from lerobot.common.train_utils import (
        get_step_checkpoint_dir,
        get_step_identifier,
        load_training_state,
        save_checkpoint,
        update_last_checkpoint,
    )
try:
    from lerobot.datasets.utils import cycle
except ImportError:
    from lerobot.utils.utils import cycle
from lerobot.utils.utils import (
    format_big_number,
    init_logging,
    inside_slurm,
)

# 自定义蒸馏模块
from adapters import DistillAdapters, align_seq_len
from hooks import HookSpec, MultiHookManager
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.constants import CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK, PRETRAINED_MODEL_DIR


# =============================================================================
# 配置扩展
# =============================================================================

@dataclass
class DistillConfig:
    """蒸馏超参，作为 DistillTrainPipelineConfig 的子字段。"""
    # 损失权重
    alpha_task: float = 1.0
    alpha_distill: float = 0.2
    alpha_feature: float = 0.4  # 兼容旧配置；新配置请优先使用下方细分权重
    alpha_vision_feature: float | None = None
    alpha_expert_feature: float | None = None
    alpha_logit: float = 0.0
    # 蒸馏开关
    feature_distill: bool = True
    vision_feature_distill: bool = True
    expert_feature_distill: bool = False
    logit_distill: bool = False
    # 预热：warmup 期间只用 task loss；单位为 optimizer step
    warmup_steps: int = 250
    distill_ramp_steps: int = 500
    # Safety gates for expensive distillation branches. The logged experiments
    # showed these branches can either train only adapters or OOM on 16GB GPUs.
    allow_frozen_vision_distill: bool = False
    allow_decision_path_distill: bool = False
    # Optional SmolVLA config overrides applied before constructing the student.
    # Use these when feature distillation should fine-tune the real VLM instead
    # of only training adapters around frozen features.
    student_train_expert_only: bool | None = None
    student_freeze_vision_encoder: bool | None = None
    student_use_cache: bool | None = False
    temperature: float = 2.0
    action_distill_loss: str = "mse"  # mse | smooth_l1 | kl
    action_distill_horizon: int = 1  # >0: distill first N chunk steps; <=0: use full common chunk
    # 教师模型
    teacher_path: str = ""
    teacher_dtype: str = "float16"
    teacher_transformer_last_layer_attr: str = "blocks"
    # 适配层维度
    student_vision_dim: int = 960
    teacher_vision_dim: int = 1024
    student_expert_dim: int = 480
    teacher_expert_dim: int = 1024
    student_action_dim: int = 7
    teacher_action_dim: int = 20
    action_align_mode: str = "teacher_abs20_to_student_rel7"

    @property
    def vision_loss_weight(self) -> float:
        return self.alpha_feature if self.alpha_vision_feature is None else self.alpha_vision_feature

    @property
    def expert_loss_weight(self) -> float:
        return self.alpha_feature if self.alpha_expert_feature is None else self.alpha_expert_feature

    @property
    def enable_vision_distill(self) -> bool:
        return self.feature_distill and self.vision_feature_distill

    @property
    def enable_expert_distill(self) -> bool:
        return self.feature_distill and self.expert_feature_distill


@dataclass
class DistillTrainPipelineConfig(TrainPipelineConfig):
    """在 LeRobot TrainPipelineConfig 基础上追加蒸馏配置。"""
    distill: DistillConfig = field(default_factory=DistillConfig)
    # 学生模型路径（同 policy.pretrained_path，单独保留以兼容现有 YAML）
    student_path: str = ""
    # 可选：显式指定恢复训练的 checkpoint 目录
    checkpoint_path: str | None = None
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


_WARNED_MESSAGES: set[str] = set()


def log_warning_once(key: str, message: str) -> None:
    if key in _WARNED_MESSAGES:
        return
    logging.warning(message)
    _WARNED_MESSAGES.add(key)


def compute_distill_scale(dc: DistillConfig, optimizer_step: int) -> float:
    """Return a [0, 1] multiplier for distillation based on optimizer steps."""
    if optimizer_step < dc.warmup_steps:
        return 0.0
    if dc.distill_ramp_steps <= 0:
        return 1.0
    post_warmup = optimizer_step - dc.warmup_steps + 1
    return min(1.0, post_warmup / dc.distill_ramp_steps)


def compute_total_optimizer_steps(total_micro_steps: int, grad_accum_steps: int) -> int:
    """Return how many optimizer updates a micro-step training run performs."""
    return math.ceil(total_micro_steps / max(1, grad_accum_steps))


def cosine_with_warmup_scale(current_step: int, warmup_steps: int, total_steps: int) -> float:
    """Optimizer-step based linear warmup + cosine decay multiplier."""
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1 + math.cos(math.pi * progress))


def compute_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = param.grad.detach().float().norm(2)
        total += float(param_norm.item() ** 2)
    return math.sqrt(total)


def set_policy_use_cache(policy: PreTrainedPolicy, cfg: TrainPipelineConfig, value: bool) -> list[tuple[Any, bool]]:
    """Temporarily set known SmolVLA cache flags and return restore state."""
    model = getattr(policy, "model", None)
    vlm_with_expert = getattr(model, "vlm_with_expert", None)
    modules = (
        cfg.policy,
        policy,
        getattr(policy, "config", None),
        model,
        getattr(model, "config", None),
        vlm_with_expert,
        getattr(vlm_with_expert, "config", None),
        getattr(vlm_with_expert, "vlm", None),
        getattr(getattr(vlm_with_expert, "vlm", None), "config", None),
        getattr(vlm_with_expert, "lm_expert", None),
        getattr(getattr(vlm_with_expert, "lm_expert", None), "config", None),
    )

    restore_state: list[tuple[Any, bool]] = []
    seen: set[int] = set()
    for module in modules:
        if module is None or id(module) in seen or not hasattr(module, "use_cache"):
            continue
        seen.add(id(module))
        restore_state.append((module, module.use_cache))
        module.use_cache = value
    return restore_state


def restore_policy_use_cache(restore_state: list[tuple[Any, bool]]) -> None:
    for module, value in restore_state:
        module.use_cache = value


def has_trainable_parameters(module: Any) -> bool:
    return module is not None and any(p.requires_grad for p in module.parameters())


def apply_student_policy_overrides(cfg: DistillTrainPipelineConfig) -> dict[str, Any]:
    """Apply distillation-time SmolVLA config overrides before model construction."""
    policy_cfg = cfg.policy
    if policy_cfg is None:
        return {}

    dc = cfg.distill
    requested = {
        "train_expert_only": dc.student_train_expert_only,
        "freeze_vision_encoder": dc.student_freeze_vision_encoder,
        "use_cache": dc.student_use_cache,
    }
    applied: dict[str, Any] = {}
    for name, value in requested.items():
        if value is None or not hasattr(policy_cfg, name):
            continue
        setattr(policy_cfg, name, value)
        applied[name] = value
    return applied


def configure_distill_branches_for_student(
    cfg: DistillTrainPipelineConfig,
    student: PreTrainedPolicy,
) -> dict[str, str]:
    """Disable or reject distillation branches that cannot affect the intended student path."""
    dc = cfg.distill
    changes: dict[str, str] = {}
    student_model = student.module if hasattr(student, "module") else student
    vlm_with_expert = getattr(getattr(student_model, "model", None), "vlm_with_expert", None)
    vlm = getattr(vlm_with_expert, "vlm", None)

    if dc.enable_vision_distill and not dc.allow_frozen_vision_distill and not has_trainable_parameters(vlm):
        wants_trainable_vlm = (
            dc.student_train_expert_only is False
            or dc.student_freeze_vision_encoder is False
        )
        if wants_trainable_vlm:
            raise RuntimeError(
                "vision_feature_distill=True but the student VLM has no trainable parameters. "
                "Set distill.student_train_expert_only=false and distill.student_freeze_vision_encoder=false "
                "before loading the student, or set distill.allow_frozen_vision_distill=true for adapter-only "
                "vision feature distillation."
            )
        dc.vision_feature_distill = False
        changes["vision_feature_distill"] = (
            "disabled because the SmolVLA VLM is frozen; the loss would mostly train the adapter."
        )

    if (dc.enable_expert_distill or dc.logit_distill) and not dc.allow_decision_path_distill:
        if dc.enable_expert_distill and dc.logit_distill:
            dc.expert_feature_distill = False
            changes["expert_feature_distill"] = (
                "disabled because expert+logit distillation shares the expensive sample_actions() decision path."
            )
        if dc.logit_distill:
            dc.logit_distill = False
            changes["logit_distill"] = (
                "disabled because it requires the expensive sample_actions() decision path."
            )

    return changes


def needs_teacher_forward(dc: DistillConfig) -> bool:
    return dc.enable_vision_distill or dc.enable_expert_distill or dc.logit_distill


def ensure_runtime_ready(cfg: DistillTrainPipelineConfig) -> None:
    """Fail fast on missing local resources or unavailable CUDA."""
    required_paths = {"student_path": Path(cfg.student_path)}
    dataset_root = getattr(cfg.dataset, "root", None)
    if dataset_root:
        required_paths["dataset.root"] = Path(dataset_root)

    for field_name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{field_name} 不存在: {path}。请确认本机模型/数据集路径配置正确。"
            )

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError(
            "未检测到可见 CUDA 设备，蒸馏训练不会回退到 CPU。"
            "请确认 NVIDIA driver、容器 GPU 透传和当前会话的 CUDA 可见性。"
        )


# =============================================================================
# 模型加载
# =============================================================================

def load_teacher(cfg: DistillTrainPipelineConfig, accelerator: Accelerator) -> PreTrainedPolicy | None:
    """加载教师模型（各 rank 本地加载），fp16 冻结部署在当前 rank device。"""
    dc = cfg.distill
    teacher_path = Path(dc.teacher_path)
    if not teacher_path.exists():
        raise FileNotFoundError(
            f"distill.teacher_path 不存在: {teacher_path}。启用蒸馏分支时必须配置教师模型路径。"
        )
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    teacher_dtype = dtype_map[dc.teacher_dtype.lower()]

    logging.info(colored(f"加载教师模型: {dc.teacher_path} -> {accelerator.device}", "blue"))
    teacher_cls = get_policy_class("xvla")
    teacher = teacher_cls.from_pretrained(dc.teacher_path)
    teacher = teacher.to(device=accelerator.device, dtype=teacher_dtype)
    # Distillation does not need decoding KV cache. Keeping it disabled avoids
    # extra attention memory usage.
    for mod in (teacher, getattr(teacher, "model", None)):
        if mod is None:
            continue
        cfg_obj = getattr(mod, "config", None)
        if cfg_obj is not None and hasattr(cfg_obj, "use_cache"):
            cfg_obj.use_cache = False
        if hasattr(mod, "use_cache"):
            setattr(mod, "use_cache", False)
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
    student = student_cls.from_pretrained(path, config=cfg.policy)

    vlm = getattr(student.model.vlm_with_expert, "vlm", None)
    lm_expert = getattr(student.model.vlm_with_expert, "lm_expert", None)
    if vlm is not None and hasattr(vlm, "gradient_checkpointing_enable"):
        vlm.gradient_checkpointing_enable()
        logging.info("学生 VLM 梯度检查点已启用")
    if lm_expert is not None and hasattr(lm_expert, "gradient_checkpointing_enable"):
        lm_expert.gradient_checkpointing_enable()
        logging.info("学生 lm_expert 梯度检查点已启用")

    # Gradient checkpointing should be paired with use_cache=False, otherwise
    # attention memory can spike in custom VLM blocks.
    for mod in (
        student,
        getattr(student, "model", None),
        getattr(student.model, "vlm_with_expert", None),
        vlm,
        lm_expert,
    ):
        if mod is None:
            continue
        cfg_obj = getattr(mod, "config", None)
        if cfg_obj is not None and hasattr(cfg_obj, "use_cache"):
            cfg_obj.use_cache = False
        if hasattr(mod, "use_cache"):
            setattr(mod, "use_cache", False)

    student.train()
    return student


def resolve_resume_checkpoint_path(cfg: DistillTrainPipelineConfig) -> Path | None:
    """Resolve the checkpoint root directory used for resume."""
    if not cfg.resume:
        return None

    resume_ckpt = cfg.checkpoint_path or str(Path(cfg.output_dir) / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK)
    resume_ckpt_path = Path(resume_ckpt)
    if not resume_ckpt_path.exists():
        raise FileNotFoundError(
            f"resume=True 但未找到恢复检查点: {resume_ckpt_path}。"
            f"可通过 --checkpoint_path 显式指定，或确认 {Path(cfg.output_dir) / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK} 存在。"
        )

    pretrained_dir = resume_ckpt_path / PRETRAINED_MODEL_DIR
    if not pretrained_dir.is_dir():
        if resume_ckpt_path.name == PRETRAINED_MODEL_DIR:
            raise NotADirectoryError(
                f"`checkpoint_path` 应指向 checkpoint 根目录，而不是 `{PRETRAINED_MODEL_DIR}` 子目录: {resume_ckpt_path}"
            )
        raise NotADirectoryError(
            f"恢复检查点缺少 `{PRETRAINED_MODEL_DIR}` 目录: {pretrained_dir}"
        )

    return resume_ckpt_path


def extract_teacher_feature_targets(
    teacher: PreTrainedPolicy,
    teacher_batch: dict,
    cfg: DistillTrainPipelineConfig,
) -> torch.Tensor:
    """Run XVLA encoder once to capture stable feature distillation targets."""

    t_inputs = teacher._build_model_inputs(teacher_batch)
    teacher_model = teacher.model

    enc = teacher_model.forward_vlm(
        input_ids=t_inputs["input_ids"],
        pixel_values=t_inputs["image_input"],
        image_mask=t_inputs["image_mask"],
    )
    return enc["vlm_features"].float()


def format_distill_stats(
    *,
    micro_step: int,
    optimizer_step: int,
    raw_task_loss: float,
    task_loss_avg: float,
    task_loss_ema: float,
    output_dict: dict,
) -> str:
    parts = [
        "distill stats:",
        f"micro_step={micro_step}",
        f"optimizer_step={optimizer_step}",
        f"loss_task_raw={raw_task_loss:.4f}",
        f"loss_task_avg={task_loss_avg:.4f}",
        f"loss_task_ema={task_loss_ema:.4f}",
        f"loss_distill={output_dict.get('loss_distill', 0.0):.4f}",
        f"distill_scale={output_dict.get('distill_scale', 0.0):.3f}",
    ]
    for key in ("loss_vision_feat", "loss_expert_feat", "loss_action_distill"):
        if key in output_dict:
            parts.append(f"{key}={output_dict[key]:.4f}")
    return " ".join(parts)


def update_best_checkpoint_link(checkpoint_dir: Path, output_dir: str | Path) -> bool:
    """Point checkpoints/best to checkpoint_dir, without deleting real directories."""
    checkpoints_dir = Path(output_dir) / CHECKPOINTS_DIR
    best_link = checkpoints_dir / "best"
    checkpoint_dir = Path(checkpoint_dir)
    if best_link.exists() or best_link.is_symlink():
        if not best_link.is_symlink():
            logging.warning("Cannot update best checkpoint link because path exists and is not a symlink: %s", best_link)
            return False
        best_link.unlink()
    best_link.symlink_to(checkpoint_dir.name)
    return True


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
    micro_step: int,
    optimizer_step: int,
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
    in_warmup = optimizer_step < dc.warmup_steps
    distill_scale = compute_distill_scale(dc, optimizer_step)
    enable_vision_distill = dc.enable_vision_distill
    enable_expert_distill = dc.enable_expert_distill
    enable_logit_distill = dc.logit_distill
    need_teacher_action_path = enable_expert_distill or enable_logit_distill
    need_student_action_path = enable_logit_distill
    log = {}
    grad_norm = torch.tensor(0.0, device=accelerator.device)
    student_grad_norm = 0.0

    student.train()

    student_vision_module = None
    student_expert_module = None
    student_model = student.module if hasattr(student, "module") else student
    if enable_vision_distill:
        student_vision_module = student_model.model.vlm_with_expert.vlm.model.connector
    if enable_expert_distill:
        student_expert_module = student_model.model.action_out_proj

    with accelerator.accumulate(student, adapters):
        hook_specs = {}
        if student_vision_module is not None:
            hook_specs["student_vision"] = HookSpec(
                module=student_vision_module,
                detach=False,
                clone=False,
            )
        if student_expert_module is not None:
            hook_specs["student_expert"] = HookSpec(
                module=student_expert_module,
                detach=False,
                clone=False,
                capture="input",
            )
        with MultiHookManager(hook_specs) as s_hooks:
            # ── 1. 学生任务损失（同时捕获 student features）──────────────────────
            with accelerator.autocast():
                task_loss, _ = student.forward(student_batch)
                task_loss = task_loss.float()

            # ── 2. 蒸馏损失 ─────────────────────────────────────────────────
            distill_loss = torch.zeros((), device=accelerator.device)

            if distill_scale > 0 and (enable_vision_distill or enable_expert_distill or enable_logit_distill):
                t_vision = None
                t_action = None
                t_expert = None
                s_expert = None
                student_pred_action = None

                if teacher is not None and teacher_batch is not None:
                    teacher_dtype = getattr(teacher.model, "_get_target_dtype", lambda: torch.float16)()
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=teacher_dtype):
                        if enable_vision_distill:
                            t_vision = extract_teacher_feature_targets(teacher, teacher_batch, cfg)
                        if need_teacher_action_path:
                            t_inputs = teacher._build_model_inputs(teacher_batch)
                            teacher_hook_specs = (
                                {
                                    "teacher_expert": HookSpec(
                                        module=teacher.model.transformer.norm,
                                        detach=True,
                                        clone=True,
                                    )
                                }
                                if enable_expert_distill
                                else {}
                            )
                            with MultiHookManager(teacher_hook_specs) as t_hooks:
                                teacher_actions = teacher.model.generate_actions(
                                    input_ids=t_inputs["input_ids"],
                                    image_input=t_inputs["image_input"],
                                    image_mask=t_inputs["image_mask"],
                                    domain_id=t_inputs["domain_id"],
                                    proprio=t_inputs["proprio"],
                                    steps=teacher.config.num_denoising_steps,
                                )
                                if enable_logit_distill:
                                    t_action = teacher_actions.float()
                                if enable_expert_distill:
                                    t_expert_raw = t_hooks["teacher_expert"].output
                                    t_expert = t_expert_raw.float() if t_expert_raw is not None else None

                if enable_expert_distill:
                    s_expert_raw = s_hooks["student_expert"].output if student_expert_module else None
                    s_expert = s_expert_raw.float() if s_expert_raw is not None else None

                if need_student_action_path:
                    action_model = student_model.model
                    with accelerator.autocast():
                        images, img_masks = student_model.prepare_images(student_batch)
                        state = student_model.prepare_state(student_batch)
                        lang_tokens = student_batch[f"{OBS_LANGUAGE_TOKENS}"]
                        lang_masks = student_batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
                        cache_state_stack: list[tuple[object, str, object]] = []
                        model_cfg = getattr(action_model, "config", None)
                        if model_cfg is not None and hasattr(model_cfg, "use_cache"):
                            cache_state_stack.append((model_cfg, "use_cache", model_cfg.use_cache))
                            model_cfg.use_cache = True
                        if hasattr(action_model, "use_cache"):
                            cache_state_stack.append((action_model, "use_cache", action_model.use_cache))
                            action_model.use_cache = True
                        if cache_state_stack:
                            log_warning_once(
                                "logit_distill_temp_enable_cache",
                                "decision-path distillation 调用 sample_actions() 时会临时启用 use_cache，调用后自动恢复。",
                            )
                        try:
                            student_pred_action = action_model.sample_actions(
                                images, img_masks, lang_tokens, lang_masks, state
                            )
                        finally:
                            for target_obj, attr_name, old_value in reversed(cache_state_stack):
                                setattr(target_obj, attr_name, old_value)
                    student_pred_action = student_pred_action[..., : dc.student_action_dim]

                if enable_vision_distill or enable_expert_distill:
                    sv_raw = s_hooks["student_vision"].output if student_vision_module else None
                    sv = sv_raw.float() if sv_raw is not None else None

                    if micro_step == 0 and accelerator.is_main_process:
                        logging.info(
                            f"[DEBUG] feature shapes: "
                            f"student_vision={tuple(sv.shape) if sv is not None else None} "
                            f"teacher_vision={tuple(t_vision.shape) if t_vision is not None else None} "
                            f"student_expert={tuple(s_expert.shape) if s_expert is not None else None} "
                            f"teacher_expert={tuple(t_expert.shape) if t_expert is not None else None}"
                        )
                    if enable_vision_distill:
                        if sv is None or t_vision is None:
                            log_warning_once(
                                "vision_distill_missing_features",
                                "vision_feature_distill=True 但 student_vision 或 teacher_vision 为 None，跳过 vision 特征蒸馏。",
                            )
                        else:
                            sv = align_seq_len(sv, t_vision)
                            sv_proj = adapters.adapt_vision(sv)
                            loss_vis = F.mse_loss(sv_proj, t_vision.detach())
                            distill_loss = distill_loss + dc.vision_loss_weight * loss_vis
                            log["loss_vision_feat"] = loss_vis.item()

                    if enable_expert_distill:
                        if t_expert is None:
                            log_warning_once(
                                "expert_distill_missing_teacher",
                                "expert_feature_distill=True 但 teacher expert target 为 None，跳过 expert 特征蒸馏。",
                            )
                        elif s_expert is None:
                            log_warning_once(
                                "expert_distill_missing_student",
                                "expert_feature_distill=True 但 student expert source 为 None，跳过 expert 特征蒸馏。",
                            )
                        elif student_pred_action is not None and not student_pred_action.requires_grad:
                            log_warning_once(
                                "decision_path_nondiff_student",
                                "student 决策路径当前不可微，已跳过 decision-path 的 expert/logit 蒸馏；如需启用，请切换到可微动作输出接口。",
                            )
                        else:
                            s_expert = align_seq_len(s_expert, t_expert)
                            s_expert_proj = adapters.adapt_expert(s_expert)
                            loss_expert = F.mse_loss(s_expert_proj, t_expert.detach())
                            distill_loss = distill_loss + dc.expert_loss_weight * loss_expert
                            log["loss_expert_feat"] = loss_expert.item()

                if (need_teacher_action_path or need_student_action_path) and micro_step == 0 and accelerator.is_main_process:
                    logging.info(
                        f"[DEBUG] action shapes: "
                        f"student_action={tuple(student_pred_action.shape) if student_pred_action is not None else None} "
                        f"teacher_action={tuple(t_action.shape) if t_action is not None else None}"
                    )

                if enable_logit_distill:
                    if t_action is None:
                        log_warning_once(
                            "logit_distill_missing_teacher",
                            "logit_distill=True 但 teacher_action 为 None，跳过 logit 蒸馏。",
                        )
                    elif student_pred_action is None:
                        log_warning_once(
                            "logit_distill_missing_student_action",
                            "logit_distill=True 但 student sample_actions() 未返回动作，跳过 logit 蒸馏。",
                        )
                    elif not student_pred_action.requires_grad:
                        log_warning_once(
                            "decision_path_nondiff_student",
                            "student 决策路径当前不可微，已跳过 decision-path 的 expert/logit 蒸馏；如需启用，请切换到可微动作输出接口。",
                        )
                    else:
                        s_act = student_pred_action.float()
                        min_T = min(s_act.shape[1], t_action.shape[1])
                        if dc.action_distill_horizon > 0:
                            min_T = min(min_T, dc.action_distill_horizon)
                        s_act = s_act[:, :min_T, :]
                        t_act = t_action[:, :min_T, : dc.teacher_action_dim].float()
                        s_act_aligned = adapters.adapt_student_action(s_act)
                        teacher_state = teacher_batch.get(OBS_STATE) if teacher_batch is not None else None
                        t_act_aligned = adapters.adapt_teacher_action(
                            t_act,
                            teacher_state=teacher_state,
                        )
                        loss_type = dc.action_distill_loss.strip().lower()
                        if loss_type == "mse":
                            log_warning_once(
                                "action_distill_temperature_ignored",
                                "action_distill_loss!=kl 时 temperature 参数会被忽略。",
                            )
                            loss_action = F.mse_loss(s_act_aligned, t_act_aligned.detach())
                        elif loss_type == "smooth_l1":
                            log_warning_once(
                                "action_distill_temperature_ignored",
                                "action_distill_loss!=kl 时 temperature 参数会被忽略。",
                            )
                            loss_action = F.smooth_l1_loss(s_act_aligned, t_act_aligned.detach())
                        elif loss_type == "kl":
                            t = dc.temperature
                            loss_action = F.kl_div(
                                F.log_softmax(s_act_aligned / t, dim=-1),
                                F.softmax(t_act_aligned.detach() / t, dim=-1),
                                reduction="batchmean",
                            ) * (t ** 2)
                        else:
                            raise ValueError(
                                "Unsupported distill.action_distill_loss="
                                f"{dc.action_distill_loss}. Expected one of: mse, smooth_l1, kl."
                            )

                        distill_loss = distill_loss + dc.alpha_logit * loss_action
                        log["loss_action_distill"] = loss_action.item()
                        log["action_distill_steps"] = float(min_T)
                        # 0: mse, 1: smooth_l1, 2: kl
                        log["action_distill_loss_type"] = {"mse": 0.0, "smooth_l1": 1.0, "kl": 2.0}[loss_type]

            # ── 3. 总损失 ────────────────────────────────────────────────────
            total_loss = dc.alpha_task * task_loss + (dc.alpha_distill * distill_scale) * distill_loss

            # ── 4. 反向 ─────────────────────────────────────────────────────
            accelerator.backward(total_loss)

            # 手动同步 adapter 梯度（adapters 未包装 DDP，梯度不会自动 all-reduce）
            if accelerator.sync_gradients and accelerator.num_processes > 1:
                for p in adapters.parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

            # ── 5. 梯度裁剪 + 优化器步进（仅在累积边界执行）──────────────────
            if accelerator.sync_gradients:
                student_grad_norm = compute_grad_norm([p for p in student.parameters() if p.requires_grad])
                if grad_clip_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(
                        list(student.parameters()) + list(adapters.parameters()),
                        grad_clip_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # 及时释放该步缓存的 expert 输出引用，降低显存滞留风险。
            if hasattr(student_model.model, "vlm_with_expert"):
                student_model.model.vlm_with_expert._last_expert_output = None

    if lr_scheduler is not None and accelerator.sync_gradients:
        lr_scheduler.step()

    log.update({
        "loss_task": task_loss.item(),
        "loss_distill": distill_loss.item(),
        "distill_scale": distill_scale,
        "in_warmup": float(in_warmup),
        "micro_step": float(micro_step + 1),
        "optimizer_step": float(optimizer_step + int(accelerator.sync_gradients)),
    })
    if accelerator.sync_gradients:
        log["optimizer_step_applied"] = 1.0
        log["student_grad_norm"] = student_grad_norm

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
    ensure_runtime_ready(cfg)

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

    resume_ckpt_path = resolve_resume_checkpoint_path(cfg)
    student_pretrained_path = resume_ckpt_path / PRETRAINED_MODEL_DIR if resume_ckpt_path else Path(cfg.student_path)

    if is_main:
        logging.info(
            f"Resume mode: {cfg.resume}, "
            f"resume_ckpt={resume_ckpt_path if resume_ckpt_path else 'None'}, "
            f"student_load_path={student_pretrained_path}"
        )

    if cfg.policy is None:
        cfg.policy = PreTrainedConfig.from_pretrained(student_pretrained_path)
        cfg.policy.pretrained_path = student_pretrained_path
        if is_main:
            source = "resume checkpoint" if resume_ckpt_path else "student_path"
            logging.info(f"Loaded policy config from {source}: {student_pretrained_path}")
    elif resume_ckpt_path:
        cfg.policy.pretrained_path = student_pretrained_path

    student_policy_overrides = apply_student_policy_overrides(cfg)
    if student_policy_overrides and is_main:
        logging.info("Applied student policy overrides: %s", student_policy_overrides)

    wandb_logger = None

    device = accelerator.device
    if device.type == "cuda":
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
    original_student_path = cfg.student_path
    cfg.student_path = str(student_pretrained_path)
    student = load_student(cfg)
    cfg.student_path = original_student_path

    # 预处理器
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=student.config,
        pretrained_path=student_pretrained_path,
        dataset_stats=dataset.meta.stats,
    )

    branch_changes = configure_distill_branches_for_student(cfg, student)
    if is_main:
        for branch, reason in branch_changes.items():
            logging.warning("distill.%s %s", branch, reason)

    # WandB（仅主进程）。Initialize after branch gating so W&B config reflects
    # the effective distillation setup, not just the requested YAML values.
    if cfg.wandb.enable and cfg.wandb.project and is_main and WandBLogger is not None and cfg.policy is not None:
        wandb_logger = WandBLogger(cfg)
    elif is_main:
        logging.info(colored("Logs 仅保存本地。", "yellow", attrs=["bold"]))

    # ── 教师模型（各 rank 本地加载）────────────────────────────────────────────
    accelerator.wait_for_everyone()
    teacher = load_teacher(cfg, accelerator) if needs_teacher_forward(cfg.distill) else None
    if teacher is None and is_main:
        logging.info("No active distillation branch needs teacher forward; skipping teacher load.")

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
    student_norm_map = getattr(student.config, "normalization_mapping", {}) or {}
    student_action_norm_mode = student_norm_map.get("ACTION", "IDENTITY")
    student_action_stats = dataset.meta.stats.get(ACTION) if getattr(dataset.meta, "stats", None) else None
    adapters = DistillAdapters(
        student_vision_dim=dc.student_vision_dim,
        teacher_vision_dim=dc.teacher_vision_dim,
        student_expert_dim=dc.student_expert_dim,
        teacher_expert_dim=dc.teacher_expert_dim,
        student_action_dim=dc.student_action_dim,
        teacher_action_dim=dc.teacher_action_dim,
        action_align_mode=dc.action_align_mode,
        student_action_stats=student_action_stats,
        student_action_norm_mode=student_action_norm_mode,
        enable_vision_distill=dc.enable_vision_distill,
        enable_expert_distill=dc.enable_expert_distill,
        enable_logit_distill=dc.logit_distill,
    ).to(device)

    # ── 优化器 & 调度器 ──────────────────────────────────────────────────────
    if is_main:
        logging.info("Creating optimizer and scheduler")

    student_params = [p for p in student.parameters() if p.requires_grad]
    adapter_params = list(adapters.parameters())
    all_params = [{"params": student_params}]
    if adapter_params:
        all_params.append({"params": adapter_params})

    # 使用 student policy 自带的 training preset（若存在）
    if cfg.use_policy_training_preset and hasattr(student, "get_optimizer_preset"):
        from lerobot.optim.factory import make_optimizer_and_scheduler
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, student)
        # 追加 adapters 参数组
        if adapter_params:
            optimizer.add_param_group({"params": adapter_params})
    else:
        from torch.optim.lr_scheduler import LambdaLR
        opt_cfg = cfg.optimizer
        lr = opt_cfg.lr if opt_cfg else 1e-4
        wd = opt_cfg.weight_decay if opt_cfg else 1e-6
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=wd)

        warmup = cfg.scheduler.warmup_steps if cfg.scheduler else 1000
        total = compute_total_optimizer_steps(cfg.steps, cfg.grad_accum_steps)

        def _lr_lambda(s):
            return cosine_with_warmup_scale(s, warmup, total)

        lr_scheduler = LambdaLR(optimizer, _lr_lambda)

    # ── 断点恢复 ─────────────────────────────────────────────────────────────
    step = 0
    if resume_ckpt_path is not None:
        step, optimizer, lr_scheduler = load_training_state(
            resume_ckpt_path, optimizer, lr_scheduler
        )
        adapters_ckpt = resume_ckpt_path / "adapters.pt"
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
        total_optimizer_steps = compute_total_optimizer_steps(cfg.steps, cfg.grad_accum_steps)
        logging.info(f"Effective batch size: {cfg.batch_size} x {accelerator.num_processes} x {cfg.grad_accum_steps} = {effective_bs}")
        logging.info(f"Approx optimizer steps: {total_optimizer_steps}")
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
    optimizer_step = step // max(1, cfg.grad_accum_steps)
    sync_step_count = optimizer_step
    task_loss_interval_sum = 0.0
    task_loss_interval_count = 0
    task_loss_ema: float | None = None
    task_loss_ema_beta = 0.98
    best_task_loss_ema = float("inf")

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
                f"开始蒸馏训练，共 {cfg.steps} 个 micro step，"
                f"vision_distill={dc.enable_vision_distill}，"
                f"expert_distill={dc.enable_expert_distill}，"
                f"logit_distill={dc.logit_distill}，"
                f"action_align_mode={dc.action_align_mode}，"
                f"action_distill_loss={dc.action_distill_loss}，"
                f"action_distill_horizon={dc.action_distill_horizon}",
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
            micro_step=step,
            optimizer_step=optimizer_step,
            cfg=cfg,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        raw_task_loss = output_dict["loss_task"]
        task_loss_interval_sum += raw_task_loss
        task_loss_interval_count += 1
        if task_loss_ema is None:
            task_loss_ema = raw_task_loss
        else:
            task_loss_ema = task_loss_ema_beta * task_loss_ema + (1 - task_loss_ema_beta) * raw_task_loss

        step += 1
        sync_applied = int(output_dict.get("optimizer_step_applied", 0.0))
        optimizer_step += sync_applied
        sync_step_count += sync_applied
        if is_main:
            progbar.update(1)
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        # 日志
        if is_log_step:
            logging.info(train_tracker)
            task_loss_avg = task_loss_interval_sum / max(1, task_loss_interval_count)
            logging.info(
                format_distill_stats(
                    micro_step=step,
                    optimizer_step=optimizer_step,
                    raw_task_loss=raw_task_loss,
                    task_loss_avg=task_loss_avg,
                    task_loss_ema=task_loss_ema,
                    output_dict=output_dict,
                )
            )
            if wandb_logger:
                wandb_log = train_tracker.to_dict()
                if output_dict:
                    wandb_log.update({f"distill/{k}": v for k, v in output_dict.items()})
                wandb_log.update({
                    "distill/loss_task_avg": task_loss_avg,
                    "distill/loss_task_ema": task_loss_ema,
                    "distill/micro_step": float(step),
                    "distill/optimizer_step": float(optimizer_step),
                    "distill/sync_step_count": float(sync_step_count),
                })
                wandb_logger.log_dict(wandb_log, step)
            train_tracker.reset_averages()
            task_loss_interval_sum = 0.0
            task_loss_interval_count = 0

        # 保存检查点
        if cfg.save_checkpoint and is_saving_step:
            if is_main:
                logging.info(f"Checkpoint after step {step}")
                ckpt_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                unwrapped_student = accelerator.unwrap_model(student)
                use_cache_restore_state = set_policy_use_cache(unwrapped_student, cfg, True)
                try:
                    save_checkpoint(
                        checkpoint_dir=ckpt_dir,
                        step=step,
                        cfg=cfg,
                        policy=unwrapped_student,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                finally:
                    restore_policy_use_cache(use_cache_restore_state)
                # 额外保存 adapters 权重（与学生 checkpoint 同目录）
                adapters_path = ckpt_dir / "adapters.pt"
                torch.save(accelerator.unwrap_model(adapters).state_dict(), adapters_path)
                update_last_checkpoint(ckpt_dir)
                if task_loss_ema is not None and task_loss_ema < best_task_loss_ema:
                    best_task_loss_ema = task_loss_ema
                    if update_best_checkpoint_link(ckpt_dir, cfg.output_dir):
                        logging.info(
                            "Updated best checkpoint after step %s with loss_task_ema=%.4f",
                            step,
                            best_task_loss_ema,
                        )
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
