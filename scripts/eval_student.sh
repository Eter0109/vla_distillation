#!/usr/bin/env bash
# 学生模型评测脚本（LIBERO 任务）
# 用法：bash scripts/eval_student.sh [可选参数]
#
# 示例：
#   # 自动选取最新检查点
#   bash scripts/eval_student.sh
#
#   # 指定检查点目录
#   bash scripts/eval_student.sh --ckpt outputs/checkpoints/step_0030000
#
#   # 指定 GPU 和 episode 数量
#   bash scripts/eval_student.sh --ckpt outputs/checkpoints/best --device cuda:2 --n_episodes 50
#
#   # 指定 LIBERO 任务类型
#   bash scripts/eval_student.sh --task libero_spatial --n_episodes 100

set -euo pipefail

# =============================================================================
# 环境激活
# =============================================================================
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

# =============================================================================
# 路径配置
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
LEROBOT_ROOT="${LEROBOT_ROOT:-${PROJECT_ROOT}/../lerobot/src}"

export PYTHONPATH="${LEROBOT_ROOT}:${SRC_DIR}:${PYTHONPATH:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
if [[ "${USE_HF_MIRROR}" == "1" ]]; then
    unset no_proxy https_proxy NO_PROXY HTTPS_PROXY HTTP_PROXY http_proxy ALL_PROXY all_proxy
    export HF_ENDPOINT
fi

# =============================================================================
# 默认参数
# =============================================================================
CKPT_PATH=""               # 空则自动寻找最新检查点
EVAL_DEVICE="cuda:2"
N_EPISODES=50
BATCH_SIZE=10
LIBERO_TASK="libero_spatial"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/eval"
DISTILL_OUTPUT_DIR="${DISTILL_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/task_only_gbs256_1gpu}"
if [[ "${DISTILL_OUTPUT_DIR}" != /* ]]; then
    DISTILL_OUTPUT_DIR="${PROJECT_ROOT}/${DISTILL_OUTPUT_DIR}"
fi

# =============================================================================
# 解析命令行参数
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            CKPT_PATH="$2"; shift 2 ;;
        --device)
            EVAL_DEVICE="$2"; shift 2 ;;
        --n_episodes)
            N_EPISODES="$2"; shift 2 ;;
        --batch_size)
            BATCH_SIZE="$2"; shift 2 ;;
        --task)
            LIBERO_TASK="$2"; shift 2 ;;
        --output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        *)
            echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ ! -d "${LEROBOT_ROOT}/lerobot" ]]; then
    echo "ERROR: 未找到 lerobot 源码目录: ${LEROBOT_ROOT}"
    exit 1
fi

# =============================================================================
# 自动寻找最新检查点
# =============================================================================
CKPT_BASE_DIR="${DISTILL_OUTPUT_DIR}/checkpoints"

if [[ -z "${CKPT_PATH}" ]]; then
    # 优先数字步数目录 / step_XXXXXXX（按步数降序取最新），其次 best
    LATEST=$(
        {
            ls -d "${CKPT_BASE_DIR}"/[0-9]* 2>/dev/null || true
            ls -d "${CKPT_BASE_DIR}"/step_* 2>/dev/null || true
        } | sort -V | tail -n 1
    )
    if [[ -n "${LATEST}" ]]; then
        CKPT_PATH="${LATEST}"
    elif [[ -d "${CKPT_BASE_DIR}/best" ]]; then
        CKPT_PATH="${CKPT_BASE_DIR}/best"
    else
        echo "ERROR: 未找到检查点目录，请先运行训练或通过 --ckpt 指定路径。"
        exit 1
    fi
    echo "自动选择检查点: ${CKPT_PATH}"
fi

# =============================================================================
# 校验检查点
# =============================================================================
if [[ ! -d "${CKPT_PATH}" ]]; then
    echo "ERROR: 检查点目录不存在: ${CKPT_PATH}"
    exit 1
fi

POLICY_PATH="${CKPT_PATH}"
if [[ -d "${CKPT_PATH}/pretrained_model" ]]; then
    POLICY_PATH="${CKPT_PATH}/pretrained_model"
fi

if [[ ! -f "${POLICY_PATH}/config.json" ]]; then
    echo "ERROR: 检查点模型目录缺少 config.json: ${POLICY_PATH}"
    exit 1
fi
if [[ ! -f "${POLICY_PATH}/model.safetensors" ]]; then
    echo "ERROR: 检查点模型目录缺少 model.safetensors: ${POLICY_PATH}"
    exit 1
fi

CKPT_NAME=$(basename "${CKPT_PATH}")
EVAL_RUN_DIR="${OUTPUT_DIR}/${LIBERO_TASK}_${CKPT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${EVAL_RUN_DIR}"

echo "======================================================"
echo "SmolVLA 学生模型评测"
echo "检查点:    ${CKPT_PATH}"
echo "模型目录:  ${POLICY_PATH}"
echo "评测设备:  ${EVAL_DEVICE}"
echo "任务:      ${LIBERO_TASK}"
echo "Episode数: ${N_EPISODES}，Batch: ${BATCH_SIZE}"
echo "结果输出:  ${EVAL_RUN_DIR}"
echo "LeRobot:   ${LEROBOT_ROOT}"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "Use Mirror : ${USE_HF_MIRROR}"
echo "======================================================"

conda run -n lerobot --no-capture-output env PYTHONPATH="${PYTHONPATH}" \
python - <<PY
import torch

device = "${EVAL_DEVICE}"
if device.startswith("cuda") and (not torch.cuda.is_available() or torch.cuda.device_count() <= 0):
    raise RuntimeError(
        f"请求使用 {device}，但当前会话没有可见 CUDA 设备。"
        "请确认 NVIDIA driver / 容器 GPU 透传是否正常，或显式传 --device cpu。"
    )
PY

# =============================================================================
# 方式一：lerobot-eval（需要 gymnasium + libero gym wrapper）
# =============================================================================
eval_lerobot() {
    echo "[方式一] lerobot-eval ..."
    GPU_ID="${EVAL_DEVICE#cuda:}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python "${LEROBOT_ROOT}/lerobot/scripts/lerobot_eval.py" \
        --policy.path="${POLICY_PATH}" \
        --env.type="libero" \
        --env.task="${LIBERO_TASK}" \
        --eval.batch_size="${BATCH_SIZE}" \
        --eval.n_episodes="${N_EPISODES}" \
        --policy.device="${EVAL_DEVICE}" \
        --policy.use_amp=false \
        --policy.use_cache=true \
        --output_dir="${EVAL_RUN_DIR}"
}

# =============================================================================
# 方式二：student_policy_wrapper 模型加载验证（不需要 gym）
# =============================================================================
eval_wrapper() {
    echo "[方式二] 回退到模型加载验证模式（无 gym 环境）..."
    GPU_ID="${EVAL_DEVICE#cuda:}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python - <<PYEOF
import sys
sys.path.insert(0, "${LEROBOT_ROOT}")
sys.path.insert(0, "${SRC_DIR}")

import json, logging
from pathlib import Path
import torch
from student_policy_wrapper import load_student_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval")

ckpt_path = "${POLICY_PATH}"
device    = "${EVAL_DEVICE}"
out_dir   = Path("${EVAL_RUN_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"加载学生模型: {ckpt_path} -> {device}")
policy = load_student_policy(ckpt_path, device)
logger.info("模型加载成功")

n_params = sum(p.numel() for p in policy.parameters())
n_train  = sum(p.numel() for p in policy.parameters() if p.requires_grad)
logger.info(f"参数量: {n_params/1e6:.2f}M  可训练: {n_train/1e6:.2f}M")

info = {
    "checkpoint": ckpt_path,
    "device": device,
    "n_params_M": round(n_params / 1e6, 2),
    "n_trainable_M": round(n_train / 1e6, 2),
    "policy_name": policy.name,
}
with open(out_dir / "eval_info.json", "w") as f:
    json.dump(info, f, indent=2, default=str)
logger.info(f"评测元信息已保存至: {out_dir / 'eval_info.json'}")
logger.info("提示：完整 LIBERO 环境评测需安装 libero gym wrapper，再使用方式一。")
PYEOF
}

# =============================================================================
# 自动选择评测方式
# =============================================================================
if python -c "import gymnasium; import lerobot.envs" 2>/dev/null; then
    eval_lerobot
else
    echo "未检测到 gymnasium / libero gym，回退至模型加载验证模式（不包含真实 LIBERO 成功率）。"
    eval_wrapper
fi

echo "======================================================"
echo "评测完成！结果目录: ${EVAL_RUN_DIR}"
echo "======================================================"
