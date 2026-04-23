#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
LEROBOT_ROOT="${LEROBOT_ROOT:-${PROJECT_ROOT}/../lerobot/src}"

DEFAULT_STUDENT=0
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/distill_config.yaml"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_distill_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG_FILE}") 2>&1

STUDENT_DEVICES=${STUDENT_DEVICES:-$DEFAULT_STUDENT}
DISTILL_CONFIG=${DISTILL_CONFIG:-$DEFAULT_CONFIG}
if [[ "${DISTILL_CONFIG}" != /* ]]; then
    DISTILL_CONFIG="${PROJECT_ROOT}/${DISTILL_CONFIG}"
fi
NUM_PROCESSES=$(echo "$STUDENT_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

if [[ ! -d "${LEROBOT_ROOT}/lerobot" ]]; then
    echo "ERROR: 未找到 lerobot 源码目录: ${LEROBOT_ROOT}"
    exit 1
fi
if [[ ! -f "${DISTILL_CONFIG}" ]]; then
    echo "ERROR: 未找到蒸馏配置: ${DISTILL_CONFIG}"
    exit 1
fi

export PYTHONPATH="${LEROBOT_ROOT}:${SRC_DIR}:${PYTHONPATH:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
USE_HF_MIRROR="${USE_HF_MIRROR:-1}"
if [[ "${USE_HF_MIRROR}" == "1" ]]; then
    unset no_proxy https_proxy NO_PROXY HTTPS_PROXY HTTP_PROXY http_proxy ALL_PROXY all_proxy
    export HF_ENDPOINT
fi

conda run -n lerobot --no-capture-output env PYTHONPATH="${PYTHONPATH}" \
python - <<PY
import lerobot
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
    raise RuntimeError(
        "未检测到可见 CUDA 设备，train_distill 不会回退到 CPU。"
        "请确认 NVIDIA driver / 容器 GPU 透传是否正常。"
    )
PY

echo "====================================================================="
echo " 学生物理卡 : $STUDENT_DEVICES"
echo " 进程数量   : $NUM_PROCESSES"
echo " 配置文件   : $DISTILL_CONFIG"
echo " LeRobot    : $LEROBOT_ROOT"
echo " HF_ENDPOINT: ${HF_ENDPOINT}"
echo " Use Mirror : ${USE_HF_MIRROR}"
echo " 日志文件   : ${LOG_FILE}"
echo "====================================================================="

cd "${PROJECT_ROOT}"

conda run -n lerobot --no-capture-output \
accelerate launch \
    --num_processes $NUM_PROCESSES \
    --gpu_ids $STUDENT_DEVICES \
    src/distill.py \
    --config "$DISTILL_CONFIG" \
    "$@"
