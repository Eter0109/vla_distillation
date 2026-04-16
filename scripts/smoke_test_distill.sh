#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
LEROBOT_ROOT="${LEROBOT_ROOT:-${PROJECT_ROOT}/../lerobot/src}"

DEFAULT_STUDENT=0
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/distill_smoke_test.yaml"

STUDENT_DEVICES=${STUDENT_DEVICES:-$DEFAULT_STUDENT}
SMOKE_TEST_CONFIG=${SMOKE_TEST_CONFIG:-$DEFAULT_CONFIG}
if [[ "${SMOKE_TEST_CONFIG}" != /* ]]; then
    SMOKE_TEST_CONFIG="${PROJECT_ROOT}/${SMOKE_TEST_CONFIG}"
fi
NUM_PROCESSES=$(echo "$STUDENT_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

if [[ ! -d "${LEROBOT_ROOT}/lerobot" ]]; then
    echo "ERROR: 未找到 lerobot 源码目录: ${LEROBOT_ROOT}"
    exit 1
fi
if [[ ! -f "${SMOKE_TEST_CONFIG}" ]]; then
    echo "ERROR: 未找到 smoke test 配置: ${SMOKE_TEST_CONFIG}"
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
        "未检测到可见 CUDA 设备，smoke test 不会回退到 CPU。"
        "请确认 NVIDIA driver / 容器 GPU 透传是否正常。"
    )
PY

echo "====================================================================="
echo " Smoke Test Config : $SMOKE_TEST_CONFIG"
echo " 学生物理卡        : $STUDENT_DEVICES"
echo " 进程数量          : $NUM_PROCESSES"
echo " LeRobot Root      : $LEROBOT_ROOT"
echo " HF_ENDPOINT       : ${HF_ENDPOINT}"
echo " Use HF Mirror     : ${USE_HF_MIRROR}"
echo "====================================================================="

cd "${PROJECT_ROOT}"

conda run -n lerobot --no-capture-output \
accelerate launch \
    --num_processes $NUM_PROCESSES \
    --gpu_ids $STUDENT_DEVICES \
    src/distill.py \
    --config "$SMOKE_TEST_CONFIG" \
    "$@"
