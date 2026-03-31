#!/usr/bin/env bash
# 蒸馏训练启动脚本
# 用法：bash scripts/train_distill.sh [可选参数]
#
# 示例：
#   bash scripts/train_distill.sh
#   bash scripts/train_distill.sh --teacher_device cuda:1 --student_devices 2,3,4,5
#   bash scripts/train_distill.sh --alpha 0.3 --accum_steps 8 --steps 50000
#   bash scripts/train_distill.sh --feature_distill false --logit_distill true

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
PROJECT_ROOT="/hqlab/workspace/zhaozy/vla_distillation"
CONFIG_FILE="${PROJECT_ROOT}/configs/distill_config.yaml"
SRC_DIR="${PROJECT_ROOT}/src"

# 添加 lerobot 与项目 src 到 PYTHONPATH
export PYTHONPATH="/hqlab/workspace/zhaozy/lerobot/src:${SRC_DIR}:${PYTHONPATH:-}"

# =============================================================================
# GPU 配置（默认：教师卡5，学生卡1,2,3,4）
# 可通过环境变量 TEACHER_DEVICE / STUDENT_DEVICES 覆盖
# 为避免教师 OOM，teacher 独占一张物理卡；torchrun 仅为学生卡启动 DDP 进程
# =============================================================================
TEACHER_DEVICE="${TEACHER_DEVICE:-5}"    # 物理 GPU 编号（教师模型）
STUDENT_DEVICES="${STUDENT_DEVICES:-1,2,3,4}"
NUM_GPUS=$(echo "${STUDENT_DEVICES}" | tr ',' '\n' | wc -l)

# 将教师卡放在第一个，学生卡依次排列，供 CUDA_VISIBLE_DEVICES 使用
# torchrun 进程看到的 cuda:0 = TEACHER_DEVICE，cuda:1..N = STUDENT_DEVICES
ALL_DEVICES="${TEACHER_DEVICE},${STUDENT_DEVICES}"

# local_rank 0 = 学生进程 0（对应物理 STUDENT_DEVICES 第 1 张）
# 教师模型挂载在 cuda:0（即物理 TEACHER_DEVICE）
# 学生 local_rank k 对应 cuda:{k+1}（物理 STUDENT_DEVICES[k]）
TEACHER_CUDA_IDX=0                                 # 在重映射后的索引
STUDENT_CUDA_OFFSET=1                              # 学生 local_rank 相对偏移

echo "======================================================"
echo "VLA 蒸馏训练"
echo "配置文件: ${CONFIG_FILE}"
echo "教师卡: cuda:${TEACHER_DEVICE}（重映射为 cuda:${TEACHER_CUDA_IDX}）"
echo "学生训练卡: ${STUDENT_DEVICES} (共 ${NUM_GPUS} 张, 重映射为 cuda:1..${NUM_GPUS})"
echo "CUDA_VISIBLE_DEVICES: ${ALL_DEVICES}"
echo "传入参数: $*"
echo "======================================================"

# 创建日志和输出目录
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/outputs/checkpoints"

# =============================================================================
# 启动分布式训练
# torchrun 会自动设置 LOCAL_RANK / RANK / WORLD_SIZE 环境变量
# 注意：此处传给 distill.py 的 --teacher_device / --student_devices 均为
#       CUDA_VISIBLE_DEVICES 重映射后的索引
# =============================================================================
CUDA_VISIBLE_DEVICES="${ALL_DEVICES}" \
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29501 \
    "${SRC_DIR}/distill.py" \
    --config "${CONFIG_FILE}" \
    --teacher_device "cuda:${TEACHER_CUDA_IDX}" \
    --student_devices "${STUDENT_DEVICES}" \
    --student_cuda_offset "${STUDENT_CUDA_OFFSET}" \
    "$@"
