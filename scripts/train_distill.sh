#!/bin/bash
set -e

# ── 环境 ────────────────────────────────────────────────────────────────────
export PYTHONPATH=/hqlab/workspace/zhaozy/lerobot/src:src/:${PYTHONPATH:-}

# ── 配置：只需要填物理 GPU 数字！───────────────────────────────────────────
# 这里填 物理 GPU 编号（你机器上真实的卡号）
DEFAULT_TEACHER=4
DEFAULT_STUDENT=5

# 读取命令行/环境变量
TEACHER_DEVICE=${TEACHER_DEVICE:-$DEFAULT_TEACHER}
STUDENT_DEVICES=${STUDENT_DEVICES:-$DEFAULT_STUDENT}
NUM_PROCESSES=1

# ── 核心修复：合并显卡，让进程同时看到教师+学生 ───────────────────────────
ALL_DEVICES="${TEACHER_DEVICE},${STUDENT_DEVICES}"
# 去重 + 排序（防止重复卡号）
ALL_DEVICES=$(echo $ALL_DEVICES | tr ',' '\n' | sort -n | uniq | tr '\n' ',' | sed 's/,$//')

# 自动映射：物理卡号 → 进程内局部 cuda:0 / cuda:1
LOCAL_TEACHER=0
LOCAL_STUDENT=1

# 最终传给代码的 teacher 设备
FINAL_TEACHER_DEVICE="cuda:$LOCAL_TEACHER"

echo "====================================================================="
echo " 🔥 单卡蒸馏 - 已自动修复 GPU 映射"
echo " 物理 GPU (可见) : $ALL_DEVICES"
echo " 教师(物理→局部) : $TEACHER_DEVICE → cuda:$LOCAL_TEACHER"
echo " 学生(物理→局部) : $STUDENT_DEVICES → cuda:$LOCAL_STUDENT"
echo "====================================================================="

# ── 启动训练 ───────────────────────────────────────────────────────────────
conda run -n lerobot --no-capture-output \
accelerate launch \
    --num_processes $NUM_PROCESSES \
    --gpu_ids $LOCAL_STUDENT \
    src/distill.py \
    --config configs/distill_config.yaml \
    --distill.teacher_device "$FINAL_TEACHER_DEVICE" \
    "$@"