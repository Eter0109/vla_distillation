#!/bin/bash
set -e

export PYTHONPATH=/hqlab/workspace/zhaozy/lerobot/src:src/:${PYTHONPATH:-}

DEFAULT_TEACHER=0
DEFAULT_STUDENT=0,1,2,3

TEACHER_DEVICE=${TEACHER_DEVICE:-$DEFAULT_TEACHER}
STUDENT_DEVICES=${STUDENT_DEVICES:-$DEFAULT_STUDENT}
NUM_PROCESSES=4

# 🔥 关键修复：让 accelerate 自动给每个进程隔离可见 GPU
export ACCELERATE_DISTRIBUTED_DEVICE_ISOLATION=1

# 教师设备（物理）
FINAL_TEACHER_DEVICE="cuda:${TEACHER_DEVICE}"

echo "====================================================================="
echo " 教师物理卡 : $TEACHER_DEVICE"
echo " 学生物理卡 : $STUDENT_DEVICES"
echo " 进程数量   : $NUM_PROCESSES"
echo "====================================================================="

# 🔥 最终启动（完全正确版）
conda run -n lerobot --no-capture-output \
accelerate launch \
    --num_processes $NUM_PROCESSES \
    --gpu_ids $STUDENT_DEVICES \
    src/distill.py \
    --config configs/distill_config.yaml \
    --distill.teacher_device "$FINAL_TEACHER_DEVICE" \
    "$@"