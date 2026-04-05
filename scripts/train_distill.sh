#!/bin/bash
set -e

export PYTHONPATH=/hqlab/workspace/zhaozy/lerobot/src:src/:${PYTHONPATH:-}

DEFAULT_STUDENT=0,1,2,3

STUDENT_DEVICES=${STUDENT_DEVICES:-$DEFAULT_STUDENT}
NUM_PROCESSES=$(echo "$STUDENT_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

echo "====================================================================="
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
    "$@"