#!/bin/bash
set -euo pipefail

export PYTHONPATH=/hqlab/workspace/zhaozy/lerobot/src:src/:${PYTHONPATH:-}

DEFAULT_STUDENT=0
DEFAULT_CONFIG=configs/distill_smoke_test.yaml

STUDENT_DEVICES=${STUDENT_DEVICES:-$DEFAULT_STUDENT}
SMOKE_TEST_CONFIG=${SMOKE_TEST_CONFIG:-$DEFAULT_CONFIG}
NUM_PROCESSES=$(echo "$STUDENT_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

echo "====================================================================="
echo " Smoke Test Config : $SMOKE_TEST_CONFIG"
echo " 学生物理卡        : $STUDENT_DEVICES"
echo " 进程数量          : $NUM_PROCESSES"
echo "====================================================================="

conda run -n lerobot --no-capture-output \
accelerate launch \
    --num_processes $NUM_PROCESSES \
    --gpu_ids $STUDENT_DEVICES \
    src/distill.py \
    --config "$SMOKE_TEST_CONFIG" \
    "$@"
