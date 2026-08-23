#!/usr/bin/env bash
# Single launch entry point for the DT/HJ/HNEK clean re-exploration long task.
set -euo pipefail

SESSION="unsb_clean_reexploration_20260824"
REPO="/home/yc/unsb_tired"
PY="/home/yc/anaconda3/envs/unsb_cov/bin/python"
CODE="$REPO/算法设计模块/code"
RUNTIME="$REPO/runtime_4090/clean_reexploration_20260824"
LOG="$RUNTIME/logs/run_long.log"
HEARTBEAT="$RUNTIME/logs/heartbeat.json"

mkdir -p "$RUNTIME/logs"

export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=2026

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-session -d -s "$SESSION" \
    "cd '$CODE' && exec '$PY' clean_reexploration/run_long.py --stage train >> '$LOG' 2>&1"
fi

cat <<EOF
tmux_session=$SESSION
log=$LOG
heartbeat=$HEARTBEAT
attach: tmux attach -t $SESSION
EOF
