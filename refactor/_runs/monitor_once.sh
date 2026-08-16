#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yc/unsb_tired
RUNS="$ROOT/refactor/_runs"
MET_HJ="$RUNS/metrics_hj"
LOG="$ROOT/logs/MONITOR_LOG.md"
HEART_LOG="$ROOT/logs/HEARTBEAT.log"
HEART_JSON="$RUNS/last_heartbeat.json"
STATUS="$ROOT/STATUS.json"

TS=$(date '+%Y-%m-%d %H:%M:%S %Z')

# 当前活跃服务：优先 HJ 实验，其次 DT 实验
if systemctl --user is-active --quiet unsb_train_hj.service; then
  svc="unsb_train_hj"
  svc_state="active"
elif systemctl --user is-active --quiet unsb_train_all_dt.service; then
  svc="unsb_train_all_dt"
  svc_state="active"
elif systemctl --user is-failed --quiet unsb_train_hj.service || systemctl --user is-failed --quiet unsb_train_all_dt.service; then
  svc="failed"
  svc_state="failed"
else
  svc="none"
  svc_state="inactive"
fi

# 阶段判断
if [ -f "$MET_HJ/hj_eval.done" ]; then
  stage="HJ_DONE"
elif [ "$svc" = "unsb_train_hj" ]; then
  if [ -f "$MET_HJ/hj_clean_roll_e200_train.log" ]; then
    stage="HJ_ROLL"
  elif [ -f "$MET_HJ/hj_clean_true_e200_train.log" ]; then
    stage="HJ_TRUE"
  else
    stage="HJ_STARTING"
  fi
elif [ -f "$RUNS/dtcov_repro_v2.done" ]; then
  stage="DT_DONE"
elif [ "$svc" = "unsb_train_all_dt" ]; then
  stage="DT_RUNNING"
else
  stage="IDLE"
fi

gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "na")
proc_count=$(pgrep -fc 'train.py' || true)
proc_count=${proc_count:-0}

latest_log=""
if [ "$stage" = "HJ_TRUE" ]; then latest_log="$MET_HJ/hj_clean_true_e200_train.log"; fi
if [ "$stage" = "HJ_ROLL" ]; then latest_log="$MET_HJ/hj_clean_roll_e200_train.log"; fi
if [ "$stage" = "HJ_DONE" ]; then latest_log="$MET_HJ/hj_clean_roll_e200_train.log"; fi
if [ "$stage" = "DT_RUNNING" ] || [ "$stage" = "DT_DONE" ]; then latest_log="$RUNS/dtcov_plain_continue.log"; fi
last_tail=$(tail -n 2 "$latest_log" 2>/dev/null | tr '\n' ' ' | cut -c1-180)

{
  echo ""
  echo "## $TS"
  echo "- service=$svc_state($svc) stage=$stage gpu=[$gpu] pid=$proc_count"
  echo "- tail: \`$last_tail\`"
} >> "$LOG"

cat > "$HEART_JSON" <<EOF
{
  "ts": "$TS",
  "service": "$svc",
  "service_state": "$svc_state",
  "stage": "$stage",
  "gpu": "$gpu",
  "pid_count": $proc_count,
  "last_log": "$last_tail"
}
EOF

echo "[$TS] service=$svc_state($svc) stage=$stage gpu=[$gpu] pid=$proc_count last_log=\"$last_tail\"" >> "$HEART_LOG"

cat > "$STATUS" <<EOF
{
  "updated_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "phase": "method_experiments",
  "service": "$svc",
  "service_state": "$svc_state",
  "stage": "$stage",
  "gpu_util_mem_mib": "$gpu",
  "pid_count": $proc_count
}
EOF
