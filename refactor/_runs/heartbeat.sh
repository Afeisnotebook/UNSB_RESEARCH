#!/usr/bin/env bash
set -uo pipefail

LOG=/home/yc/unsb_tired/logs/HEARTBEAT.log
JSON=/home/yc/unsb_tired/refactor/_runs/last_heartbeat.json
mkdir -p "$(dirname "$LOG")" "$(dirname "$JSON")"

TS=$(date '+%Y-%m-%d %H:%M:%S %Z')

procs=$(pgrep -af 'train.py|run_dtcov_repro.sh' | grep -v pgrep || true)
done_file=/home/yc/unsb_tired/refactor/_runs/dtcov_repro.done

if [ -f "$done_file" ]; then
  state=DONE
elif [ -n "$procs" ]; then
  state=RUNNING
else
  state=STALLED
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "na")
else
  gpu="na"
fi

latest_log=$(tail -n 3 /home/yc/unsb_tired/refactor/_runs/dtcov_warmup.log 2>/dev/null | tail -1 | cut -c1-140 || echo "no log yet")

echo "[$TS] state=$state gpu_util_mem_mib=[$gpu] pid_count=$(echo "$procs" | grep -c . || true) last_log=\"$latest_log\"" >> "$LOG"

cat > "$JSON" <<EOF
{
  "ts": "$TS",
  "state": "$state",
  "gpu": "$gpu",
  "pid_count": $(echo "$procs" | grep -c . || echo 0),
  "last_log": "$latest_log"
}
EOF
