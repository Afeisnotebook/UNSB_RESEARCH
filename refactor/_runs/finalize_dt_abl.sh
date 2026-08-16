#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/yc/unsb_tired
PY=/home/yc/anaconda3/envs/unsb_cov/bin/python
DONE="$ROOT/refactor/_runs/metrics_abl/dt_abl_extra.done"
FLAG="$ROOT/refactor/_runs/metrics_abl/.dt_abl_finalized"
LOG="$ROOT/refactor/_runs/finalize_dt_abl.log"

# 只有三连跑成功结束、且尚未收尾时才动作；训练仍运行时直接退出。
if [ ! -f "$DONE" ]; then
  exit 0
fi
if [ -f "$FLAG" ]; then
  exit 0
fi
if systemctl --user is-active --quiet unsb_train_all_dt.service; then
  exit 0
fi

"$PY" "$ROOT/refactor/_runs/summarize_dt_abl_extra.py" > "$LOG" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  touch "$FLAG"
fi
exit 0
