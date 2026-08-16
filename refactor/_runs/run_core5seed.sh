#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yc/unsb_tired/refactor/_runs

bash "$ROOT/run_dt_pilot_multiseed.sh" 2029 2030
bash "$ROOT/run_hj_pilot_multiseed.sh" 2029 2030
echo "DONE" > "$ROOT/metrics_ms/core5seed.done"
