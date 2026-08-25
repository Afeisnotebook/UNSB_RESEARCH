#!/usr/bin/env bash
set -euo pipefail

# UNSB 动机图旁路任务：训练 + 离线测量 + 汇总。
# 必须在宿主机 GPU 上执行（本 Codex 沙箱无 GPU 设备）。

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
SRC="$ROOT/src"
PY="${PY:-python3}"
REFACTOR="${REFACTOR:-$REPO_ROOT/foundation/canonical/src}"
WORK_ROOT="${UNSB_MOTIVATION_RUN_ROOT:-$REPO_ROOT/runs/MOT-001}"
CKPT="$WORK_ROOT/checkpoints"
RAW="$WORK_ROOT/raw"
REPORT_DIR="$WORK_ROOT/reports"
FIGURE_DIR="$WORK_ROOT/figures"
SINGLE="$ROOT/datasets/single"
AIO="$ROOT/datasets/aio"
GPU="${GPU:-0}"
DOMAINS=(FoggyCityscapes LowLightTrafficData RainCityscapes RSCityscapes SnowTrafficData)

command -v "$PY" >/dev/null 2>&1 || [[ -x "$PY" ]] || {
  echo "python interpreter not found: $PY" >&2
  exit 2
}
[[ -f "$REFACTOR/train.py" ]] || {
  echo "baseline train.py not found under REFACTOR=$REFACTOR" >&2
  exit 2
}
[[ -d "$SINGLE" && -d "$AIO" ]] || {
  echo "prepared datasets missing under $ROOT/datasets; raw data are not shipped" >&2
  exit 2
}

mkdir -p "$CKPT" "$RAW" "$REPORT_DIR" "$FIGURE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

COMMON=(
  --checkpoints_dir "$CKPT" --mode sb --dataset_mode unaligned --direction AtoB
  --lambda_SB 1.0 --lambda_NCE 1.0 --tau 0.01 --batch_size 1
  --load_size 128 --crop_size 128 --preprocess resize_and_crop
  --num_threads 0 --gpu_ids 0 --n_epochs_decay 0 --lr 0.0001
  --save_latest_freq 1000000 --print_freq 100 --display_freq 1000000
  --display_id -1 --no_html --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond
  --normG instance --normD instance --pool_size 0 --num_timesteps 5 --no_flip
)

say() { printf '[motivation %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [[ "${WAIT_MAIN:-0}" == "1" ]]; then
  : "${MAIN_RUNS_ROOT:?set MAIN_RUNS_ROOT when WAIT_MAIN=1}"
  say "waiting for main stage2 marker and any active main stage3 search queue ..."
  while [[ ! -f "$MAIN_RUNS_ROOT/metrics_clean_core/core_clean.done" ]] \
        || pgrep -f "$MAIN_RUNS_ROOT/run_stage3_search_queue.sh" >/dev/null \
        || pgrep -f "$MAIN_RUNS_ROOT/run_hnek_search_variant.py" >/dev/null; do
    sleep 300
  done
  say "main stage2 done and stage3 queue idle, proceeding"
fi

cd "$REFACTOR"

train() {
  local name="$1"; shift
  say "TRAIN $name"
  nice -n 19 "$PY" train.py "${COMMON[@]}" --name "$name" --seed 2026 "$@" \
    > "$REPORT_DIR/${name}.train.log" 2>&1
  say "DONE $name"
}

for d in "${DOMAINS[@]}"; do
  train "single_${d}_s2026" --dataroot "$SINGLE/$d" --model sb \
    --n_epochs 20 --save_epoch_freq 1
done

train "aio_plain_s2026" --dataroot "$AIO" --model sb \
  --n_epochs 20 --save_epoch_freq 1

train "aio_dt_s2026" --dataroot "$AIO" --model dtcov \
  --n_epochs 5 --save_epoch_freq 1 --continue_train --epoch latest \
  --pretrained_name aio_plain_s2026 \
  --dtcov_lambda 0.001 --dtcov_lambda_schedule ramp_hold_cosine_decay \
  --dtcov_ramp_start_epoch 1 --dtcov_ramp_end_epoch 5 \
  --dtcov_decay_start_epoch 15 --dtcov_decay_end_epoch 25 --dtcov_lambda_min 0.0 \
  --dtcov_m 4 --dtcov_region_patch 32 --dtcov_u_floor 1e-8 --dtcov_norm_eps 1e-4 \
  --dtcov_norm_momentum 0.98 --dtcov_norm_clip 3.0 \
  --dtcov_domain_balance grouped_domain --dtcov_warmup_iters 300 --dtcov_time_mode actual

say "building measurement plan"
"$PY" "$SRC/build_measure_plan.py"

say "running measurement"
"$PY" "$SRC/run_measure.py" --plan "$ROOT/MEASURE_PLAN.json" \
  --root "$ROOT" --refactor-root "$REFACTOR" --device cuda \
  > "$REPORT_DIR/measure.log" 2>&1

say "summarizing"
"$PY" "$SRC/summarize_motivation.py" --raw-dir "$RAW" \
  --out "$REPORT_DIR/MOTIVATION_SUMMARY.json" --bridge-times 1,2,3

say "writing checkpoint index"
"$PY" - "$CKPT" "$WORK_ROOT/CHECKPOINT_INDEX.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
ckpt = Path(sys.argv[1]); out = Path(sys.argv[2])
entries = []
for p in sorted(ckpt.rglob("*_net_G.pth")):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    entries.append({"path": str(p.relative_to(ckpt)), "sha256": h.hexdigest()})
out.write_text(json.dumps({"schema_version": 1, "entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"indexed {len(entries)} checkpoints")
PY

say "ALL_DONE"
