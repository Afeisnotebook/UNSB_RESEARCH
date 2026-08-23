#!/usr/bin/env bash
set -euo pipefail

# Train only Single-task and Plain All-in-One arms for additional seeds used by
# the window audit.  DT/HJ are intentionally excluded.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/yc/anaconda3/envs/unsb_cov/bin/python"
REFACTOR="/home/yc/unsb_tired/refactor/baseline"
CKPT="$ROOT/checkpoints"
SINGLE="$ROOT/datasets/single"
AIO="$ROOT/datasets/aio"
LOGDIR="$ROOT/reports/window_seed"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-2027 2028}"
DOMAINS=(FoggyCityscapes LowLightTrafficData RainCityscapes RSCityscapes SnowTrafficData)

mkdir -p "$CKPT" "$LOGDIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

COMMON=(
  --checkpoints_dir "$CKPT" --mode sb --dataset_mode unaligned --direction AtoB
  --lambda_SB 1.0 --lambda_NCE 1.0 --tau 0.01 --batch_size 1
  --load_size 128 --crop_size 128 --preprocess resize_and_crop
  --num_threads 0 --gpu_ids "$GPU" --n_epochs_decay 0 --lr 0.0001
  --save_latest_freq 1000000 --print_freq 100 --display_freq 1000000
  --display_id -1 --no_html --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond
  --normG instance --normD instance --pool_size 0 --num_timesteps 5 --no_flip
)

say() { printf '[window-seed %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

cd "$REFACTOR"

train() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.train.log"
  say "TRAIN $name"
  nice -n 19 "$PY" train.py "${COMMON[@]}" --name "$name" "$@" > "$log" 2>&1
  say "DONE $name"
}

for seed in $SEEDS; do
  for domain in "${DOMAINS[@]}"; do
    name="single_${domain}_s${seed}"
    if [[ -f "$CKPT/$name/20_net_G.pth" ]]; then
      say "SKIP existing $name"
      continue
    fi
    train "$name" --dataroot "$SINGLE/$domain" --model sb \
      --n_epochs 20 --save_epoch_freq 1 --seed "$seed"
  done

  name="aio_plain_s${seed}"
  if [[ -f "$CKPT/$name/20_net_G.pth" ]]; then
    say "SKIP existing $name"
    continue
  fi
  train "$name" --dataroot "$AIO" --model sb \
    --n_epochs 20 --save_epoch_freq 1 --seed "$seed"
done

say "ALL_TRAIN_DONE"
