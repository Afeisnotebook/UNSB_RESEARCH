#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yc/unsb_tired/refactor/baseline
PY=/home/yc/anaconda3/envs/unsb_cov/bin/python
DATA=/home/yc/UNSB_Cov5/datasets/final6_train160_test40_unpaired
CKPT=/home/yc/unsb_tired/refactor/_runs/checkpoints
LOG=/home/yc/unsb_tired/refactor/_runs
GPU=0
SEED=2026

COMMON=(--dataroot "$DATA" --checkpoints_dir "$CKPT" --mode sb --dataset_mode unaligned --direction AtoB
  --lambda_SB 1.0 --lambda_NCE 1.0 --tau 0.01 --batch_size 16 --load_size 128 --crop_size 128
  --preprocess resize_and_crop --num_threads 4 --gpu_ids "$GPU" --seed "$SEED" --n_epochs_decay 0
  --lr 0.0001 --save_latest_freq 5000 --print_freq 100 --display_freq 400 --display_id -1 --no_html
  --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond --normG instance --normD instance
  --pool_size 0 --num_timesteps 5)

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONDONTWRITEBYTECODE=1

echo "=== stage: continue plain from warmup to 200 ==="
"$PY" train.py "${COMMON[@]}" --name dtcov_clean_plain_e200 --model sb --n_epochs 200 \
  --save_epoch_freq 50 --continue_train --epoch latest --pretrained_name dtcov_clean_warmup_e20 \
  >> "$LOG/plain_baseline.log" 2>&1

echo "DONE" >> "$LOG/plain_baseline.done"
