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

echo "=== stage 1: warmup 20 epochs (reuse if exists) ==="
if [ ! -f "$CKPT/dtcov_clean_warmup_e20/20_net_G.pth" ]; then
  "$PY" train.py "${COMMON[@]}" --name dtcov_clean_warmup_e20 --model sb --n_epochs 20 --save_epoch_freq 20 \
    > "$LOG/dtcov_warmup.log" 2>&1
fi

echo "=== stage 2: dtcov window 25 epochs (ramp5 hold15 decay25) ==="
"$PY" train.py "${COMMON[@]}" --name dtcov_clean_best_e200 --model dtcov --n_epochs 25 --save_epoch_freq 10 \
  --continue_train --epoch latest --pretrained_name dtcov_clean_warmup_e20 \
  --dtcov_lambda 0.001 --dtcov_lambda_schedule ramp_hold_cosine_decay \
  --dtcov_ramp_start_epoch 1 --dtcov_ramp_end_epoch 5 \
  --dtcov_decay_start_epoch 15 --dtcov_decay_end_epoch 25 --dtcov_lambda_min 0.0 \
  --dtcov_m 4 --dtcov_region_patch 32 --dtcov_u_floor 1e-8 --dtcov_norm_eps 1e-4 \
  --dtcov_norm_momentum 0.98 --dtcov_norm_clip 3.0 --dtcov_domain_balance grouped_domain \
  --dtcov_warmup_iters 300 --dtcov_time_mode actual \
  > "$LOG/dtcov_window.log" 2>&1

echo "=== stage 3: continue plain to 200 ==="
"$PY" train.py "${COMMON[@]}" --name dtcov_clean_best_e200 --model dtcov --n_epochs 200 --epoch_count 26 \
  --save_epoch_freq 50 --continue_train --epoch latest --dtcov_lambda 0 \
  > "$LOG/dtcov_plain_continue.log" 2>&1

echo "DONE" > "$LOG/dtcov_repro_v2.done"
