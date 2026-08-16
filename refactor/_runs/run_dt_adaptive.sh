#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yc/unsb_tired/refactor/baseline
PY=/home/yc/anaconda3/envs/unsb_cov/bin/python
DATA=/home/yc/UNSB_Cov5/datasets/final6_train160_test40_unpaired
CKPT=/home/yc/unsb_tired/refactor/_runs/checkpoints
MET=/home/yc/unsb_tired/refactor/_runs/metrics_adaptive
EVAL_PY=/home/yc/UNSB_Cov5/tools/evaluate_restoration.py
DIAG=/home/yc/unsb_tired/refactor/_runs/diagnostics/dtcov_adaptive.jsonl
GPU=0
mkdir -p "$MET"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1

COMMON=(--dataroot "$DATA" --checkpoints_dir "$CKPT" --mode sb --dataset_mode unaligned --direction AtoB
  --lambda_SB 1.0 --lambda_NCE 1.0 --tau 0.01 --batch_size 16 --load_size 128 --crop_size 128
  --preprocess resize_and_crop --num_threads 4 --gpu_ids "$GPU" --seed 2026 --n_epochs_decay 0
  --lr 0.0001 --save_latest_freq 5000 --print_freq 100 --display_id -1 --no_html
  --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond --normG instance --normD instance
  --pool_size 0 --num_timesteps 5)

NAME=dtcov_adaptive_e200

echo "=== adaptive DT window (25 epochs) ==="
"$PY" train.py "${COMMON[@]}" --name "$NAME" --model dtcov --n_epochs 25 --save_epoch_freq 10 \
  --continue_train --epoch latest --pretrained_name dtcov_clean_warmup_e20 \
  --dtcov_lambda 0.001 --dtcov_lambda_schedule adaptive \
  --dtcov_ramp_start_epoch 1 --dtcov_ramp_end_epoch 5 \
  --dtcov_adaptive_epsilon 0.02 --dtcov_adaptive_patience 5 \
  --dtcov_m 4 --dtcov_warmup_iters 300 --dtcov_diag_out "$DIAG" \
  > "$MET/${NAME}_train.log" 2>&1

echo "=== continue plain to 200 ==="
"$PY" train.py "${COMMON[@]}" --name "$NAME" --model dtcov --n_epochs 200 --epoch_count 26 \
  --save_epoch_freq 50 --continue_train --epoch latest --dtcov_lambda 0 \
  >> "$MET/${NAME}_train.log" 2>&1

echo "=== eval ==="
"$PY" test.py --dataroot "$DATA" --name "$NAME" --checkpoints_dir "$CKPT" --results_dir "$MET/results" \
  --epoch 200 --model sb --mode sb --dataset_mode unaligned --direction AtoB \
  --batch_size 1 --load_size 128 --crop_size 128 --preprocess resize_and_crop \
  --num_threads 0 --gpu_ids "$GPU" --num_test 240 --serial_batches \
  --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond \
  --normG instance --normD instance --num_timesteps 5 >> "$MET/${NAME}_test.log" 2>&1
"$PY" "$EVAL_PY" --manifest "$DATA/manifest.csv" \
  --fake-dir "$MET/results/$NAME/test_200/images/fake_5" --split test \
  --out "$MET/$NAME" --device cuda >> "$MET/${NAME}_eval.log" 2>&1
echo "DONE" > "$MET/adaptive.done"
