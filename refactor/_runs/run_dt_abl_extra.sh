#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yc/unsb_tired/refactor/baseline
PY=/home/yc/anaconda3/envs/unsb_cov/bin/python
TRAIN=/home/yc/UNSB_Cov5/datasets/final6_train160_test40_unpaired
CKPT=/home/yc/unsb_tired/refactor/_runs/checkpoints
MET=/home/yc/unsb_tired/refactor/_runs/metrics_abl
EVAL_PY=/home/yc/UNSB_Cov5/tools/evaluate_restoration.py
GPU=0
mkdir -p "$MET"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1

COMMON=(--dataroot "$TRAIN" --checkpoints_dir "$CKPT" --mode sb --dataset_mode unaligned --direction AtoB
  --lambda_SB 1.0 --lambda_NCE 1.0 --tau 0.01 --batch_size 16 --load_size 128 --crop_size 128
  --preprocess resize_and_crop --num_threads 4 --gpu_ids "$GPU" --seed 2026 --n_epochs_decay 0
  --lr 0.0001 --save_latest_freq 5000 --print_freq 100 --display_id -1 --no_html
  --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond --normG instance --normD instance
  --pool_size 0 --num_timesteps 5)

# DT ablation: reuse warmup, 25-epoch window, then plain to 200
dt_ablation() {
  local name="$1"; shift
  "$PY" train.py "${COMMON[@]}" --name "$name" --model dtcov --n_epochs 25 --save_epoch_freq 10 \
    --continue_train --epoch latest --pretrained_name dtcov_clean_warmup_e20 \
    --dtcov_lambda 0.001 --dtcov_lambda_schedule ramp_hold_cosine_decay \
    --dtcov_ramp_start_epoch 1 --dtcov_ramp_end_epoch 5 \
    --dtcov_decay_start_epoch 15 --dtcov_decay_end_epoch 25 --dtcov_lambda_min 0.0 \
    --dtcov_m 4 --dtcov_warmup_iters 300 "$@" > "$MET/${name}_train.log" 2>&1
  "$PY" train.py "${COMMON[@]}" --name "$name" --model dtcov --n_epochs 200 --epoch_count 26 \
    --save_epoch_freq 50 --continue_train --epoch latest --dtcov_lambda 0 >> "$MET/${name}_train.log" 2>&1
  "$PY" test.py --dataroot "$TRAIN" --name "$name" --checkpoints_dir "$CKPT" --results_dir "$MET/results" \
    --epoch 200 --model sb --mode sb --dataset_mode unaligned --direction AtoB \
    --batch_size 1 --load_size 128 --crop_size 128 --preprocess resize_and_crop \
    --num_threads 0 --gpu_ids "$GPU" --num_test 240 --serial_batches \
    --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond \
    --normG instance --normD instance --num_timesteps 5 >> "$MET/${name}_test.log" 2>&1
  "$PY" "$EVAL_PY" --manifest "$TRAIN/manifest.csv" \
    --fake-dir "$MET/results/$name/test_200/images/fake_5" --split test \
    --out "$MET/$name" --device cuda >> "$MET/${name}_eval.log" 2>&1
}

echo "=== DT-A2 frozen teacher -> self ==="
dt_ablation dt_abl_a2_self --dtcov_teacher self
echo "=== DT-A3 domain x time EMA -> global ==="
dt_ablation dt_abl_a3_global --dtcov_norm_mode global
echo "=== DT-A5 signal normalization -> raw U ==="
dt_ablation dt_abl_a5_nonorm --dtcov_signal_norm off
echo "DONE" > "$MET/dt_abl_extra.done"
