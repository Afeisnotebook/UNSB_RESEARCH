#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yc/unsb_tired/refactor/baseline
PY=/home/yc/anaconda3/envs/unsb_cov/bin/python
TRAIN_DATA=/home/yc/UNSB_Patch/datasets/final6_train160_test40_unpaired
VALO=/home/yc/UNSB_Patch/datasets/final6train_valO5x16_offset560_unpaired
CKPT=/home/yc/unsb_tired/refactor/_runs/checkpoints
RES=/home/yc/unsb_tired/refactor/_runs/results_hj
MET=/home/yc/unsb_tired/refactor/_runs/metrics_hj
EVAL_PY=/home/yc/UNSB_Cov5/tools/evaluate_restoration.py
GPU=0

mkdir -p "$MET"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1

COMMON=(--dataroot "$TRAIN_DATA" --checkpoints_dir "$CKPT" --mode sb --dataset_mode unaligned --direction AtoB
  --lambda_SB 1.0 --lambda_NCE 1.0 --tau 0.01 --batch_size 16 --load_size 128 --crop_size 128
  --preprocess resize_and_crop --num_threads 4 --gpu_ids "$GPU" --seed 2026 --n_epochs_decay 0
  --lr 0.0001 --save_latest_freq 4800 --print_freq 160 --display_freq 1600 --display_id -1 --no_html
  --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond --normG instance --normD instance
  --pool_size 0 --num_timesteps 5 --no_flip)

train_hj() {
  local name="$1"; local control="$2"
  "$PY" train.py "${COMMON[@]}" --name "$name" --model hj --n_epochs 200 --save_epoch_freq 50 \
    --continue_train --epoch latest --pretrained_name dtcov_clean_warmup_e20 \
    --hj_enable True --hj_layers 0 --hj_start_epoch 5 --hj_probe_mode central_consensus \
    --hj_strength 0.5 --hj_boundary_scale 0.001 --hj_min_risk 0.05 \
    --hj_direction joint --hj_scales 1,2,4 --hj_gate_quantile 0.75 --hj_control "$control" \
    >> "$MET/${name}_train.log" 2>&1
}

eval_hj() {
  local name="$1"
  "$PY" test.py --dataroot "$VALO" --name "$name" --checkpoints_dir "$CKPT" --results_dir "$RES" \
    --epoch 200 --model sb --mode sb --dataset_mode unaligned --direction AtoB \
    --batch_size 1 --load_size 128 --crop_size 128 --preprocess resize_and_crop \
    --num_threads 0 --gpu_ids "$GPU" --num_test 80 --serial_batches --no_flip \
    --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond \
    --normG instance --normD instance --num_timesteps 5 >> "$MET/${name}_test.log" 2>&1
  "$PY" "$EVAL_PY" --manifest "$VALO/manifest.csv" \
    --fake-dir "$RES/$name/test_200/images/fake_5" --split test \
    --out "$MET/$name" --device cuda >> "$MET/${name}_eval.log" 2>&1
}

echo "=== train hj true ==="
train_hj hj_clean_true_e200 true
echo "=== train hj roll ==="
train_hj hj_clean_roll_e200 roll
echo "=== eval plain/true/roll on val-O ==="
eval_hj dtcov_clean_plain_e200
eval_hj hj_clean_true_e200
eval_hj hj_clean_roll_e200
echo "DONE" > "$MET/hj_eval.done"
