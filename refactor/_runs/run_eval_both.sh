#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yc/unsb_tired/refactor/baseline
PY=/home/yc/anaconda3/envs/unsb_cov/bin/python
DATA=/home/yc/UNSB_Cov5/datasets/final6_train160_test40_unpaired
CKPT=/home/yc/unsb_tired/refactor/_runs/checkpoints
RES=/home/yc/unsb_tired/refactor/_runs/results
MET=/home/yc/unsb_tired/refactor/_runs/metrics
EVAL_PY=/home/yc/UNSB_Cov5/tools/evaluate_restoration.py
GPU=0

mkdir -p "$MET"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONDONTWRITEBYTECODE=1

eval_branch() {
  local name="$1"
  "$PY" test.py --dataroot "$DATA" --name "$name" --checkpoints_dir "$CKPT" --results_dir "$RES" \
    --epoch 200 --model sb --mode sb --dataset_mode unaligned --direction AtoB \
    --batch_size 1 --load_size 128 --crop_size 128 --preprocess resize_and_crop \
    --num_threads 0 --gpu_ids "$GPU" --num_test 240 \
    --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond \
    --normG instance --normD instance --num_timesteps 5 > "$MET/${name}_test.log" 2>&1

  "$PY" "$EVAL_PY" --manifest "$DATA/manifest.csv" \
    --fake-dir "$RES/$name/test_200/images/fake_5" --split test \
    --out "$MET/$name" --device cuda >> "$MET/${name}_eval.log" 2>&1
}

eval_branch dtcov_clean_plain_e200
eval_branch dtcov_clean_best_e200
echo "DONE" > "$MET/eval_both.done"
