#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yc/unsb_tired/refactor/baseline
PY=/home/yc/anaconda3/envs/unsb_cov/bin/python
DATA=/home/yc/UNSB_Cov5/datasets/final6_train160_test40_unpaired
CKPT=/home/yc/unsb_tired/refactor/_runs/checkpoints
MET=/home/yc/unsb_tired/refactor/_runs/metrics_ms
EVAL_PY=/home/yc/UNSB_Cov5/tools/evaluate_restoration.py
GPU=0
SEEDS=("${@:-2027 2028}")

mkdir -p "$MET"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1

COMMON=(--dataroot "$DATA" --checkpoints_dir "$CKPT" --mode sb --dataset_mode unaligned --direction AtoB
  --lambda_SB 1.0 --lambda_NCE 1.0 --tau 0.01 --batch_size 16 --load_size 128 --crop_size 128
  --preprocess resize_and_crop --num_threads 4 --gpu_ids "$GPU" --n_epochs_decay 0
  --lr 0.0001 --save_latest_freq 5000 --print_freq 100 --display_id -1 --no_html
  --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond --normG instance --normD instance
  --pool_size 0 --num_timesteps 5)

run_seed() {
  local s="$1"
  local plain="dtcov_plain_s$s"
  local best="dtcov_best_s$s"

  echo "=== [$s] plain to 200 ==="
  "$PY" train.py "${COMMON[@]}" --seed "$s" --name "$plain" --model sb --n_epochs 200 \
    --save_epoch_freq 50 --continue_train --epoch latest --pretrained_name dtcov_clean_warmup_e20 \
    > "$MET/${plain}_train.log" 2>&1

  echo "=== [$s] DT window 25 ==="
  "$PY" train.py "${COMMON[@]}" --seed "$s" --name "$best" --model dtcov --n_epochs 25 --save_epoch_freq 10 \
    --continue_train --epoch latest --pretrained_name dtcov_clean_warmup_e20 \
    --dtcov_lambda 0.001 --dtcov_lambda_schedule ramp_hold_cosine_decay \
    --dtcov_ramp_start_epoch 1 --dtcov_ramp_end_epoch 5 \
    --dtcov_decay_start_epoch 15 --dtcov_decay_end_epoch 25 --dtcov_lambda_min 0.0 \
    --dtcov_m 4 --dtcov_warmup_iters 300 --dtcov_time_mode actual \
    > "$MET/${best}_train.log" 2>&1

  echo "=== [$s] DT continue plain to 200 ==="
  "$PY" train.py "${COMMON[@]}" --seed "$s" --name "$best" --model dtcov --n_epochs 200 --epoch_count 26 \
    --save_epoch_freq 50 --continue_train --epoch latest --dtcov_lambda 0 \
    >> "$MET/${best}_train.log" 2>&1

  for n in "$plain" "$best"; do
    echo "=== [$s] eval $n ==="
    "$PY" test.py --dataroot "$DATA" --name "$n" --checkpoints_dir "$CKPT" --results_dir "$MET/results" \
      --epoch 200 --model sb --mode sb --dataset_mode unaligned --direction AtoB \
      --batch_size 1 --load_size 128 --crop_size 128 --preprocess resize_and_crop \
      --num_threads 0 --gpu_ids "$GPU" --num_test 240 --serial_batches --seed "$s" \
      --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond \
      --normG instance --normD instance --num_timesteps 5 > "$MET/${n}_test.log" 2>&1
    "$PY" "$EVAL_PY" --manifest "$DATA/manifest.csv" \
      --fake-dir "$MET/results/$n/test_200/images/fake_5" --split test \
      --out "$MET/$n" --device cuda >> "$MET/${n}_eval.log" 2>&1
  done
}

for s in "${SEEDS[@]}"; do
  run_seed "$s"
done

echo "=== deltas vs plain ==="
for s in "${SEEDS[@]}"; do
  "$PY" - "$s" "$MET" <<'PYEOF'
import json, sys, os
s, met = sys.argv[1], sys.argv[2]
def psnr(name):
    with open(os.path.join(met, name, "metrics_summary.json")) as f:
        return json.load(f)["summary"]["overall"]["psnr"]
p = psnr(f"dtcov_plain_s{s}")
d = psnr(f"dtcov_best_s{s}")
print(f"seed={s} plain={p:.4f} dt={d:.4f} delta={d-p:+.4f}")
PYEOF
done
echo "DONE" > "$MET/pilot.done"
