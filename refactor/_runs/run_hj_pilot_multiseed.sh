#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yc/unsb_tired/refactor/baseline
PY=/home/yc/anaconda3/envs/unsb_cov/bin/python
TRAIN_DATA=/home/yc/UNSB_Patch/datasets/final6_train160_test40_unpaired
VALO=/home/yc/UNSB_Patch/datasets/final6train_valO5x16_offset560_unpaired
CKPT=/home/yc/unsb_tired/refactor/_runs/checkpoints
RES=/home/yc/unsb_tired/refactor/_runs/results_hj_ms
MET=/home/yc/unsb_tired/refactor/_runs/metrics_hj_ms
EVAL_PY=/home/yc/UNSB_Cov5/tools/evaluate_restoration.py
GPU=0
SEEDS=("${@:-2027 2028}")

mkdir -p "$MET"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1

COMMON=(--dataroot "$TRAIN_DATA" --checkpoints_dir "$CKPT" --mode sb --dataset_mode unaligned --direction AtoB
  --lambda_SB 1.0 --lambda_NCE 1.0 --tau 0.01 --batch_size 16 --load_size 128 --crop_size 128
  --preprocess resize_and_crop --num_threads 4 --gpu_ids "$GPU" --n_epochs_decay 0
  --lr 0.0001 --save_latest_freq 4800 --print_freq 160 --display_freq 1600 --display_id -1 --no_html
  --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond --normG instance --normD instance
  --pool_size 0 --num_timesteps 5 --no_flip)

train_plain() {
  local s="$1"; local name="hj_plain_s$s"
  "$PY" train.py "${COMMON[@]}" --seed "$s" --name "$name" --model sb --n_epochs 200 --save_epoch_freq 50 \
    --continue_train --epoch latest --pretrained_name dtcov_clean_warmup_e20 \
    > "$MET/${name}_train.log" 2>&1
}

train_hj() {
  local s="$1"; local control="$2"; local name="hj_${control}_s$s"
  "$PY" train.py "${COMMON[@]}" --seed "$s" --name "$name" --model hj --n_epochs 200 --save_epoch_freq 50 \
    --continue_train --epoch latest --pretrained_name dtcov_clean_warmup_e20 \
    --hj_enable True --hj_layers 0 --hj_start_epoch 5 --hj_probe_mode central_consensus \
    --hj_strength 0.5 --hj_boundary_scale 0.001 --hj_min_risk 0.05 \
    --hj_direction joint --hj_scales 1,2,4 --hj_gate_quantile 0.75 --hj_control "$control" \
    > "$MET/${name}_train.log" 2>&1
}

eval_hj() {
  local name="$1"; local s="$2"
  "$PY" test.py --dataroot "$VALO" --name "$name" --checkpoints_dir "$CKPT" --results_dir "$RES" \
    --epoch 200 --model sb --mode sb --dataset_mode unaligned --direction AtoB \
    --batch_size 1 --load_size 128 --crop_size 128 --preprocess resize_and_crop \
    --num_threads 0 --gpu_ids "$GPU" --num_test 80 --serial_batches --no_flip --seed "$s" \
    --netG resnet_9blocks_cond --netD basic_cond --netE basic_cond \
    --normG instance --normD instance --num_timesteps 5 > "$MET/${name}_test.log" 2>&1
  "$PY" "$EVAL_PY" --manifest "$VALO/manifest.csv" \
    --fake-dir "$RES/$name/test_200/images/fake_5" --split test \
    --out "$MET/$name" --device cuda >> "$MET/${name}_eval.log" 2>&1
}

for s in "${SEEDS[@]}"; do
  echo "=== [$s] plain ==="; train_plain "$s"
  echo "=== [$s] true ==="; train_hj "$s" true
  echo "=== [$s] roll ==="; train_hj "$s" roll
  echo "=== [$s] eval ==="
  eval_hj "hj_plain_s$s" "$s"
  eval_hj "hj_true_s$s" "$s"
  eval_hj "hj_roll_s$s" "$s"
done

echo "=== results (val-O overall PSNR) ==="
for s in "${SEEDS[@]}"; do
  "$PY" - "$s" "$MET" <<'PYEOF'
import json, sys, os
s, met = sys.argv[1], sys.argv[2]
def psnr(name):
    with open(os.path.join(met, name, "metrics_summary.json")) as f:
        return json.load(f)["summary"]["overall"]["psnr"]
p, t, r = psnr(f"hj_plain_s{s}"), psnr(f"hj_true_s{s}"), psnr(f"hj_roll_s{s}")
print(f"seed={s} plain={p:.4f} true={t:.4f} roll={r:.4f} true-plain={t-p:+.4f} true-roll={t-r:+.4f}")
PYEOF
done
echo "DONE" > "$MET/hj_pilot.done"
