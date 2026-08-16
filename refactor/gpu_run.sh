#!/usr/bin/env bash
# 两个重构 agent 共享同一张 4090，用 flock 串行化所有 GPU 训练/评测。
# 用法: ./gpu_run.sh <tag> <command...>
# 例子: ./gpu_run.sh dt_repro /home/yc/anaconda3/envs/unsb_cov/bin/python train.py ...
set -euo pipefail
LOCK=/home/yc/unsb_tired/refactor/.gpu.lock
TAG="${1:?need a tag}"
shift

echo "[gpu_run:$TAG] waiting for GPU lock ..."
exec 9>"$LOCK"
flock -x 9
echo "[gpu_run:$TAG] acquired GPU lock $(date -Is)"
start=$(date +%s)
set +e
"$@"
rc=$?
set -e
end=$(date +%s)
echo "[gpu_run:$TAG] released GPU lock $(date -Is), rc=$rc, elapsed=$((end-start))s"
exit $rc
