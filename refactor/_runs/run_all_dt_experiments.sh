#!/usr/bin/env bash
set -euo pipefail
bash /home/yc/unsb_tired/refactor/_runs/run_plain_baseline.sh
bash /home/yc/unsb_tired/refactor/_runs/run_dtcov_repro.sh
bash /home/yc/unsb_tired/refactor/_runs/run_eval_both.sh
/home/yc/anaconda3/envs/unsb_cov/bin/python /home/yc/unsb_tired/refactor/_runs/summarize_dt_comparison.py
