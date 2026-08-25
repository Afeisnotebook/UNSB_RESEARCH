#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
SRC="$ROOT/src"
PY="${PY:-python3}"
WORK_ROOT="${UNSB_MOTIVATION_RUN_ROOT:-$REPO_ROOT/runs/MOT-001}"
RAW="$WORK_ROOT/raw"
FIG="$WORK_ROOT/figures"
REPORT_DIR="$WORK_ROOT/reports"
SUMMARY="$REPORT_DIR/MOTIVATION_SUMMARY.json"
REPORT="$REPORT_DIR/MOTIVATION_FIGURE_REPORT_CN.md"
LEDGER="$WORK_ROOT/CLAIM_LEDGER.json"
MANIFEST="$WORK_ROOT/MANIFEST.sha256"

mkdir -p "$FIG" "$REPORT_DIR"

say() { printf '[finalize %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [[ ! -s "$SUMMARY" || "${FORCE:-0}" == "1" ]]; then
  say "summarizing raw evidence"
  "$PY" "$SRC/summarize_motivation.py" \
    --raw-dir "$RAW" \
    --out "$SUMMARY" \
    --bridge-times 1,2,3
fi

say "plotting figures"
"$PY" "$SRC/plot_motivation.py" \
  --raw-dir "$RAW" \
  --out-dir "$FIG" \
  --bridge-times 1,2,3

say "adjudicating draft"
"$PY" "$SRC/adjudicate_motivation.py" \
  --summary "$SUMMARY" \
  --raw-dir "$RAW" \
  --report "$REPORT" \
  --ledger "$LEDGER"

say "writing MANIFEST.sha256"
{
  find "$RAW" -type f \( -name '*.jsonl' -o -name '*.npz' -o -name 'panel_b_pca.json' \) -print0 | sort -z | xargs -0 sha256sum
  find "$FIG" -type f -name '*.png' -print0 | sort -z | xargs -0 sha256sum
  [[ -f "$SUMMARY" ]] && sha256sum "$SUMMARY"
  [[ -f "$REPORT" ]] && sha256sum "$REPORT"
  [[ -f "$LEDGER" ]] && sha256sum "$LEDGER"
} > "$MANIFEST"

say "ALL_DONE"
