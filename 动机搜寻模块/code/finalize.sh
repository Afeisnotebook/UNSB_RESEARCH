#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
RAW="$ROOT/raw"
FIG="$ROOT/figures"
REPORT_DIR="$ROOT/reports"
SUMMARY="$REPORT_DIR/MOTIVATION_SUMMARY.json"
REPORT="$REPORT_DIR/MOTIVATION_FIGURE_REPORT_CN.md"
LEDGER="$ROOT/CLAIM_LEDGER.json"
MANIFEST="$ROOT/MANIFEST.sha256"

mkdir -p "$FIG" "$REPORT_DIR"

say() { printf '[finalize %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [[ ! -s "$SUMMARY" || "${FORCE:-0}" == "1" ]]; then
  say "summarizing raw evidence"
  "$PY" "$ROOT/code/summarize_motivation.py" \
    --raw-dir "$RAW" \
    --out "$SUMMARY" \
    --bridge-times 1,2,3
fi

say "plotting figures"
"$PY" "$ROOT/code/plot_motivation.py" \
  --raw-dir "$RAW" \
  --out-dir "$FIG" \
  --bridge-times 1,2,3

say "adjudicating draft"
"$PY" "$ROOT/code/adjudicate_motivation.py" \
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
