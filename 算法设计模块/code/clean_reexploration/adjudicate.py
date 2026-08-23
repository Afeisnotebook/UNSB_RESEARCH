"""Mechanical, target-blind-*after-freeze* adjudicator.

This module is a pure function of the paired evaluator's raw evidence.  It only
computes the frozen mechanical labels from the task prompt; it never invents a
scientific interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path


DEVELOPMENT_GAIN_PSNR = 0.15
HANDOFF_GAIN_FULL = 0.10
HANDOFF_GAIN_PLAIN = 0.15
POSITIVE_DOMAINS_MIN = 3
PAIRED_DOMAIN_TOTAL = 5


def _method_label(entry: dict) -> str:
    delta = float(entry.get("delta_psnr", 0.0))
    lower = float(entry.get("delta_psnr_ci_low", 0.0))
    positive = int(entry.get("positive_domains", 0))
    if delta >= DEVELOPMENT_GAIN_PSNR and lower > 0 and positive >= POSITIVE_DOMAINS_MIN:
        return "DEVELOPMENT_GAIN"
    return "DEVELOPMENT_NO_GAIN"


def _handoff_label(handoff: dict, full: dict, plain: dict) -> str:
    delta_full = float(handoff.get("psnr_macro", 0.0)) - float(full.get("psnr_macro", 0.0))
    delta_plain = float(handoff.get("psnr_macro", 0.0)) - float(plain.get("psnr_macro", 0.0))
    lower_full = float(handoff.get("vs_full_ci_low", 0.0))
    lower_plain = float(handoff.get("vs_plain_ci_low", 0.0))
    positive = int(handoff.get("positive_domains_vs_plain", 0))
    if (
        delta_full >= HANDOFF_GAIN_FULL
        and delta_plain >= HANDOFF_GAIN_PLAIN
        and lower_full > 0
        and lower_plain > 0
        and positive >= POSITIVE_DOMAINS_MIN
    ):
        return "HANDOFF_OPTIMIZATION_GAIN"
    return "HANDOFF_OPTIMIZATION_NO_GAIN"


def adjudicate(evidence: dict) -> dict:
    """Return the mechanical summary from raw paired-evaluator evidence."""
    plain = evidence.get("canonical_plain", {})
    out: dict = {
        "schema_version": 1,
        "run_id": evidence.get("run_id"),
        "labels": {},
    }
    for method in ("dt", "hj"):
        entry = evidence.get(method)
        if entry:
            out["labels"][method] = _method_label(entry)

    full = evidence.get("hnek_full", {})
    handoff = evidence.get("hnek_handoff")
    if full and handoff:
        out["labels"]["hnek_handoff_vs_full"] = _handoff_label(handoff, full, plain)

    e50_handoff = evidence.get("hnek_e50_handoff")
    if full and e50_handoff:
        out["labels"]["hnek_e50_handoff_vs_full"] = _handoff_label(e50_handoff, full, plain)

    return out


def adjudicate_file(raw_path: Path, out_path: Path) -> dict:
    evidence = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    summary = adjudicate(evidence)
    out = Path(out_path)
    out.write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
