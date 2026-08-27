"""Freeze SEARCH-003 local evidence, ranking and honest stopping decision."""

from __future__ import annotations

import json
from pathlib import Path

from .ledger import HypothesisLedger
from .protocol import promotion_decision
from .search001_compat import modules


EVAL_STEPS = (400, 800, 1200, 1600, 2000, 2400)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _trajectory(lane: Path, plain: Path, steps=EVAL_STEPS) -> list[dict]:
    compare = modules()[2].compare
    result = []
    for step in steps:
        method_path = lane / f"metrics_step_{step}.json"
        plain_path = plain / f"metrics_step_{step}.json"
        if method_path.is_file() and plain_path.is_file():
            result.append(compare(_read(method_path), _read(plain_path), step=step))
    return result


def _late_mean(trajectory: list[dict]) -> float:
    rows = trajectory[-3:]
    return sum(float(row["macro_psnr_delta"]) for row in rows) / len(rows)


def _absolute_rollback(trajectory: list[dict]) -> float:
    values = [float(row["macro_psnr"]) for row in trajectory]
    rolling = [sum(values[index - 2:index + 1]) / 3 for index in range(2, len(values))]
    return max(rolling) - sum(values[-3:]) / 3 if rolling else 0.0


def freeze_local_report(output_dir: Path, small_dir: Path, repo_root: Path) -> dict:
    output_dir = Path(output_dir)
    small_dir = Path(small_dir)
    plain_dir = small_dir / "plain"
    lanes = {
        "G2-HJ-FBDFC8::proposal_only": small_dir / "G2-HJ-FBDFC8__proposal_only",
        "G2-HJ-FBDFC8::full": small_dir / "G2-HJ-FBDFC8__full",
        "G1-HJ-RHDFC8::full": small_dir / "G1-HJ-RHDFC8__full",
        "G2-DT-FBGC8::full": small_dir / "G2-DT-FBGC8__full",
        "G1-DT-RHGC8::full": small_dir / "G1-DT-RHGC8__full",
    }
    summaries = []
    per_domain = {}
    for name, lane in lanes.items():
        trajectory = _trajectory(lane, plain_dir)
        if not trajectory:
            continue
        long_complete = {row["step"] for row in trajectory} >= set(EVAL_STEPS)
        gate = promotion_decision(trajectory) if long_complete else {
            "promote": False,
            "reasons": ["incomplete_2400_trajectory"],
        }
        summary = {
            "name": name,
            "trajectory": trajectory,
            "last_step": int(trajectory[-1]["step"]),
            "late_available_mean_delta": _late_mean(trajectory),
            "absolute_rolling_rollback": _absolute_rollback(trajectory),
            "promotion": gate,
        }
        summaries.append(summary)
        per_domain[name] = [
            {"step": row["step"], "domain_delta": row["domain_delta"]}
            for row in trajectory
        ]
    summaries.sort(
        key=lambda row: (
            row["promotion"].get("promote", False),
            row["last_step"],
            row["late_available_mean_delta"],
            row["trajectory"][-1]["macro_psnr_delta"],
        ),
        reverse=True,
    )
    winner = summaries[0]
    if winner["promotion"]["promote"]:
        classification = "route1_sustained_local"
        recommendation = "eligible_for_full_view"
    else:
        classification = "weak_fallback"
        recommendation = "not_eligible_for_full_view_or_4090"

    decomposition = {
        "plain": [
            {
                "step": step,
                "macro_psnr": _read(plain_dir / f"metrics_step_{step}.json")["macro_psnr"],
                "macro_ssim": _read(plain_dir / f"metrics_step_{step}.json")["macro_ssim"],
            }
            for step in EVAL_STEPS
        ],
        "lanes": {
            row["name"]: [
                {
                    "step": item["step"],
                    "candidate_macro_psnr": item["macro_psnr"],
                    "plain_macro_psnr": item["plain_macro_psnr"],
                    "macro_psnr_delta": item["macro_psnr_delta"],
                    "macro_ssim_delta": item["macro_ssim_delta"],
                }
                for item in row["trajectory"]
            ]
            for row in summaries
        },
        "confirmation20_opened": False,
    }
    (output_dir / "LOCAL_RANKING.json").write_text(
        json.dumps({
            "schema": "clean-unsb-search003-local-ranking-v1",
            "ranking": summaries,
            "rule": (
                "promotion first, then completed horizon, late available mean delta, "
                "final delta; incomplete lanes cannot displace a completed lane"
            ),
            "confirmation20_opened": False,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "PER_DOMAIN_TRAJECTORY.json").write_text(
        json.dumps({
            "schema": "clean-unsb-search003-domain-trajectory-v1",
            "lanes": per_domain,
            "confirmation20_opened": False,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ABSOLUTE_RELATIVE_DECOMPOSITION.json").write_text(
        json.dumps(decomposition, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ledger = HypothesisLedger(output_dir / "HYPOTHESIS_LEDGER.json")
    transitions = {
        "G2-DT-FBGC8": {
            "status": "closed_second_generation_failure",
            "evidence": "identical to G1 DT at 400/800; final delta -0.381266",
        },
        "G2-HJ-FBDFC8": {
            "status": "closed_second_generation_failure",
            "evidence": (
                "full selector passed 400/800 but failed late promotion; "
                "late-three mean -0.431388 and final -0.606743"
            ),
        },
    }
    for hypothesis_id, transition in transitions.items():
        if any(
            entry["id"] == hypothesis_id and entry["status"] == "proposed"
            for entry in ledger.entries
        ):
            ledger.transition(
                hypothesis_id,
                status=transition["status"],
                evidence={"summary": transition["evidence"]},
            )

    candidate = {
        "schema": "clean-unsb-search003-candidate-v1",
        "candidate_id": "G2-HJ-FBDFC8",
        "variant": "proposal_only",
        "classification": classification,
        "promotion_passed": bool(winner["promotion"].get("promote", False)),
        "recommendation": recommendation,
        "why_frozen": (
            "best completed 2400-step lane under the frozen ranking, but it violates "
            "absolute-retention and/or guardrail conditions"
        ),
        "code": (
            "research/searches/SEARCH-003-evidence-guided-discovery/src/receding.py"
        ),
        "configuration": {
            "parent_probe": "hj",
            "mode": "proposal_only",
            "horizon": 8,
            "seed": 2026,
            "train_per_domain": 25,
            "target_steps": 2400,
        },
        "trajectory": winner["trajectory"],
        "promotion": winner["promotion"],
        "compute": "one HJ proposal branch per committed update; no selector overhead",
        "risk": [
            "single seed and discovery10 only",
            "negative 1600-step delta and large absolute rolling rollback",
            "late positive points partly coincide with matched-plain collapse",
            "not approved for full-view or 4090 validation under SEARCH-003 gates",
        ],
        "local_reproduction": (
            "E:\\conda\\python.exe research\\searches\\"
            "SEARCH-003-evidence-guided-discovery\\run_search.py --stage candidate "
            "--candidate G2-HJ-FBDFC8 --candidate-mode proposal_only --candidate-steps 2400"
        ),
        "confirmation20_opened": False,
    }
    (output_dir / "CANDIDATE.json").write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    backups = [
        {
            "rank": 2,
            "candidate_id": "G2-HJ-FBDFC8",
            "variant": "full",
            "reason_not_selected": "late-three mean and final delta are negative",
        },
        {
            "rank": 3,
            "candidate_id": "G2-DT-FBGC8",
            "variant": "full",
            "reason_not_selected": "second gate changed no decisions; 800 delta is negative",
        },
    ]
    (output_dir / "BACKUP_CANDIDATES.json").write_text(
        json.dumps({
            "schema": "clean-unsb-search003-backups-v1",
            "backups": backups,
            "all_not_promoted": True,
            "confirmation20_opened": False,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stop = {
        "schema": "clean-unsb-search003-route1-stop-v1",
        "route1_sustained_candidate_found": False,
        "reason": (
            "target-blind local UNSB loss/gradient consistency delayed or reshaped "
            "reversal but did not ensure sustained paired restoration quality"
        ),
        "honest_stop_category": (
            "correction_valid_on_unpaired_native_objective_but_not_equivalent_to_psnr"
        ),
        "third_generation_allowed": False,
        "full_view_started": False,
        "seed_2027_started": False,
        "confirmation20_opened": False,
        "next_route": "plan gap-aware handoff separately; do not mutate SEARCH-003",
    }
    (output_dir / "ROUTE1_STOP.json").write_text(
        json.dumps(stop, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "winner": candidate,
        "ranking": summaries,
        "backups": backups,
        "stop": stop,
    }
