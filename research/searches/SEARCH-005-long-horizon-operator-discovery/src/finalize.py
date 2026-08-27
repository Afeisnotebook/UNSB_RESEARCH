"""Build SEARCH-005's immutable route-1 ledger and honest weak-fallback handoff."""

from __future__ import annotations

import json
import os
from pathlib import Path


SEARCH_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = SEARCH_ROOT / "artifacts"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def result_for(candidate_id: str) -> tuple[dict | None, str | None]:
    small = ARTIFACTS / "SMALL_VIEW_RESULTS" / f"{candidate_id}.json"
    micro = ARTIFACTS / "MICRO_RESULTS" / f"{candidate_id}.json"
    if small.is_file():
        return read(small), "small_view_2400"
    if micro.is_file():
        return read(micro), "micro_800"
    return None, None


def summary(result: dict | None) -> dict | None:
    if result is None:
        return None
    trajectory = result["trajectory"]
    late = trajectory[-3:]
    absolute = [float(row["macro_psnr"]) for row in trajectory]
    return {
        "completed_horizon": int(trajectory[-1]["step"]),
        "late_available_mean_delta": sum(
            float(row["macro_psnr_delta"]) for row in late
        ) / len(late),
        "final_delta": float(trajectory[-1]["macro_psnr_delta"]),
        "final_positive_domains": int(trajectory[-1]["positive_domains"]),
        "final_worst_domain_delta": float(trajectory[-1]["worst_domain_delta"]),
        "final_ssim_delta": float(trajectory[-1]["macro_ssim_delta"]),
        "absolute_peak_to_final_rollback": max(absolute) - absolute[-1],
        "guardrails_pass_at_final": bool(trajectory[-1]["guardrails_pass"]),
    }


def main() -> int:
    cards = {
        path.stem: read(path)
        for path in sorted((ARTIFACTS / "DERIVATION_CARDS").glob("*.json"))
    }
    gates = {
        path.stem: read(path)
        for path in sorted((ARTIFACTS / "ENGINEERING_GATES").glob("*.json"))
    }
    decision_file = read(ARTIFACTS / "CANDIDATE_DECISIONS.json")
    decisions = {
        row["candidate_id"]: row for row in decision_file["decisions"]
    }

    ledger_entries = []
    ranked = []
    atlas = []
    for candidate_id, card in cards.items():
        result, stage = result_for(candidate_id)
        row_summary = summary(result)
        entry = {
            "candidate_id": candidate_id,
            "generation": card.get("generation"),
            "parent_probes": card.get("parent_probes", []),
            "causal_failure_class": card.get("causal_failure_class"),
            "mathematical_update": card.get("mathematical_update"),
            "identity_condition": card.get("identity_condition"),
            "paired_target_access": card.get("paired_target_access", False),
            "uses_fixed_window": card.get("uses_fixed_window", False),
            "engineering_gate": gates.get(candidate_id),
            "experiment_stage": stage,
            "experiment_summary": row_summary,
            "decision": decisions.get(candidate_id),
        }
        ledger_entries.append(entry)
        if result is not None:
            ranked.append({
                "candidate_id": candidate_id,
                "stage": stage,
                **row_summary,
                "trajectory": result["trajectory"],
            })
            previous_sign = None
            for point in result["trajectory"]:
                sign = "positive" if point["macro_psnr_delta"] > 0 else "negative"
                atlas.append({
                    "candidate_id": candidate_id,
                    "source": "SEARCH-005-matched-local",
                    "step": int(point["step"]),
                    "macro_psnr_delta": float(point["macro_psnr_delta"]),
                    "macro_ssim_delta": float(point["macro_ssim_delta"]),
                    "positive_domains": int(point["positive_domains"]),
                    "worst_domain_delta": float(point["worst_domain_delta"]),
                    "sign": sign,
                    "sign_changed_from_previous_observation": (
                        previous_sign is not None and sign != previous_sign
                    ),
                    "paired_metric_used_for_control": False,
                })
                previous_sign = sign

    historical = read(ARTIFACTS / "HISTORICAL_COORDINATE_REANALYSIS.json")
    previous_sign = None
    for step, delta in historical["full100_hnek_macro_psnr_delta"].items():
        sign = "positive" if delta > 0 else "negative"
        atlas.append({
            "candidate_id": "historical_hnek_anchor",
            "source": "SEARCH-002-full100-reanalysis",
            "step": int(step),
            "macro_psnr_delta": float(delta),
            "sign": sign,
            "sign_changed_from_previous_observation": (
                previous_sign is not None and sign != previous_sign
            ),
            "paired_metric_used_for_control": False,
        })
        previous_sign = sign

    ranked.sort(
        key=lambda row: (
            int(row["completed_horizon"]),
            float(row["late_available_mean_delta"]),
            float(row["final_delta"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        row["promotion_passed"] = False
        row["promotion_failure"] = (
            "no candidate satisfied positive final, domain coverage, worst-domain, "
            "SSIM and absolute-retention gates at the registered long horizon"
        )

    write(ARTIFACTS / "HYPOTHESIS_LEDGER.json", {
        "schema": "clean-unsb-search005-hypothesis-ledger-v1",
        "route": "route1_mathematical_operator_discovery",
        "entries": ledger_entries,
        "generation1_mechanisms": 6,
        "causal_revisions": 4,
        "promotion_count": 0,
        "route1_exhausted_under_registered_cap": True,
        "confirmation20_opened": False,
    })
    atlas_path = ARTIFACTS / "REVERSAL_ATLAS.jsonl"
    atlas_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in atlas),
        encoding="utf-8",
    )
    write(ARTIFACTS / "LOCAL_RANKING.json", {
        "schema": "clean-unsb-search005-local-ranking-v1",
        "ranking": ranked,
        "rule": "completed registered horizon, then late-available mean delta, then final delta; promotion gates remain mandatory",
        "all_not_promoted": True,
        "confirmation20_opened": False,
    })

    pcoa = next(row for row in ranked if row["candidate_id"] == "G1-GAME-PCOA")
    pcoa_card = cards["G1-GAME-PCOA"]
    candidate = {
        "schema": "clean-unsb-search005-candidate-v1",
        "candidate_id": "G1-GAME-PCOA",
        "name": pcoa_card["name"],
        "classification": "weak_fallback",
        "promotion_passed": False,
        "why_frozen": "only route-1 operator to remain positive at 400, 800 and 1200, and the only new operator completed at 2400; it nevertheless reverses at 1600 and 2400",
        "claim_status": "not_a_sustained_algorithm_result",
        "code": [
            "research/searches/SEARCH-005-long-horizon-operator-discovery/src/model_operators.py",
            "research/searches/SEARCH-005-long-horizon-operator-discovery/src/operators.py"
        ],
        "mathematical_update": pcoa_card["mathematical_update"],
        "configuration": {
            "seed": 2026,
            "train_per_domain": 25,
            "discovery_per_domain": 10,
            "target_steps": 2400,
            "tuned_hyperparameters": [],
            "paired_target_access": False,
            "uses_fixed_window": False,
        },
        "trajectory": pcoa["trajectory"],
        "late_available_mean_delta": pcoa["late_available_mean_delta"],
        "final_delta": pcoa["final_delta"],
        "absolute_peak_to_final_rollback": pcoa["absolute_peak_to_final_rollback"],
        "risk": [
            "single seed and discovery10 only",
            "negative at 1600 and 2400 with large absolute rollback",
            "2000-step relative gain coincides with matched-plain collapse",
            "its norm-preserving Generation-2 revision failed by 800",
            "not eligible for a paper claim or automatic 4090 promotion"
        ],
        "local_reproduction": "python research/searches/SEARCH-005-long-horizon-operator-discovery/run_search.py --stage small",
        "conditional_4090_command": "python research/searches/SEARCH-005-long-horizon-operator-discovery/run_search.py --stage full --gpu 0 --seed 2026 --full-steps 12000 --full-eval 1000 2000 3000 4000 6000 8000 10000 12000",
        "4090_condition": "run only if explicitly accepting that this is a weak fallback, not a locally sustained winner",
        "confirmation20_opened": False,
    }
    write(ARTIFACTS / "CANDIDATE.json", candidate)
    write(ARTIFACTS / "BACKUP_CANDIDATES.json", {
        "schema": "clean-unsb-search005-backups-v1",
        "backups": [
            {
                "rank": 2,
                "candidate_id": "G1-DT-CNDRP",
                "classification": "near_neutral_micro_backup",
                "evidence": "800-step delta -0.0045 dB with 4/6 positive domains; SSIM and worst-domain guards fail; its G2 block revision is worse",
            },
            {
                "rank": 3,
                "candidate_id": "historical_hnek_anchor",
                "classification": "long_horizon_oscillatory_probe",
                "evidence": "full100 final delta +0.0556 dB and 8k/10k/12k mean +0.0063 dB, but repeated sign changes and no smooth retention",
            },
        ],
        "all_not_promoted": True,
        "confirmation20_opened": False,
    })
    write(ARTIFACTS / "ROUTE1_STOP.json", {
        "schema": "clean-unsb-search005-route1-stop-v1",
        "status": "complete_no_sustained_candidate",
        "answer": "target-blind positive windows exist, but no tested self-null, invariant, unbiased or coupled-dynamics operator retained them through the registered local horizon",
        "failure_class": "signals_found_but_no_safe_long_horizon_correction",
        "not_claimed": [
            "that DT, HJ or HNEK contain no signal",
            "that an exit threshold would solve route 1",
            "that PCOA is a sustained winner",
            "that confirmation20 or multiple seeds were tested"
        ],
        "full_view_not_started_reason": "zero SEARCH-005 candidates passed the 2400-step promotion gate",
        "seed_validation_not_started_reason": "zero candidates reached full-view promotion",
        "next_route": "a separately approved route-2 gap-aware handoff or a genuinely new failure class; do not silently continue threshold search inside SEARCH-005",
        "confirmation20_opened": False,
    })

    table_rows = []
    for row in ranked:
        table_rows.append(
            f"| {row['rank']} | {row['candidate_id']} | {row['completed_horizon']} | "
            f"{row['late_available_mean_delta']:+.3f} | {row['final_delta']:+.3f} | "
            f"{row['final_positive_domains']}/6 | no |"
        )
    report = """# SEARCH-005 final route-1 report

## Outcome

No tested mathematical operator produced a sustained local win. Positive windows are real and reproducible, but every admissible mechanism either failed by 800 updates or reversed by the independent 2400-update trajectory. No full-view, second-seed or confirmation20 experiment was opened.

The frozen weak fallback is **G1-GAME-PCOA**, not because it passed, but because it was the only new route-1 operator positive at 400/800/1200 and the only one that completed 2400. It reversed at 1600 (-0.570 dB) and 2400 (-0.861 dB), so it cannot support a sustained-method claim.

## Ranking

| Rank | Candidate | Horizon | Late available mean ΔPSNR | Final ΔPSNR | Final positive domains | Promoted |
|---:|---|---:|---:|---:|---:|:---:|
""" + "\n".join(table_rows) + """

## Causal conclusions

- DT-style sensitivity preconditioning can be nearly neutral but does not preserve structural quality; the block-safe revision worsened late PSNR.
- HJ-style projected correction produces a strong early window, but independent future-batch consensus removes rather than stabilizes the benefit.
- HNEK-style physical coordinate defects are not sufficient predictors: direct path correction damages content and the native gradient often already reduces the measured defect.
- Coupled-game optimism changes the phase and yields reproducible positive windows. Removing radial amplification strengthens 400-step quality but causes a larger 800-step reversal, so predictable angular motion alone is not a safe long-run correction.
- Historical HNEK repeatedly changes sign through 12k. This is evidence for an oscillatory endogenous training distribution, not evidence for a single correct exit point.

## Honest boundary

SEARCH-005 exhausted its registered six Generation-1 mechanisms and one causal revision per failure class. It did not drift into fixed-window, paired-PSNR, whole-branch selection or handoff optimization. A route-2 handoff study would need a separate plan and claim.
"""
    (SEARCH_ROOT / "RESULTS.md").write_text(report, encoding="utf-8")
    print(
        f"finalized candidates={len(ranked)} atlas_rows={len(atlas)} "
        f"fallback={candidate['candidate_id']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
