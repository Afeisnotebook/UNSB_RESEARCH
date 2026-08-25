"""Frozen lane definitions and target-blind ranking rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


NEW_MECHANISMS = ("lbst", "ptq", "dcum", "aeb")
LEGACY_PROBES = ("dt_anchor", "hj_anchor", "hnek_anchor")


@dataclass(frozen=True)
class LaneSpec:
    name: str
    model: str = "sb"
    mechanisms: tuple[str, ...] = ()
    family: str = "plain"
    estimated_g_flops_multiplier: float = 1.0

    def to_dict(self) -> dict:
        value = asdict(self)
        value["mechanisms"] = list(self.mechanisms)
        return value


def frozen_lanes() -> list[LaneSpec]:
    return [
        LaneSpec("plain"),
        LaneSpec("dt_anchor", model="dtcov", family="legacy"),
        LaneSpec("hj_anchor", model="hj", family="legacy"),
        LaneSpec("hnek_anchor", model="hnek_search", family="legacy"),
        LaneSpec("lbst", mechanisms=("lbst",), family="new"),
        LaneSpec("ptq", mechanisms=("ptq",), family="new"),
        LaneSpec("dcum", mechanisms=("dcum",), family="new"),
        LaneSpec(
            "aeb", mechanisms=("aeb",), family="new",
            estimated_g_flops_multiplier=2.0,
        ),
    ]


def synthesize(name: str, left: LaneSpec, right: LaneSpec) -> LaneSpec:
    """Combine compatible mechanisms while preserving the legacy model owner."""
    if left.model != "sb" and right.model != "sb" and left.model != right.model:
        raise ValueError("two different legacy model owners cannot be combined")
    model = left.model if left.model != "sb" else right.model
    mechanisms = tuple(dict.fromkeys(left.mechanisms + right.mechanisms))
    if not mechanisms:
        raise ValueError("a synthesis needs at least one new mechanism")
    multiplier = 2.0 if "aeb" in mechanisms else 1.0
    return LaneSpec(
        name=name,
        model=model,
        mechanisms=mechanisms,
        family="synthesis",
        estimated_g_flops_multiplier=multiplier,
    )


def score_row(summary: dict) -> tuple:
    """Frozen ordering: late delta, final delta, coverage, worst, rollback."""
    trajectory = summary.get("trajectory", [])
    late = trajectory[-3:] if trajectory else []
    late_delta = sum(float(row["macro_psnr_delta"]) for row in late) / max(len(late), 1)
    final = trajectory[-1] if trajectory else {}
    return (
        late_delta,
        float(final.get("macro_psnr_delta", float("-inf"))),
        int(final.get("positive_domains", -1)),
        float(final.get("worst_domain_delta", float("-inf"))),
        -float(summary.get("peak_to_final_rollback", float("inf"))),
    )


def ranked(summaries: Iterable[dict]) -> list[dict]:
    return sorted(summaries, key=score_row, reverse=True)


def classify(winner: dict, all_rows: list[dict]) -> str:
    final = winner["trajectory"][-1]
    late_positive = all(
        float(row["macro_psnr_delta"]) > 0 for row in winner["trajectory"][-3:]
    )
    guardrails = bool(final.get("guardrails_pass", False))
    if (
        late_positive
        and float(final["macro_psnr_delta"]) > 0
        and int(final["positive_domains"]) >= 4
        and guardrails
    ):
        return "strong_local_signal"
    if float(final["macro_psnr_delta"]) > 0:
        only_aeb_positive = all(
            ("aeb" in row.get("spec", {}).get("mechanisms", []))
            for row in all_rows
            if row.get("trajectory")
            and float(row["trajectory"][-1]["macro_psnr_delta"]) > 0
        )
        if "aeb" in winner.get("spec", {}).get("mechanisms", []) and only_aeb_positive:
            return "compute_only_signal"
        return "positive_but_fragile"
    return "weak_fallback"
