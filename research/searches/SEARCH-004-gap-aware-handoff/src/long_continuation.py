"""Registered long matched continuation for promoted route-2 transports."""

from __future__ import annotations

from pathlib import Path

from .analyze import domain_delta, metric_at, source_metrics
from .engine import HandoffEngine, atomic_json


def _late_mean(values: list[float], count: int = 3) -> float:
    selected = values[-min(len(values), int(count)):]
    return sum(selected) / len(selected)


def analyze_long(
    *, checkpoint, results: dict[str, dict], evaluation_horizons: tuple[int, ...]
) -> dict:
    if "P_common_plain" not in results or "A_hard_disable" not in results:
        raise RuntimeError("long continuation requires matched P and A controls")
    plain = results["P_common_plain"]
    hard = results["A_hard_disable"]
    source_method, _ = source_metrics(checkpoint)
    rows = []
    for arm, result in sorted(results.items()):
        trajectory = []
        for horizon in evaluation_horizons:
            metric = metric_at(result, horizon)
            p_metric = metric_at(plain, horizon)
            a_metric = metric_at(hard, horizon)
            if metric is None or p_metric is None or a_metric is None:
                raise RuntimeError(f"missing long metric: {arm}/h{horizon}")
            deltas = domain_delta(metric, p_metric)
            trajectory.append({
                "horizon": int(horizon),
                "macro_psnr": float(metric["macro_psnr"]),
                "macro_ssim": float(metric["macro_ssim"]),
                "macro_lpips": metric.get("macro_lpips"),
                "delta_plain": float(metric["macro_psnr"] - p_metric["macro_psnr"]),
                "lift_hard": float(metric["macro_psnr"] - a_metric["macro_psnr"]),
                "positive_domains": sum(value > 0.0 for value in deltas.values()),
                "worst_domain": min(deltas.values()),
                "domain_delta": deltas,
            })
        final = trajectory[-1]
        final_metric = metric_at(result, evaluation_horizons[-1])
        final_plain = metric_at(plain, evaluation_horizons[-1])
        late_delta = _late_mean([row["delta_plain"] for row in trajectory])
        late_absolute = _late_mean([row["macro_psnr"] for row in trajectory])
        late_peak = max(row["macro_psnr"] for row in trajectory[-3:])
        late_rollback = float(late_peak - final["macro_psnr"])
        transport = result.get("transport_record") or {}
        guardrails = {
            "late_positive": late_delta > 0.0,
            "final_positive": final["delta_plain"] > 0.0,
            "coverage": final["positive_domains"] >= 4,
            "worst_domain": final["worst_domain"] > -1.0,
            "ssim": float(final_metric["macro_ssim"]) >= float(final_plain["macro_ssim"]),
            "lpips": (
                final_metric.get("macro_lpips") is not None
                and final_plain.get("macro_lpips") is not None
                and float(final_metric["macro_lpips"]) <= float(final_plain["macro_lpips"])
            ),
            "absolute_late_rollback": late_rollback <= 0.30,
        }
        rows.append({
            "checkpoint_id": checkpoint.checkpoint_id,
            "arm": arm,
            "trajectory": trajectory,
            "late_three_delta_plain": late_delta,
            "late_three_absolute_psnr": late_absolute,
            "final_delta_plain": final["delta_plain"],
            "final_lift_hard": final["lift_hard"],
            "final_positive_domains": final["positive_domains"],
            "final_worst_domain": final["worst_domain"],
            "absolute_final_change_from_source": (
                float(final["macro_psnr"]) - float(source_method["macro_psnr"])
            ),
            "absolute_late_peak_psnr": float(late_peak),
            "absolute_peak_to_final_rollback": late_rollback,
            "transport_identity": transport.get("identity"),
            "target_blind_defect_reduction": transport.get(
                "target_blind_defect_reduction"
            ),
            "guardrails": guardrails,
            "passes_long": all(guardrails.values()),
        })
    ranked = sorted(
        rows,
        key=lambda row: (
            row["passes_long"], row["late_three_delta_plain"],
            row["final_delta_plain"], row["final_positive_domains"],
            row["final_worst_domain"],
        ),
        reverse=True,
    )
    return {
        "schema": "clean-unsb-search004-long-analysis-v1",
        "checkpoint": checkpoint.to_dict(),
        "evaluation_horizons": list(evaluation_horizons),
        "rows": rows,
        "ranking": [row["arm"] for row in ranked],
        "winner": ranked[0]["arm"],
        "confirmation20_opened": False,
    }


def run_long_continuation(
    *,
    checkpoint,
    rows: list[dict],
    train_view: Path,
    data_root: Path,
    output: Path,
    protocol,
    seed: int,
    gpu: int,
    horizon: int,
    arms: tuple[str, ...],
    eval_interval: int,
) -> dict:
    if "P_common_plain" not in arms or "A_hard_disable" not in arms:
        raise ValueError("long arms must contain P_common_plain and A_hard_disable")
    evaluation_horizons = tuple(
        range(int(eval_interval), int(horizon) + 1, int(eval_interval))
    )
    if not evaluation_horizons or evaluation_horizons[-1] != int(horizon):
        raise ValueError("long horizon must be divisible by eval interval")
    destination = Path(output) / "long_continuation" / checkpoint.checkpoint_id
    destination.mkdir(parents=True, exist_ok=True)
    engine = HandoffEngine(
        checkpoint=checkpoint,
        rows=rows,
        train_view=train_view,
        work_dir=Path(output) / "long_work",
        seed=seed,
        gpu=gpu,
        max_horizon=horizon,
    )
    results = {}
    try:
        for arm in arms:
            path = destination / f"{arm}_h{horizon}.json"
            if path.is_file():
                import json

                results[arm] = json.loads(path.read_text(encoding="utf-8"))
                continue
            print(
                f"SEARCH004 long checkpoint={checkpoint.checkpoint_id} "
                f"arm={arm} horizon={horizon}",
                flush=True,
            )
            result = engine.run_arm(
                arm=arm,
                horizon=horizon,
                protocol=protocol,
                data_root=data_root,
                eval_count=70,
                eval_start=10,
                include_lpips=True,
                save_state=destination / f"{arm}_h{horizon}.pt",
                evaluation_horizons=evaluation_horizons,
            )
            atomic_json(path, result)
            results[arm] = result
    finally:
        engine.close()
    analysis = analyze_long(
        checkpoint=checkpoint,
        results=results,
        evaluation_horizons=evaluation_horizons,
    )
    atomic_json(destination / "LONG_ANALYSIS.json", analysis)
    return analysis
