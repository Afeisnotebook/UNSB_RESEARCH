"""Target-aware discovery evaluator. Confirmation identities are hard-blocked."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def stable_seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0x7FFF_FFFF


def bridge_times(count: int, device) -> torch.Tensor:
    increments = np.asarray([0.0] + [1.0 / (index + 1) for index in range(count - 1)])
    times = np.cumsum(increments)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    return torch.tensor(np.concatenate([np.zeros(1), times]), dtype=torch.float32, device=device)


def read_image(path: Path, size: int = 128) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        value = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(value).permute(2, 0, 1).contiguous().unsqueeze(0)


def to_unit_image(value: torch.Tensor) -> np.ndarray:
    return (
        value.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5
    ).clip(0.0, 1.0)


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else -10.0 * math.log10(mse)


def select_discovery_rows(
    rows: list[dict], *, start_per_domain: int, count_per_domain: int
) -> list[dict]:
    if start_per_domain < 0 or count_per_domain <= 0:
        raise ValueError("invalid discovery slice")
    selected = []
    for domain in sorted({row["domain"] for row in rows}):
        domain_rows = sorted(
            (row for row in rows if row["domain"] == domain and row["split"] == "discovery"),
            key=lambda row: int(row["order"]),
        )
        take = domain_rows[start_per_domain : start_per_domain + count_per_domain]
        if len(take) != count_per_domain:
            raise RuntimeError(f"{domain}: discovery slice is incomplete")
        selected.extend(take)
    if any(row["split"] != "discovery" for row in selected):
        raise RuntimeError("confirmation20 access blocked")
    return selected


def rollout(model, source: torch.Tensor, *, seed: int) -> torch.Tensor:
    source = source.to(model.device)
    times = bridge_times(int(model.opt.num_timesteps), model.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    current = None
    endpoint = None
    with torch.no_grad():
        for index in range(int(model.opt.num_timesteps)):
            if index == 0:
                current = source
            else:
                delta = times[index] - times[index - 1]
                denominator = times[-1] - times[index - 1]
                interpolation = delta / denominator
                scale = delta * (1.0 - interpolation)
                noise = torch.randn(source.shape, generator=generator).to(model.device)
                current = (
                    (1.0 - interpolation) * current
                    + interpolation * endpoint.detach()
                    + (scale * float(model.opt.tau)).sqrt() * noise
                )
            time_index = torch.full((source.shape[0],), index, device=model.device, dtype=torch.long)
            latent = torch.randn(
                (source.shape[0], 4 * int(model.opt.ngf)), generator=generator
            ).to(model.device)
            endpoint = model._predict_endpoint(model.netG, current, time_index, latent)
    return endpoint


def _lpips_model(device):
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception:
        return None


def evaluate(
    model,
    *,
    rows: list[dict],
    data_root: Path,
    start_per_domain: int,
    count_per_domain: int,
    eval_seed: int,
    include_lpips: bool = False,
) -> dict:
    selected = select_discovery_rows(
        rows, start_per_domain=start_per_domain, count_per_domain=count_per_domain
    )
    training_modes = {
        name: getattr(model, "net" + name).training for name in model.model_names
    }
    model.eval()
    perceptual = _lpips_model(model.device) if include_lpips else None
    image_rows = []
    for row in selected:
        input_path = Path(data_root) / row["input_relpath"]
        target_path = Path(data_root) / row["target_relpath"]
        source = read_image(input_path)
        target_tensor = read_image(target_path)
        output = rollout(
            model,
            source,
            seed=stable_seed(eval_seed, row["domain"], row["stem"], "bridge"),
        )
        output_image = to_unit_image(output)
        target_image = to_unit_image(target_tensor)
        value = {
            "domain": row["domain"],
            "stem": row["stem"],
            "order": int(row["order"]),
            "psnr": psnr(output_image, target_image),
            "ssim": float(
                structural_similarity(target_image, output_image, data_range=1.0, channel_axis=2)
            ),
            "lpips": None,
        }
        if perceptual is not None:
            with torch.no_grad():
                value["lpips"] = float(
                    perceptual(output.to(model.device).clamp(-1, 1), target_tensor.to(model.device)).item()
                )
        image_rows.append(value)
    for name, was_training in training_modes.items():
        getattr(model, "net" + name).train(was_training)

    by_domain = defaultdict(list)
    for row in image_rows:
        by_domain[row["domain"]].append(row)
    domains = {}
    for domain, values in sorted(by_domain.items()):
        domains[domain] = {
            "n": len(values),
            "psnr": float(np.mean([value["psnr"] for value in values])),
            "ssim": float(np.mean([value["ssim"] for value in values])),
            "lpips": (
                None
                if any(value["lpips"] is None for value in values)
                else float(np.mean([value["lpips"] for value in values]))
            ),
        }
    return {
        "split": "discovery",
        "start_per_domain": start_per_domain,
        "count_per_domain": count_per_domain,
        "eval_seed": int(eval_seed),
        "macro_psnr": float(np.mean([value["psnr"] for value in domains.values()])),
        "macro_ssim": float(np.mean([value["ssim"] for value in domains.values()])),
        "macro_lpips": (
            None
            if any(value["lpips"] is None for value in domains.values())
            else float(np.mean([value["lpips"] for value in domains.values()]))
        ),
        "domains": domains,
        "images": image_rows,
        "confirmation20_opened": False,
    }


def compare(method: dict, plain: dict, *, step: int) -> dict:
    if (
        method["start_per_domain"] != plain["start_per_domain"]
        or method["count_per_domain"] != plain["count_per_domain"]
    ):
        raise RuntimeError("matched evaluation slice mismatch")
    domain_delta = {
        domain: {
            "psnr": method["domains"][domain]["psnr"] - plain["domains"][domain]["psnr"],
            "ssim": method["domains"][domain]["ssim"] - plain["domains"][domain]["ssim"],
            "lpips": (
                None
                if method["domains"][domain]["lpips"] is None
                or plain["domains"][domain]["lpips"] is None
                else method["domains"][domain]["lpips"] - plain["domains"][domain]["lpips"]
            ),
        }
        for domain in method["domains"]
    }
    psnr_values = [value["psnr"] for value in domain_delta.values()]
    macro_ssim_delta = method["macro_ssim"] - plain["macro_ssim"]
    macro_lpips_delta = (
        None
        if method["macro_lpips"] is None or plain["macro_lpips"] is None
        else method["macro_lpips"] - plain["macro_lpips"]
    )
    guardrails = (
        min(psnr_values) >= -1.0
        and macro_ssim_delta >= -0.01
        and (macro_lpips_delta is None or macro_lpips_delta <= 0.02)
    )
    return {
        "step": int(step),
        "macro_psnr": method["macro_psnr"],
        "plain_macro_psnr": plain["macro_psnr"],
        "macro_psnr_delta": method["macro_psnr"] - plain["macro_psnr"],
        "final_psnr": method["macro_psnr"],
        "macro_ssim_delta": macro_ssim_delta,
        "macro_lpips_delta": macro_lpips_delta,
        "positive_domains": sum(value > 0 for value in psnr_values),
        "worst_domain_delta": min(psnr_values),
        "domain_delta": domain_delta,
        "guardrails_pass": guardrails,
    }
