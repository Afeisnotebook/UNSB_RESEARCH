#!/usr/bin/env python3
"""Real parameter-gradient conflict on the shared AIO SBModel.

The original SBModel training graph is reconstructed read-only from saved
``netG/netD/netE/netF`` checkpoints.  For each selected epoch and domain, one
unpaired trainA/trainB mini-batch is passed through the generator objective,
and netG weight gradients are collected per layer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(
    os.environ.get("UNSB_MOTIVATION_ROOT", Path(__file__).resolve().parent.parent)
).expanduser().resolve()
REPO_ROOT = ROOT.parents[2]
BASELINE_ROOT = Path(
    os.environ.get(
        "UNSB_BASELINE_ROOT", REPO_ROOT / "foundation" / "canonical" / "src"
    )
).expanduser().resolve()
DOMAINS = [
    "FoggyCityscapes",
    "LowLightTrafficData",
    "RainCityscapes",
    "RSCityscapes",
    "SnowTrafficData",
]
EPOCHS = [1, 4, 5, 6, 20]
GRAD_LAYERS = [
    "model_res.2.conv_fin.2",
    "model_res.5.conv_fin.2",
    "model_res.8.conv_fin.2",
    "model.4",
    "model_upsample.5",
]


def _train_image_paths(domain: str, side: str, count: int = 3) -> list[Path]:
    d = ROOT / "datasets" / "aio" / side
    files = sorted(p for p in d.glob(f"{domain}__*.png"))
    return files[:count]


def _to_input(path: Path, device: str) -> dict:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((128, 128), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)
    return tensor


def _flatten_grad(module: torch.nn.Module) -> np.ndarray | None:
    if not hasattr(module, "weight") or module.weight is None or module.weight.grad is None:
        return None
    g = module.weight.grad.detach().reshape(-1)
    if g.numel() == 0 or not torch.isfinite(g).all():
        return None
    return g.cpu().numpy().astype(np.float32)


def _pairwise(vectors: dict[str, np.ndarray]) -> dict:
    domains = sorted(vectors)
    vals = []
    pairs = []
    for i, a in enumerate(domains):
        for b in domains[i + 1 :]:
            va = vectors[a]
            vb = vectors[b]
            denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
            c = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0
            vals.append(c)
            pairs.append((a, b, c))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean_cosine": float(np.mean(arr)) if arr.size else float("nan"),
        "conflict_fraction": float(np.mean(arr < 0.0)) if arr.size else float("nan"),
        "pairs": pairs,
    }


def _make_opt(epoch: int) -> object:
    sys.path.insert(0, str(BASELINE_ROOT))
    from options.train_options import TrainOptions

    cmd = (
        f"--dataroot {ROOT / 'datasets' / 'aio'} "
        f"--checkpoints_dir {ROOT / 'checkpoints'} "
        "--name aio_plain_s2026 --model sb --mode sb --dataset_mode unaligned --direction AtoB "
        "--lambda_SB 1.0 --lambda_NCE 1.0 --tau 0.01 --batch_size 1 "
        "--load_size 128 --crop_size 128 --preprocess resize_and_crop --num_threads 0 --gpu_ids 0 "
        "--n_epochs_decay 0 --lr 0.0001 --save_latest_freq 1000000 --print_freq 100 "
        "--display_freq 1000000 --display_id -1 --no_html --netG resnet_9blocks_cond "
        "--netD basic_cond --netE basic_cond --normG instance --normD instance --pool_size 0 "
        "--num_timesteps 5 --no_flip --continue_train --epoch latest"
    )
    opt = TrainOptions(cmd_line=cmd).parse()
    opt.epoch = epoch
    return opt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--epochs", default=",".join(str(e) for e in EPOCHS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    epochs = [int(x) for x in args.epochs.split(",") if x.strip()]
    sys.path.insert(0, str(BASELINE_ROOT))
    from models.sb_model import SBModel

    out = {"seed": 2026, "method": "aio_plain_s2026", "layers": GRAD_LAYERS, "epochs": {}}
    by_name_cache = {}

    for epoch in epochs:
        opt = _make_opt(epoch)
        model = SBModel(opt)
        # Initialize netF once with a small deterministic batch.
        a_paths = _train_image_paths(DOMAINS[0], "trainA")
        b_paths = _train_image_paths(DOMAINS[0], "trainB")
        data1 = {
            "A": _to_input(a_paths[0], args.device),
            "B": _to_input(b_paths[0], args.device),
            "A_paths": str(a_paths[0]),
            "B_paths": str(b_paths[0]),
        }
        data2 = {
            "A": _to_input(a_paths[1], args.device),
            "B": _to_input(b_paths[1], args.device),
            "A_paths": str(a_paths[1]),
            "B_paths": str(b_paths[1]),
        }
        model.data_dependent_initialize(data1, data2)
        model.setup(opt)

        epoch_grads = {layer: {} for layer in GRAD_LAYERS}
        modules = dict(model.netG.named_modules())
        for domain in DOMAINS:
            a_paths = _train_image_paths(domain, "trainA")
            b_paths = _train_image_paths(domain, "trainB")
            data1 = {
                "A": _to_input(a_paths[0], args.device),
                "B": _to_input(b_paths[0], args.device),
                "A_paths": str(a_paths[0]),
                "B_paths": str(b_paths[0]),
            }
            data2 = {
                "A": _to_input(a_paths[1], args.device),
                "B": _to_input(b_paths[1], args.device),
                "A_paths": str(a_paths[1]),
                "B_paths": str(b_paths[1]),
            }
            model.set_input(data1, data2)
            model.forward()
            loss_G = model.compute_G_loss()
            model.netG.zero_grad()
            if model.opt.netF == "mlp_sample":
                model.netF.zero_grad()
            loss_G.backward()
            for layer in GRAD_LAYERS:
                g = _flatten_grad(modules[layer])
                if g is not None:
                    epoch_grads[layer][domain] = g
            print(f"GRAD_REAL aio e{epoch} {domain}")

        layer_results = {}
        for layer in GRAD_LAYERS:
            layer_results[layer] = _pairwise(epoch_grads[layer])
            norms = {
                d: float(np.linalg.norm(v))
                for d, v in epoch_grads[layer].items()
            }
            layer_results[layer]["norm_by_domain"] = norms
        out["epochs"][str(epoch)] = layer_results
        del model
        torch.cuda.empty_cache()

    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out_json}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
