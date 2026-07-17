#!/usr/bin/env python3
"""One full-prior LCA forward/backward/optimizer smoke test."""

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.algorithms import LCA
from configs.LCA_config import get_model_a_parser
from configs.data_model_configs import HAR
from configs.hparams import HAR as HARHParams
from dataloader.dataloader import data_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="dataset")
    parser.add_argument("--source", default="18")
    parser.add_argument("--target", default="14")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    dataset_config = HAR()
    hparams = {**HARHParams().train_params, **HARHParams().alg_hparams["LCA"]}
    data_dir = str(Path(args.data_path) / "HAR")
    src_loader = data_generator(data_dir, args.source, dataset_config, hparams, "train")
    tgt_loader = data_generator(data_dir, args.target, dataset_config, hparams, "train")
    src_x, src_y = next(iter(src_loader))
    tgt_x, _ = next(iter(tgt_loader))
    src_x, src_y = src_x.to(device), src_y.to(device)
    tgt_x = tgt_x.to(device)

    model_config = get_model_a_parser().parse_args(["--type", "type1", "--lr", "0.001"])
    if model_config.No_prior:
        raise AssertionError("smoke test must exercise the complete prior")
    model = LCA(model_config, dataset_config, hparams, device).to(device)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    (sm, ss, sz), src_rec, src_pred = model.get_features(src_x)
    (tm, ts, tz), tgt_rec, tgt_pred = model.get_features(tgt_x)
    losses = model._LCA__loss_function(
        sm, ss, sz, src_x, src_rec, src_pred,
        tm, ts, tz, tgt_x, tgt_rec, tgt_pred,
        src_y, 1, no_kl=False,
    )
    names = ("class_loss", "rec_loss", "sparsity_loss", "kld_loss", "structure_loss")
    for name, value in zip(names, losses):
        scalar = float(value.detach().cpu())
        print(f"{name}: {scalar:.8f}")
        if not math.isfinite(scalar):
            raise AssertionError(f"{name} is not finite")
    total = (
        losses[0] * model.config.class_weight
        + losses[1] * model.config.rec_weight
        + losses[2] * model.config.sparsity_weight
        + losses[3] * model.config.z_kl_weight
        + losses[4] * model.config.structure_weight
    )
    model.optimizer.zero_grad()
    total.backward()
    critical = {
        "z_net": next(model.z_net.parameters()).grad,
        "transition_prior": next(model.transition_prior_fix.parameters()).grad,
        "threshold": model.threa.grad,
    }
    missing = [name for name, grad in critical.items() if grad is None]
    for name, grad in critical.items():
        print(f"gradient {name}: {'OK' if grad is not None else 'NONE'}")
    if missing:
        raise AssertionError("missing gradients: " + ", ".join(missing))
    model.optimizer.step()
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    print(f"peak_gpu_memory_mb: {peak_mb:.2f}")
    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

