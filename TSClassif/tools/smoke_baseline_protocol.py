#!/usr/bin/env python3
"""Run one real LCA optimizer step under each baseline protocol."""

import argparse
import collections
import json
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_minimal_einops_for_offline_smoke():
    try:
        import einops  # noqa: F401
        return
    except ImportError:
        pass

    module = types.ModuleType("einops")

    def rearrange(value, pattern, **axes):
        normalized = " ".join(pattern.split())
        if normalized == "b l m -> b m l":
            return value.permute(0, 2, 1)
        if normalized == "b m n p -> (b m) n p":
            b, m, n, p = value.shape
            return value.reshape(b * m, n, p)
        if normalized == "b l m -> (b m) 1 l":
            b, length, m = value.shape
            return value.permute(0, 2, 1).reshape(b * m, 1, length)
        if normalized == "(b m) l -> b l m":
            b = axes["b"]
            _, length = value.shape
            m = value.shape[0] // b
            return value.reshape(b, m, length).permute(0, 2, 1)
        raise RuntimeError(f"offline smoke does not implement einops pattern: {pattern}")

    module.rearrange = rearrange
    sys.modules["einops"] = module


_install_minimal_einops_for_offline_smoke()

from algorithms.algorithms import LCA  # noqa: E402
from configs.LCA_config import get_model_a_parser  # noqa: E402
from configs.data_model_configs import HAR  # noqa: E402
from dataloader.dataloader import data_generator  # noqa: E402
from protocol_policy import PROTOCOL_NAMES, get_protocol_policy  # noqa: E402
from utils import AverageMeter, fix_randomness  # noqa: E402


def _synthetic_batches(config, batch_size):
    generator = torch.Generator().manual_seed(1729)
    source_x = torch.randn(
        batch_size, config.input_channels, config.sequence_len, generator=generator
    )
    target_x = torch.randn(
        batch_size, config.input_channels, config.sequence_len, generator=generator
    ) + 3.0
    source_y = torch.arange(batch_size) % config.num_classes
    target_y = torch.zeros(batch_size, dtype=torch.long)
    return [(source_x, source_y)], [(target_x, target_y)]


def _real_batches(data_path, source, target, config, hparams):
    source_loader = data_generator(data_path, source, config, hparams, "train")
    target_loader = data_generator(data_path, target, config, hparams, "train")
    return [next(iter(source_loader))], [next(iter(target_loader))]


def _run_one(protocol, args):
    fix_randomness(0)
    config = HAR()
    hparams = {"num_epochs": 1, "batch_size": args.batch_size}
    model_args = get_model_a_parser().parse_args([])
    model_args.type = "type1"
    model_args.No_prior = False
    policy = get_protocol_policy(protocol)
    model = LCA(model_args, config, hparams, torch.device(args.device), policy)
    model.to(args.device)
    model.capture_batch_audit = True
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    if args.synthetic:
        source, target = _synthetic_batches(config, args.batch_size)
    else:
        source, target = _real_batches(
            str(Path(args.data_path) / "HAR"), args.source, args.target,
            config, hparams,
        )
    source = [(x.to(args.device), y.to(args.device)) for x, y in source]
    target = [(x.to(args.device), y.to(args.device)) for x, y in target]
    meters = collections.defaultdict(AverageMeter)
    smoke_epoch = model.config.start_psuedo_step + 1
    model.training_epoch(source, target, meters, epoch=smoke_epoch)
    audit = model.last_batch_audit
    finite_losses = {
        name: bool(torch.isfinite(torch.tensor(value)).item())
        for name, value in audit["losses"].items()
    }
    optimizer_step_changed_parameter = any(
        not torch.equal(parameters_before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    base_dist_accessible = bool(torch.isfinite(model.base_dist.mean).all().item())
    passed = (
        all(finite_losses.values())
        and all(audit["gradient_groups"].values())
        and optimizer_step_changed_parameter
        and base_dist_accessible
    )
    return {
        "protocol": protocol,
        "classifier_output_type": policy.classifier_output_type,
        "cross_entropy_input_type": policy.cross_entropy_input_type,
        "prior_disabled": bool(model.config.No_prior),
        "base_dist_accessible": base_dist_accessible,
        "pseudo_label_path_exercised": smoke_epoch > model.config.start_psuedo_step,
        "optimizer_step_changed_parameter": optimizer_step_changed_parameter,
        "finite_losses": finite_losses,
        "losses": audit["losses"],
        "gradient_groups": audit["gradient_groups"],
        "threshold_gradient_nonzero_audit": audit["threshold_gradient_nonzero"],
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--real-har", action="store_true")
    parser.add_argument("--data-path", default="dataset")
    parser.add_argument("--source", default="18")
    parser.add_argument("--target", default="14")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [_run_one(name, args) for name in PROTOCOL_NAMES]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    for record in records:
        protocol_output = output.parent / f"smoke_{record['protocol']}.json"
        protocol_output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps(records, indent=2))
    return 0 if all(record["passed"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
