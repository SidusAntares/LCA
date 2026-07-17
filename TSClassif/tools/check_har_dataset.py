#!/usr/bin/env python3
"""Validate every domain referenced by the official HAR scenarios."""

import argparse
import collections
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compat import load_torch_file
from configs.data_model_configs import HAR


def validate_file(path, batch_size):
    try:
        dataset = load_torch_file(path)
    except Exception as exc:
        return None, [f"cannot load: {type(exc).__name__}: {exc}"]
    try:
        errors = []
        if not isinstance(dataset, dict):
            return None, [f"expected dict, got {type(dataset).__name__}"]
        if "samples" not in dataset or "labels" not in dataset:
            return None, ["missing samples and/or labels"]
        samples = torch.as_tensor(dataset["samples"])
        labels = torch.as_tensor(dataset["labels"]).reshape(-1)
        if samples.ndim != 3:
            errors.append(f"samples must be 3-D, got {tuple(samples.shape)}")
            return None, errors
        if samples.shape[1:] == (9, 128):
            interpreted = samples
        elif samples.shape[1:] == (128, 9):
            interpreted = samples.transpose(1, 2)
        else:
            errors.append(f"samples cannot be interpreted as [N, 9, 128]: {tuple(samples.shape)}")
            interpreted = samples
        if labels.numel() != samples.shape[0]:
            errors.append(f"label count {labels.numel()} != sample count {samples.shape[0]}")
        if labels.numel() and (labels.min().item() < 0 or labels.max().item() > 5):
            errors.append(f"labels outside [0, 5]: min={labels.min().item()} max={labels.max().item()}")
        if labels.numel() and not torch.equal(labels, labels.long().to(labels.dtype)):
            errors.append("labels contain non-integer class values")
        if not torch.isfinite(interpreted).all().item():
            errors.append("samples contain NaN or Inf")
        if path.name.startswith("train_") and samples.shape[0] < batch_size:
            errors.append(f"training sample count {samples.shape[0]} < batch_size {batch_size}")
        distribution = collections.Counter(int(value) for value in labels.tolist())
        info = {
            "samples": int(samples.shape[0]),
            "shape": tuple(interpreted.shape),
            "distribution": dict(sorted(distribution.items())),
        }
        return info, errors
    except Exception as exc:
        return None, [f"validation error: {type(exc).__name__}: {exc}"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "HAR")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    domains = sorted({domain for pair in HAR().scenarios for domain in pair}, key=int)
    print("Official HAR domains: " + ", ".join(domains))
    failures = []
    for domain in domains:
        for split in ("train", "test"):
            path = args.data_dir / f"{split}_{domain}.pt"
            if not path.is_file():
                failures.append(f"{path}: missing")
                print(f"{domain} {split}: MISSING")
                continue
            info, errors = validate_file(path, args.batch_size)
            if info:
                print(
                    f"{domain} {split}: N={info['samples']} shape={info['shape']} "
                    f"classes={info['distribution']}"
                )
            for error in errors:
                failures.append(f"{path}: {error}")
    if failures:
        print("Dataset failures:")
        for failure in failures:
            print(f"- {failure}")
    print("PASS" if not failures else "FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
