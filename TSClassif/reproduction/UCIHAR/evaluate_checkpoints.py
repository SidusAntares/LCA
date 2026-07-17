#!/usr/bin/env python3
"""Stateless best/last checkpoint evaluation with protocol provenance."""

import argparse
import csv
import hashlib
import inspect
import platform
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

TSCLASSIF_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TSCLASSIF_ROOT))

from algorithms.algorithms import LCA
from compat import load_torch_file
from configs.LCA_config import get_model_a_parser
from configs.data_model_configs import HAR as HARConfig
from configs.hparams import HAR as HARHParams
from dataloader.dataloader import data_generator
from reproduction.UCIHAR.protocol_audit import audit_corrected_public_implementation
from reproduction.UCIHAR.result_protocol import reported_metric_fields
from run_config import parse_run_ids


FIELDS = [
    "scenario", "source", "target", "run_id", "seed",
    "best_clean_macro_f1", "last_clean_macro_f1",
    "official_reported_f1", "current_reported_f1", "accuracy", "status",
    "runtime_seconds", "checkpoint_best", "checkpoint_last", "git_commit",
    "python_version", "torch_version", "cuda_version", "gpu_name",
    "get_features_returns_z", "metric_reset_fixed", "training_protocol",
    "metric_protocol", "clean_metric_protocol", "best_epoch", "last_epoch",
    "best_state_sha256", "last_state_sha256",
    "protocol_fingerprint_sha256", "is_legacy",
]


def clean_evaluate(model, state_dict, target_loader, device):
    """Evaluate one state with fresh arrays and exactly one target-test pass."""

    model.load_state_dict(state_dict)
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for data, batch_labels in target_loader:
            data = data.float().to(device)
            scores = model.inference(data)
            predictions.extend(scores.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.view(-1).long().cpu().tolist())
    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        labels=list(range(6)),
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)
    return float(accuracy), float(macro_f1)


def audit_clean_metric_runtime():
    source = inspect.getsource(clean_evaluate)
    return all(
        marker in source
        for marker in (
            "labels = []",
            "predictions = []",
            "labels=list(range(6))",
            "zero_division=0",
        )
    ) and "torchmetrics" not in source


def state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_training_row(experiment_dir, scenario, run_id):
    with (experiment_dir / "results.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("scenario") == scenario and row.get("run") == str(run_id):
                return row
    raise RuntimeError(f"missing training result for {scenario} run {run_id}")


def build_model_and_loader(data_path, target, device):
    dataset_config = HARConfig()
    hparams = HARHParams().train_params
    model_args = get_model_a_parser().parse_args([])
    model = LCA(model_args, dataset_config, hparams, device).to(device)
    loader = data_generator(
        str(Path(data_path) / "HAR"),
        target,
        dataset_config,
        hparams,
        "test",
    )
    return model, loader


def validate_checkpoint_metadata(checkpoint, audit, legacy):
    metadata = checkpoint.get("metadata")
    if legacy:
        return metadata or {}
    if not isinstance(metadata, dict):
        raise RuntimeError("formal reproduction checkpoint has no run metadata")
    if metadata.get("get_features_returns_z") is not True:
        raise RuntimeError("checkpoint was not produced by corrected z implementation")
    if metadata.get("protocol_fingerprint_sha256") != audit["fingerprint_sha256"]:
        raise RuntimeError("checkpoint protocol fingerprint differs from current frozen code")
    return metadata


def evaluate_run(
    experiment_dir, data_path, source, target, run_id, device, legacy=False
):
    audit = audit_corrected_public_implementation()
    if not audit["passed"]:
        raise RuntimeError(
            f"corrected public implementation audit failed: {audit['checks']}"
        )
    metric_reset_fixed = audit_clean_metric_runtime()
    if not metric_reset_fixed:
        raise RuntimeError("clean checkpoint metric audit failed")

    scenario = f"{source}_to_{target}"
    checkpoint_path = experiment_dir / f"{scenario}_run_{run_id}" / "checkpoint.pt"
    checkpoint = load_torch_file(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"invalid checkpoint object: {checkpoint_path}")
    metadata = validate_checkpoint_metadata(checkpoint, audit, legacy)
    training_row = read_training_row(experiment_dir, scenario, run_id)
    if not legacy and training_row.get("metric_protocol") != metadata.get(
        "metric_protocol"
    ):
        raise RuntimeError("results.csv metric protocol disagrees with checkpoint")

    model, target_loader = build_model_and_loader(data_path, target, device)
    evaluated = {}
    hashes = {}
    for checkpoint_name in ("best", "last"):
        state = checkpoint.get(checkpoint_name)
        if state is None:
            raise RuntimeError(
                f"checkpoint has no {checkpoint_name} state: {checkpoint_path}"
            )
        hashes[checkpoint_name] = state_dict_sha256(state)
        evaluated[checkpoint_name] = clean_evaluate(
            model, state, target_loader, device
        )

    reported = reported_metric_fields(training_row)
    gpu_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    return {
        "scenario": scenario,
        "source": source,
        "target": target,
        "run_id": run_id,
        "seed": run_id,
        "best_clean_macro_f1": evaluated["best"][1],
        "last_clean_macro_f1": evaluated["last"][1],
        **reported,
        "accuracy": evaluated["best"][0],
        "status": training_row.get("status", "success"),
        "runtime_seconds": training_row.get("runtime_seconds", ""),
        "checkpoint_best": f"{checkpoint_path}::best",
        "checkpoint_last": f"{checkpoint_path}::last",
        "git_commit": training_row.get("git_commit", "unknown"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "gpu_name": gpu_name,
        "get_features_returns_z": audit["checks"]["get_features_returns_z"],
        "metric_reset_fixed": metric_reset_fixed,
        "training_protocol": metadata.get("training_protocol", "legacy_unknown"),
        "metric_protocol": training_row.get("metric_protocol", "legacy_unknown"),
        "clean_metric_protocol": "clean_checkpoint",
        "best_epoch": metadata.get("best_epoch", ""),
        "last_epoch": metadata.get("last_epoch", ""),
        "best_state_sha256": hashes["best"],
        "last_state_sha256": hashes["last"],
        "protocol_fingerprint_sha256": audit["fingerprint_sha256"],
        "is_legacy": legacy,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("dataset"))
    parser.add_argument("--scenario", required=True, help="SRC,TGT")
    parser.add_argument("--run-ids", default="0,1,2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    source, target = [part.strip() for part in args.scenario.split(",")]
    device = torch.device(args.device)
    rows = [
        evaluate_run(
            args.experiment_dir,
            args.data_path,
            source,
            target,
            run_id,
            device,
            legacy=args.legacy,
        )
        for run_id in parse_run_ids(args.run_ids)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
