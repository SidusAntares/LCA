#!/usr/bin/env python3
"""Strict, stateless evaluation of protocol-qualified LCA checkpoints."""

import argparse
import csv
import hashlib
import platform
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

TSCLASSIF_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TSCLASSIF_ROOT))

from algorithms.algorithms import LCA
from checkpoint_metadata import validate_checkpoint_metadata
from compat import load_torch_file
from configs.LCA_config import get_model_a_parser
from configs.data_model_configs import HAR as HARConfig
from configs.hparams import HAR as HARHParams
from dataloader.dataloader import data_generator
from protocol_policy import PROTOCOL_NAMES, get_protocol_policy
from reproduction.UCIHAR.protocol_audit import audit_corrected_public_implementation
from run_config import parse_run_ids


FIELDS = [
    "scenario", "source", "target", "run_id", "seed", "protocol",
    "last_clean_macro_f1", "best_source_clean_macro_f1",
    "last_accuracy", "best_source_accuracy", "primary_checkpoint",
    "primary_clean_macro_f1", "official_forward_f1",
    "official_accumulated_compute_f1_audit", "status", "runtime_seconds",
    "peak_gpu_memory_mb", "checkpoint_path", "last_state_sha256",
    "best_source_state_sha256", "upstream_commit", "current_commit",
    "python_version", "torch_version", "cuda_version", "gpu_name",
    "classifier_output_type", "cross_entropy_input_type",
    "target_test_reads_during_training", "checkpoint_selection_rule",
    "metric_backend", "torchmetrics_version", "best_source_epoch",
    "last_epoch", "protocol_fingerprint_sha256",
]


def clean_evaluate(model, state_dict, target_loader, device, num_classes=6):
    """Evaluate one state with fresh arrays and exactly one target-test pass."""
    model.load_state_dict(state_dict)
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for data, batch_labels in target_loader:
            scores = model.inference(data.float().to(device))
            predictions.extend(scores.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.view(-1).long().cpu().tolist())
    macro_f1 = f1_score(
        labels, predictions, average="macro", labels=list(range(num_classes)),
        zero_division=0,
    )
    return float(accuracy_score(labels, predictions)), float(macro_f1)


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
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("scenario") == scenario and row.get("run") == str(run_id)
        ]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one training result for {scenario} run {run_id}, got {len(rows)}"
        )
    return rows[0]


def build_model_and_loader(data_path, target, device, policy):
    dataset_config = HARConfig()
    hparams = HARHParams().train_params
    model_args = get_model_a_parser().parse_args([])
    model = LCA(model_args, dataset_config, hparams, device, policy).to(device)
    loader = data_generator(
        str(Path(data_path) / "HAR"), target, dataset_config, hparams, "test"
    )
    return model, loader


def evaluate_run(experiment_dir, data_path, source, target, run_id, device, protocol):
    policy = get_protocol_policy(protocol)
    audit = audit_corrected_public_implementation()
    if not audit["passed"]:
        raise RuntimeError(f"implementation audit failed: {audit['checks']}")
    scenario = f"{source}_to_{target}"
    checkpoint_path = experiment_dir / f"{scenario}_run_{run_id}" / "checkpoint.pt"
    checkpoint = load_torch_file(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"invalid checkpoint object: {checkpoint_path}")
    metadata = validate_checkpoint_metadata(
        checkpoint.get("metadata", {}), policy, audit["fingerprint_sha256"]
    )
    if metadata["scenario"] != scenario or metadata["run_id"] != run_id:
        raise RuntimeError("checkpoint scenario/run metadata mismatch")
    training_row = read_training_row(experiment_dir, scenario, run_id)
    if training_row.get("protocol") != protocol:
        raise RuntimeError("results.csv protocol disagrees with checkpoint request")

    model, target_loader = build_model_and_loader(data_path, target, device, policy)
    evaluated = {}
    hashes = {}
    for checkpoint_name in ("last", "best_source"):
        state = checkpoint.get(checkpoint_name)
        if state is None:
            raise RuntimeError(f"checkpoint has no {checkpoint_name} state")
        hashes[checkpoint_name] = state_dict_sha256(state)
        evaluated[checkpoint_name] = clean_evaluate(
            model, state, target_loader, device, model.config.num_classes
        )
    primary = policy.primary_checkpoint
    official_forward = (
        training_row.get("official_forward_f1", "")
        if policy.metric_protocol == "torchmetrics_1_3_2_forward_persistent_state"
        else ""
    )
    official_compute = (
        training_row.get("official_accumulated_compute_f1_audit", "")
        if policy.metric_protocol == "torchmetrics_1_3_2_forward_persistent_state"
        else ""
    )
    return {
        "scenario": scenario, "source": source, "target": target,
        "run_id": run_id, "seed": metadata["seed"], "protocol": protocol,
        "last_clean_macro_f1": evaluated["last"][1],
        "best_source_clean_macro_f1": evaluated["best_source"][1],
        "last_accuracy": evaluated["last"][0],
        "best_source_accuracy": evaluated["best_source"][0],
        "primary_checkpoint": primary,
        "primary_clean_macro_f1": evaluated[primary][1],
        "official_forward_f1": official_forward,
        "official_accumulated_compute_f1_audit": official_compute,
        "status": training_row.get("status", ""),
        "runtime_seconds": training_row.get("runtime_seconds", ""),
        "peak_gpu_memory_mb": training_row.get("peak_gpu_memory_mb", ""),
        "checkpoint_path": str(checkpoint_path),
        "last_state_sha256": hashes["last"],
        "best_source_state_sha256": hashes["best_source"],
        "upstream_commit": metadata["upstream_commit"],
        "current_commit": metadata["current_commit"],
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "classifier_output_type": metadata["classifier_output_type"],
        "cross_entropy_input_type": metadata["cross_entropy_input_type"],
        "target_test_reads_during_training": metadata["target_test_reads_during_training"],
        "checkpoint_selection_rule": metadata["checkpoint_selection_rule"],
        "metric_backend": metadata["metric_backend"],
        "torchmetrics_version": metadata["torchmetrics_version"],
        "best_source_epoch": metadata["best_source_epoch"],
        "last_epoch": metadata["last_epoch"],
        "protocol_fingerprint_sha256": metadata["protocol_fingerprint_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("dataset"))
    parser.add_argument("--scenario", required=True, help="SRC,TGT")
    parser.add_argument("--run-ids", default="0,1,2")
    parser.add_argument("--protocol", required=True, choices=PROTOCOL_NAMES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, target = [part.strip() for part in args.scenario.split(",")]
    rows = [
        evaluate_run(args.experiment_dir, args.data_path, source, target, run_id,
                     torch.device(args.device), args.protocol)
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
