"""Build and strictly validate protocol-qualified checkpoint metadata."""

REQUIRED_METADATA_FIELDS = {
    "protocol", "upstream_commit", "current_commit", "scenario", "run_id",
    "seed", "epoch", "model_training_mode", "classifier_output_type",
    "cross_entropy_input_type", "target_test_reads_during_training",
    "checkpoint_selection_rule", "get_features_returns_z",
    "base_dist_mean_registered", "prior_enabled", "loss_weights",
    "pseudo_threshold", "pseudo_start_epoch", "python_version",
    "torch_version", "cuda_version", "gpu_name",
    "cudnn_deterministic", "cudnn_benchmark",
    "deterministic_algorithms_enabled", "metric_backend",
    "torchmetrics_version", "protocol_fingerprint_sha256",
    "primary_checkpoint", "epoch_mode_trace", "best_source_epoch",
    "last_epoch",
}


def validate_checkpoint_metadata(metadata, policy, expected_fingerprint):
    if metadata.get("protocol") != policy.name:
        raise RuntimeError(
            f"protocol mismatch: expected {policy.name}, got {metadata.get('protocol')}"
        )
    missing = REQUIRED_METADATA_FIELDS - set(metadata)
    if missing:
        raise RuntimeError(f"checkpoint metadata missing fields: {sorted(missing)}")
    if metadata["classifier_output_type"] != policy.classifier_output_type:
        raise RuntimeError("classifier output metadata mismatch")
    if metadata["cross_entropy_input_type"] != policy.cross_entropy_input_type:
        raise RuntimeError("cross entropy input metadata mismatch")
    if metadata["checkpoint_selection_rule"] != policy.checkpoint_selection_rule:
        raise RuntimeError("checkpoint selection metadata mismatch")
    if metadata["primary_checkpoint"] != policy.primary_checkpoint:
        raise RuntimeError("primary checkpoint metadata mismatch")
    if not metadata["get_features_returns_z"]:
        raise RuntimeError("checkpoint does not use sampled z features")
    if not metadata["base_dist_mean_registered"]:
        raise RuntimeError("checkpoint does not register the prior base buffer")
    if not metadata["prior_enabled"]:
        raise RuntimeError("formal baseline checkpoint disabled the prior")
    required_reads = policy.required_target_test_reads_during_training
    if (
        required_reads is not None
        and metadata["target_test_reads_during_training"] != required_reads
    ):
        raise RuntimeError(
            f"{policy.name} read target-test labels during training"
        )
    if (
        expected_fingerprint is not None
        and metadata["protocol_fingerprint_sha256"] != expected_fingerprint
    ):
        raise RuntimeError("checkpoint protocol fingerprint mismatch")
    return metadata
