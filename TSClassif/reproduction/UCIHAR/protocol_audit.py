"""Runtime source audit for the corrected public LCA implementation."""

import ast
import hashlib
from pathlib import Path


TSCLASSIF_ROOT = Path(__file__).resolve().parents[2]
AUDITED_FILES = [
    TSCLASSIF_ROOT / "algorithms" / "algorithms.py",
    TSCLASSIF_ROOT / "configs" / "LCA_config.py",
    TSCLASSIF_ROOT / "models" / "models.py",
    TSCLASSIF_ROOT / "dataloader" / "dataloader.py",
    TSCLASSIF_ROOT / "protocol_policy.py",
    TSCLASSIF_ROOT / "metric_protocol.py",
    TSCLASSIF_ROOT / "checkpoint_metadata.py",
]


def _get_features_returns_sampled_z(source):
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_features"
    )
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    return bool(returns) and "(z_mean, z_std, z)" in ast.unparse(returns[-1].value)


def audit_corrected_public_implementation():
    sources = {
        path.relative_to(TSCLASSIF_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in AUDITED_FILES
    }
    algorithm = sources["algorithms/algorithms.py"]
    config = sources["configs/LCA_config.py"]
    models = sources["models/models.py"]
    dataloader = sources["dataloader/dataloader.py"]
    policy = sources["protocol_policy.py"]
    checks = {
        "get_features_returns_z": _get_features_returns_sampled_z(algorithm),
        "base_dist_mean_registered": "register_buffer('base_dist_mean'" in algorithm,
        "base_dist_var_registered": "register_buffer('base_dist_var'" in algorithm,
        "public_checkpoint_selector": (
            "self.policy.checkpoint_candidate(epoch, avg_meter)" in algorithm
            and "upstream_cumulative_src_cls_every_10" in policy
            and 'meters["Src_cls_loss"].avg' in policy
        ),
        "pseudo_epoch_30": "default=30" in config,
        "pseudo_threshold_099": "default=0.99" in config,
        "softmax_classifier": (
            'output_type == "probabilities"' in models
            and "nn.Softmax(dim=-1)" in models
        ),
        "public_normalization": "transforms.Normalize" in dataloader,
    }
    digest = hashlib.sha256()
    for name in sorted(sources):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sources[name].encode("utf-8"))
        digest.update(b"\0")
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fingerprint_sha256": digest.hexdigest(),
    }
