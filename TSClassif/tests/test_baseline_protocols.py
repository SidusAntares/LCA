import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import einops  # noqa: F401
except ImportError:
    einops_stub = types.ModuleType("einops")
    einops_stub.rearrange = lambda value, *args, **kwargs: value
    sys.modules["einops"] = einops_stub


class ProtocolPolicyTests(unittest.TestCase):
    def test_protocols_are_centralized_and_isolated(self):
        from protocol_policy import get_protocol_policy

        cp = get_protocol_policy("corrected_public")
        clean = get_protocol_policy("clean_baseline")
        self.assertEqual(cp.classifier_output_type, "probabilities")
        self.assertEqual(clean.classifier_output_type, "logits")
        self.assertTrue(cp.evaluate_target_during_training)
        self.assertFalse(clean.evaluate_target_during_training)
        self.assertTrue(cp.cumulative_epoch_meters)
        self.assertFalse(clean.cumulative_epoch_meters)
        self.assertEqual(cp.checkpoint_selection_rule, "upstream_cumulative_src_cls_every_10")
        self.assertEqual(clean.checkpoint_selection_rule, "current_epoch_source_supervised")
        self.assertEqual(clean.primary_checkpoint, "last")
        self.assertNotEqual(cp.output_namespace, clean.output_namespace)

    def test_unknown_protocol_fails(self):
        from protocol_policy import get_protocol_policy

        with self.assertRaisesRegex(ValueError, "unknown protocol"):
            get_protocol_policy("paper-ish")

    def test_clean_pseudo_labels_use_softmax_probabilities(self):
        from protocol_policy import get_protocol_policy

        logits = torch.tensor([[2.0, 0.0, -1.0]])
        probs = get_protocol_policy("clean_baseline").probabilities(logits)
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(1)))
        self.assertTrue(torch.allclose(probs, torch.softmax(logits, dim=1)))

    def test_epoch_mode_and_meter_policy_are_centralized(self):
        from protocol_policy import get_protocol_policy

        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        meters = {"old": object()}
        model.eval()
        get_protocol_policy("corrected_public").begin_epoch(model, meters)
        self.assertFalse(model.training)
        self.assertIn("old", meters)
        get_protocol_policy("clean_baseline").begin_epoch(model, meters)
        self.assertTrue(model.training)
        self.assertEqual(meters, {})

    def test_checkpoint_rules_use_only_protocol_approved_source_loss(self):
        from protocol_policy import get_protocol_policy

        class Meter:
            def __init__(self, value):
                self.avg = value

        meters = {"Src_cls_loss": Meter(3.0), "source_supervised_loss": Meter(2.0)}
        cp = get_protocol_policy("corrected_public")
        clean = get_protocol_policy("clean_baseline")
        self.assertEqual(cp.checkpoint_candidate(9, meters), (True, 3.0))
        self.assertEqual(cp.checkpoint_candidate(8, meters), (False, 3.0))
        self.assertEqual(clean.checkpoint_candidate(1, meters), (True, 2.0))


class MetricSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.first_preds = torch.tensor([0, 0, 1, 2, 2, 2])
        self.first_labels = torch.tensor([0, 1, 1, 2, 0, 2])
        self.second_preds = torch.tensor([0, 1, 1, 1, 2, 0])
        self.second_labels = torch.tensor([0, 0, 1, 2, 2, 2])
        self.a = f1_score(self.first_labels, self.first_preds, average="macro", labels=[0, 1, 2], zero_division=0)
        self.b = f1_score(self.second_labels, self.second_preds, average="macro", labels=[0, 1, 2], zero_division=0)
        self.c = f1_score(
            torch.cat([self.first_labels, self.second_labels]),
            torch.cat([self.first_preds, self.second_preds]),
            average="macro", labels=[0, 1, 2], zero_division=0,
        )
        self.assertEqual(len({round(self.a, 8), round(self.b, 8), round(self.c, 8)}), 3)

    def test_torchmetrics_forward_returns_current_call_value(self):
        from metric_protocol import TorchMetrics132ForwardCompat

        metric = TorchMetrics132ForwardCompat(num_classes=3)
        first = metric(self.first_preds, self.first_labels)
        second = metric(self.second_preds, self.second_labels)
        self.assertAlmostEqual(first, self.a)
        self.assertAlmostEqual(second, self.b)

    def test_torchmetrics_compute_returns_accumulated_value(self):
        from metric_protocol import TorchMetrics132ForwardCompat

        metric = TorchMetrics132ForwardCompat(num_classes=3)
        metric(self.first_preds, self.first_labels)
        metric(self.second_preds, self.second_labels)
        self.assertAlmostEqual(metric.compute(), self.c)

    def test_official_forward_f1_matches_upstream_call_semantics(self):
        from metric_protocol import TorchMetrics132ForwardCompat

        metric = TorchMetrics132ForwardCompat(num_classes=3)
        metric(self.first_preds, self.first_labels)
        self.assertAlmostEqual(metric(self.second_preds, self.second_labels), self.b)
        self.assertNotAlmostEqual(metric.compute(), self.b)

    def test_clean_f1_is_stateless(self):
        value = f1_score(
            self.second_labels, self.second_preds, average="macro",
            labels=[0, 1, 2], zero_division=0,
        )
        self.assertAlmostEqual(value, self.b)


class ClassifierContractTests(unittest.TestCase):
    class Config:
        features_len = 1
        final_out_channels = 4
        num_classes = 3

    def test_clean_classifier_returns_logits(self):
        from models.models import classifier

        model = classifier(self.Config(), output_type="logits")
        output = model(torch.randn(5, 4))
        self.assertFalse(torch.allclose(output.sum(dim=1), torch.ones(5), atol=1e-5))

    def test_corrected_public_preserves_original_classifier_behavior(self):
        from models.models import classifier

        model = classifier(self.Config(), output_type="probabilities")
        output = model(torch.randn(5, 4))
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.ones(5), atol=1e-5))

    def test_clean_cross_entropy_uses_logits(self):
        from protocol_policy import get_protocol_policy

        policy = get_protocol_policy("clean_baseline")
        logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
        labels = torch.tensor([0])
        self.assertTrue(torch.allclose(policy.cross_entropy_input(logits), logits))
        torch.nn.functional.cross_entropy(policy.cross_entropy_input(logits), labels).backward()
        self.assertIsNotNone(logits.grad)


class MetadataTests(unittest.TestCase):
    @staticmethod
    def valid_metadata(policy):
        return {
            "protocol": policy.name, "upstream_commit": "upstream",
            "current_commit": "current", "scenario": "18_to_14",
            "run_id": 0, "seed": 0, "epoch": 40,
            "model_training_mode": True, "epoch_mode_trace": [],
            "classifier_output_type": policy.classifier_output_type,
            "cross_entropy_input_type": policy.cross_entropy_input_type,
            "target_test_reads_during_training": (
                40 if policy.name == "corrected_public" else 0
            ),
            "checkpoint_selection_rule": policy.checkpoint_selection_rule,
            "primary_checkpoint": policy.primary_checkpoint,
            "get_features_returns_z": True,
            "base_dist_mean_registered": True, "prior_enabled": True,
            "loss_weights": {}, "pseudo_threshold": 0.99,
            "pseudo_start_epoch": 30, "python_version": "3.9",
            "torch_version": "2.3.1", "cuda_version": "11.8",
            "gpu_name": "test", "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "deterministic_algorithms_enabled": False,
            "metric_backend": "test", "torchmetrics_version": "1.3.2",
            "best_source_epoch": 39, "last_epoch": 40,
            "protocol_fingerprint_sha256": "fingerprint",
        }

    def test_metadata_mismatch_fails_immediately(self):
        from checkpoint_metadata import validate_checkpoint_metadata
        from protocol_policy import get_protocol_policy

        with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
            validate_checkpoint_metadata(
                {"protocol": "corrected_public"},
                get_protocol_policy("clean_baseline"),
                expected_fingerprint=None,
            )

    def test_complete_metadata_is_accepted_and_missing_field_fails(self):
        from checkpoint_metadata import validate_checkpoint_metadata
        from protocol_policy import get_protocol_policy

        policy = get_protocol_policy("clean_baseline")
        metadata = self.valid_metadata(policy)
        self.assertIs(
            validate_checkpoint_metadata(metadata, policy, "fingerprint"), metadata
        )
        del metadata["seed"]
        with self.assertRaisesRegex(RuntimeError, "missing fields"):
            validate_checkpoint_metadata(metadata, policy, "fingerprint")

    def test_clean_metadata_rejects_target_test_reads(self):
        from checkpoint_metadata import validate_checkpoint_metadata
        from protocol_policy import get_protocol_policy

        policy = get_protocol_policy("clean_baseline")
        metadata = self.valid_metadata(policy)
        metadata["target_test_reads_during_training"] = 1
        with self.assertRaisesRegex(RuntimeError, "read target-test"):
            validate_checkpoint_metadata(metadata, policy, "fingerprint")


class SmokeContractTests(unittest.TestCase):
    def test_synthetic_smoke_outputs_two_complete_protocol_records(self):
        script = ROOT / "tools" / "smoke_baseline_protocol.py"
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "smoke.json"
            completed = subprocess.run(
                [sys.executable, str(script), "--synthetic", "--output", str(out)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual({item["protocol"] for item in records}, {"corrected_public", "clean_baseline"})
            for item in records:
                self.assertTrue(item["passed"])
                self.assertFalse(item["prior_disabled"])
                self.assertTrue(item["base_dist_accessible"])
                self.assertTrue(item["pseudo_label_path_exercised"])
                self.assertTrue(item["optimizer_step_changed_parameter"])
                self.assertTrue(all(item["finite_losses"].values()))
                self.assertTrue(all(item["gradient_groups"].values()))


if __name__ == "__main__":
    unittest.main()
