import ast
import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent


def source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class OfflineHarContractTests(unittest.TestCase):
    def test_server_sync_uploads_code_to_exact_non_nested_destination(self):
        script_path = REPOSITORY_ROOT / "sync_to_server.sh"
        self.assertTrue(script_path.is_file(), "sync_to_server.sh")
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('REMOTE_HOST="${REMOTE_HOST:-10.150.10.38}"', script)
        self.assertIn('REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/data/user/LCA}"', script)
        self.assertIn('--exclude="${PROJECT_NAME}/TSClassif/dataset"', script)
        self.assertIn('--exclude="${PROJECT_NAME}/**/checkpoint.pt"', script)
        self.assertIn('scp "$ARCHIVE_PATH"', script)
        self.assertIn("tar -xzf", script)
        self.assertNotIn("scp -r", script)

    def test_har_script_uses_supported_explicit_arguments(self):
        script = source("scripts/HAR.sh")
        self.assertIn("--num_runs 3", script)
        self.assertNotIn("--num_run 3", script)
        self.assertIn("--phase train", script)
        self.assertIn("--data_path dataset", script)
        self.assertIn("--device cuda:0", script)

    def test_cli_exposes_reproduction_selection_controls(self):
        run_source = source("run.py")
        for flag in (
            "--scenario",
            "--num_epochs",
            "--run_ids",
            "--training_protocol",
            "--metric_protocol",
        ):
            self.assertIn(flag, run_source)

    def test_get_features_returns_sampled_latent(self):
        tree = ast.parse(source("algorithms/algorithms.py"))
        get_features = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_features"
        )
        returned = [node for node in ast.walk(get_features) if isinstance(node, ast.Return)]
        self.assertTrue(returned)
        self.assertIn("(z_mean, z_std, z)", ast.unparse(returned[-1].value))

    def test_dataloader_preserves_public_normalization_and_uses_compat_load(self):
        dataloader_source = source("dataloader/dataloader.py")
        self.assertIn("from torchvision import transforms", dataloader_source)
        self.assertIn("transforms.Normalize", dataloader_source)
        self.assertIn("load_torch_file", dataloader_source)
        compat_source = source("compat.py")
        self.assertIn("inspect.signature", compat_source)
        self.assertIn("weights_only", compat_source)

    def test_public_metric_history_is_separate_from_clean_evaluation(self):
        trainer_source = source("trainers/abstract_trainer.py")
        self.assertNotIn("from torchmetrics", trainer_source)
        for name in ("accuracy_score", "f1_score", "roc_auc_score"):
            self.assertIn(name, trainer_source)
        self.assertIn("_official_scores", trainer_source)
        self.assertIn(
            'accumulate=self.metric_protocol == "official_stateful_no_reset"',
            trainer_source,
        )
        evaluator = source("reproduction/UCIHAR/evaluate_checkpoints.py")
        self.assertNotIn("_official_scores", evaluator)

    def test_two_target_evaluation_protocols_are_explicit(self):
        algorithm_source = source("algorithms/algorithms.py")
        self.assertIn('training_protocol == "paper_code_protocol"', algorithm_source)
        self.assertIn('training_protocol == "baseline_clean_protocol"', algorithm_source)
        self.assertIn("metric = value_method(is_train=False)[1]", algorithm_source)
        self.assertIn("self.train()", algorithm_source)
        paper_block = algorithm_source.split(
            'training_protocol == "paper_code_protocol"', 1
        )[1].split("logger.debug(f'-------------------------------------')", 1)[0]
        self.assertNotIn("self.train()", paper_block)

    def test_checkpoint_selection_preserves_public_src_cls_loss(self):
        algorithm_source = source("algorithms/algorithms.py")
        self.assertIn("avg_meter['Src_cls_loss'].avg < best_src_risk", algorithm_source)
        self.assertNotIn("Src_selection_loss", algorithm_source)

    def test_target_test_is_loaded_before_training_but_not_used_for_selection(self):
        trainer_source = source("trainers/train.py")
        load_position = trainer_source.index("self.load_data(")
        initialize_position = trainer_source.index("self.initialize_algorithm()", load_position)
        update_position = trainer_source.index("self.algorithm.update", initialize_position)
        self.assertLess(load_position, initialize_position)
        self.assertLess(initialize_position, update_position)
        selection_block = source("algorithms/algorithms.py").split(
            "# saving the best model based on src risk", 1
        )[1].split("logger.debug", 1)[0]
        self.assertNotIn("value_method", selection_block)
        self.assertNotIn("trg_value", selection_block)

    def test_clean_checkpoint_evaluator_is_separate_from_training(self):
        evaluator_path = ROOT / "reproduction" / "UCIHAR" / "evaluate_checkpoints.py"
        self.assertTrue(evaluator_path.is_file())
        evaluator = evaluator_path.read_text(encoding="utf-8")
        self.assertIn("for checkpoint_name in (\"best\", \"last\")", evaluator)
        self.assertIn("labels=list(range(6))", evaluator)
        self.assertIn("zero_division=0", evaluator)
        self.assertIn("f1_score(", evaluator)
        self.assertIn("accuracy_score(", evaluator)
        self.assertNotIn("from torchmetrics", evaluator)
        self.assertIn("reported_metric_fields", evaluator)
        self.assertNotIn('"get_features_returns_z": True', evaluator)
        self.assertNotIn('"metric_reset_fixed": True', evaluator)

    def test_results_and_checkpoints_record_protocol_metadata(self):
        trainer_source = source("trainers/train.py")
        for field in (
            "training_protocol",
            "metric_protocol",
            "get_features_returns_z",
            "metric_reset_fixed",
        ):
            self.assertIn(field, trainer_source)
        abstract_source = source("trainers/abstract_trainer.py")
        self.assertIn('"metadata": metadata', abstract_source)
        self.assertIn('"best_epoch"', trainer_source)

    def test_failures_are_recorded_and_reraised_before_checkpoint(self):
        trainer_source = source("trainers/train.py")
        self.assertIn("failed_runs.jsonl", trainer_source)
        self.assertIn("traceback.format_exc", trainer_source)
        self.assertRegex(trainer_source, r"except Exception[\s\S]+?raise")

    def test_cuda_peak_tracking_initializes_current_device_before_reset(self):
        trainer_source = source("trainers/train.py")
        self.assertIn("torch.cuda.set_device(self.device)", trainer_source)
        self.assertIn("torch.cuda.reset_peak_memory_stats()", trainer_source)
        set_device = trainer_source.index("torch.cuda.set_device(self.device)")
        reset_peak = trainer_source.index("torch.cuda.reset_peak_memory_stats()")
        self.assertLess(set_device, reset_peak)
        self.assertNotIn("torch.cuda.reset_peak_memory_stats(self.device)", trainer_source)

    def test_required_tools_and_four_gpu_script_exist(self):
        for relative_path in (
            "tools/check_environment.py",
            "tools/check_har_dataset.py",
            "tools/smoke_test_lca.py",
            "tools/merge_har_results.py",
            "scripts/run_HAR_4gpu.sh",
        ):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_trainer_can_skip_complete_runs_on_resume(self):
        trainer_source = source("trainers/train.py")
        self.assertIn("_run_complete", trainer_source)
        self.assertIn("skip complete run", trainer_source)


class RunSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT))
        from run_config import parse_run_ids, parse_scenario, resolve_run_ids

        cls.parse_run_ids = staticmethod(parse_run_ids)
        cls.parse_scenario = staticmethod(parse_scenario)
        cls.resolve_run_ids = staticmethod(resolve_run_ids)

    def test_scenario_requires_exactly_two_nonempty_domains(self):
        self.assertEqual(self.parse_scenario("18,14"), [("18", "14")])
        for invalid in ("18", "18,14,6", "18,"):
            with self.assertRaises(ValueError):
                self.parse_scenario(invalid)

    def test_explicit_run_ids_are_unique_and_validated(self):
        self.assertEqual(self.parse_run_ids("0,1,2,1"), [0, 1, 2])
        with self.assertRaises(ValueError):
            self.parse_run_ids("0,-1")

    def test_default_run_ids_follow_num_runs(self):
        self.assertEqual(self.resolve_run_ids(None, 3), [0, 1, 2])
        self.assertEqual(self.resolve_run_ids("2,0", 3), [2, 0])


class ResultMergeTests(unittest.TestCase):
    def test_merge_preserves_metadata_and_writes_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            input_dir = temp / "input" / "task"
            input_dir.mkdir(parents=True)
            result_file = input_dir / "results.csv"
            checkpoint = temp / "checkpoint.pt"
            checkpoint.write_bytes(b"test checkpoint marker")
            fields = [
                "scenario", "run", "acc", "f1_score", "auroc", "status",
                "runtime_seconds", "peak_gpu_memory_mb", "checkpoint_path",
                "git_commit", "python_version", "torch_version", "cuda_version",
            ]
            with result_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "scenario": "18_to_14", "run": "0", "acc": "0.8",
                    "f1_score": "0.7", "auroc": "0.9", "status": "success",
                    "runtime_seconds": "12", "peak_gpu_memory_mb": "100",
                    "checkpoint_path": str(checkpoint), "git_commit": "abc",
                    "python_version": "3.10", "torch_version": "2.1",
                    "cuda_version": "12.1",
                })
            output = temp / "merged.csv"
            completed = subprocess.run(
                [
                    sys.executable, str(ROOT / "tools/merge_har_results.py"),
                    "--input-root", str(temp / "input"), "--output", str(output),
                    "--expected-scenarios", "18_to_14",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["checkpoint_path"], str(checkpoint))
            self.assertEqual(rows[0]["git_commit"], "abc")
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[-2]["status"], "summary_mean")
            self.assertEqual(rows[-1]["status"], "summary_std")

    def test_merge_marks_failed_and_missing_runs_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            input_dir = temp / "input" / "task"
            input_dir.mkdir(parents=True)
            failure = input_dir / "failed_runs.jsonl"
            failure.write_text(
                '{"scenario":"18_to_14","run":0,"exception_type":"RuntimeError"}\n',
                encoding="utf-8",
            )
            output = temp / "merged.csv"
            completed = subprocess.run(
                [
                    sys.executable, str(ROOT / "tools/merge_har_results.py"),
                    "--input-root", str(temp / "input"), "--output", str(output),
                    "--expected-scenarios", "18_to_14", "6_to_13",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            statuses = {(row["scenario"], row["status"]) for row in rows}
            self.assertIn(("18_to_14", "failed"), statuses)
            self.assertIn(("6_to_13", "missing"), statuses)


if __name__ == "__main__":
    unittest.main()
