import csv
import json
import tempfile
import unittest
from pathlib import Path

from TSClassif.reproduction.UCIHAR.summarize_results import summarize
from TSClassif.reproduction.UCIHAR.result_protocol import reported_metric_fields
from TSClassif.reproduction.UCIHAR.compare_protocol_results import compare_rows


FIELDS = [
    "scenario", "source", "target", "run_id", "seed",
    "best_clean_macro_f1", "last_clean_macro_f1",
    "official_reported_f1", "accuracy", "status",
    "runtime_seconds", "checkpoint_best", "checkpoint_last", "git_commit",
    "python_version", "torch_version", "cuda_version", "gpu_name",
    "get_features_returns_z", "metric_reset_fixed",
    "training_protocol", "metric_protocol", "clean_metric_protocol",
    "current_reported_f1", "best_epoch", "last_epoch",
    "best_state_sha256", "last_state_sha256", "is_legacy",
]


def write_rows(path, scenario, best_values, paper_offset=0.0, legacy=False):
    source, target = scenario.split("_to_")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for run_id, value in enumerate(best_values):
            writer.writerow({
                "scenario": scenario,
                "source": source,
                "target": target,
                "run_id": run_id,
                "seed": run_id,
                "best_clean_macro_f1": value,
                "last_clean_macro_f1": value - 0.01,
                "official_reported_f1": value + paper_offset,
                "current_reported_f1": "",
                "accuracy": value,
                "status": "success",
                "get_features_returns_z": "true",
                "metric_reset_fixed": "true",
                "training_protocol": "paper_code_protocol",
                "metric_protocol": "official_stateful_no_reset",
                "clean_metric_protocol": "clean_checkpoint",
                "is_legacy": str(legacy).lower(),
            })


class UciharSummaryTests(unittest.TestCase):
    def test_diagnostic_script_is_explicit_and_does_not_start_full_run(self):
        root = Path(__file__).resolve().parents[1]
        script_path = root / "reproduction" / "UCIHAR" / "run_diagnostic.sh"
        self.assertTrue(script_path.is_file())
        script = script_path.read_text(encoding="utf-8")
        for gpu, scenario in ((0, "20,9"), (1, "7,18"), (2, "9,19")):
            self.assertIn(f"run_task {gpu} {scenario}", script)
        self.assertIn('CUDA_VISIBLE_DEVICES="$gpu"', script)
        self.assertIn('--scenario "$scenario"', script)
        self.assertIn("--run_ids 0,1,2", script)
        self.assertIn("--num_epochs 40", script)
        self.assertIn("--training_protocol", script)
        self.assertIn("--metric_protocol", script)
        self.assertIn("summarize_results.py", script)
        self.assertNotIn("exec bash", script)
        self.assertNotIn('bash "$OUT/run_full.sh"', script)
        self.assertIn(
            "Diagnostic passed. Review diagnostic files, then run run_full.sh manually.",
            script,
        )

    def test_diagnostic_summary_passes_when_all_thresholds_hold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for scenario, values in (
                ("20_to_9", [0.69, 0.70, 0.695]),
                ("7_to_18", [0.90, 0.91, 0.92]),
                ("9_to_19", [0.96, 0.97, 0.98]),
            ):
                path = root / f"{scenario}.csv"
                write_rows(path, scenario, values)
                inputs.append(path)
            decision = summarize(inputs, root / "out", diagnostic=True)
            self.assertFalse(decision["stop"])
            self.assertEqual(decision["completed_runs"], 9)
            self.assertAlmostEqual(decision["paper_overall_mean"], 0.8582)
            self.assertAlmostEqual(
                decision["reproduced_overall_mean"],
                (0.695 + 0.91 + 0.97) / 3,
            )
            self.assertTrue((root / "out" / "raw_runs.csv").is_file())
            self.assertTrue((root / "out" / "scenario_summary.csv").is_file())
            self.assertTrue((root / "out" / "paper_comparison.csv").is_file())
            report = (root / "out" / "REPRODUCTION_REPORT.md")
            self.assertTrue(report.is_file())
            report_text = report.read_text(encoding="utf-8")
            self.assertIn("corrected public implementation", report_text)
            self.assertIn("20→9", report_text)

    def test_diagnostic_summary_warns_on_public_clean_metric_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for scenario, values in (
                ("20_to_9", [0.69, 0.70, 0.695]),
                ("7_to_18", [0.90, 0.91, 0.92]),
                ("9_to_19", [0.96, 0.97, 0.98]),
            ):
                path = root / f"{scenario}.csv"
                write_rows(path, scenario, values, paper_offset=0.06)
                inputs.append(path)
            decision = summarize(inputs, root / "out", diagnostic=True)
            self.assertFalse(decision["stop"])
            self.assertIn("official_clean_gap_over_0.05", decision["warnings"])
            self.assertNotIn("official_clean_gap_over_0.05", decision["reasons"])

    def test_legacy_rows_never_enter_formal_outputs_or_completion_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = root / "formal.csv"
            legacy = root / "legacy.csv"
            write_rows(formal, "20_to_9", [0.69, 0.70, 0.695])
            write_rows(legacy, "18_to_14", [1.0], legacy=True)
            decision = summarize([formal, legacy], root / "out", diagnostic=True)
            with (root / "out" / "raw_runs.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["scenario"] for row in rows}, {"20_to_9"})
            self.assertEqual(decision["completed_runs"], 3)

    def test_reported_metric_fields_only_populates_official_for_official_protocol(self):
        self.assertEqual(
            reported_metric_fields({
                "metric_protocol": "official_stateful_no_reset",
                "f1_score": "0.8",
            }),
            {"official_reported_f1": "0.8", "current_reported_f1": ""},
        )
        self.assertEqual(
            reported_metric_fields({
                "metric_protocol": "stateless_current",
                "f1_score": "0.7",
            }),
            {"official_reported_f1": "", "current_reported_f1": "0.7"},
        )

    def test_full_script_retrains_all_three_18_to_14_seeds(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "reproduction" / "UCIHAR" / "run_full.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("run_ids=1,2", script)
        self.assertIn("--run_ids 0,1,2", script)
        self.assertIn("--training_protocol", script)
        self.assertIn("--metric_protocol", script)
        self.assertIn("legacy_18_to_14_run0.csv", script)
        summary_call = script.split('python "$OUT/summarize_results.py"', 1)[1]
        self.assertNotIn("legacy_18_to_14_run0.csv", summary_call)

    def test_protocol_comparison_entry_compares_hash_epoch_and_clean_f1(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "reproduction" / "UCIHAR" / "run_protocol_comparison.sh"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        self.assertIn("--scenario 20,9", text)
        self.assertIn("--run_ids 0", text)
        self.assertIn("paper_code_protocol", text)
        self.assertIn("baseline_clean_protocol", text)
        self.assertIn("compare_protocol_results.py", text)

    def test_protocol_comparison_reports_parameter_epoch_and_metric_equality(self):
        paper = {
            "best_state_sha256": "abc",
            "last_state_sha256": "def",
            "best_epoch": "39",
            "best_clean_macro_f1": "0.70",
            "last_clean_macro_f1": "0.69",
        }
        clean = dict(paper)
        comparison = compare_rows(paper, clean)
        self.assertTrue(comparison["best_state_sha256_equal"])
        self.assertTrue(comparison["last_state_sha256_equal"])
        self.assertTrue(comparison["best_epoch_equal"])
        self.assertTrue(comparison["best_clean_macro_f1_equal"])


if __name__ == "__main__":
    unittest.main()
