import sys
import json
import platform
import subprocess
import time
import torch
import os
import pandas as pd
import numpy as np
import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from trainers.abstract_trainer import AbstractTrainer
from compat import load_torch_file
from checkpoint_metadata import validate_checkpoint_metadata
import traceback


warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super().__init__(args)

        self.results_columns = [
            "scenario", "run", "acc", "f1_score", "auroc", "status",
            "runtime_seconds", "peak_gpu_memory_mb", "checkpoint_path",
            "git_commit", "python_version", "torch_version", "cuda_version",
            "protocol", "official_forward_f1",
            "official_accumulated_compute_f1_audit",
            "target_test_reads_during_training", "classifier_output_type",
            "cross_entropy_input_type", "best_source_epoch", "last_epoch",
            "protocol_fingerprint_sha256",
        ]
        self.evaluation_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        self.risks_columns = ["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"]

    def _git_commit(self):
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.home_path, text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def _record_failure(self, src_id, trg_id, run_id, exc):
        failure = {
            "scenario": f"{src_id}_to_{trg_id}",
            "run": run_id,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "protocol": self.protocol,
        }
        failure_path = os.path.join(self.exp_log_dir, "failed_runs.jsonl")
        with open(failure_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        return failure

    def _load_existing_table(self, name, columns):
        path = os.path.join(self.exp_log_dir, f"{name}.csv")
        if not os.path.isfile(path):
            return pd.DataFrame(columns=columns)
        table = pd.read_csv(path)
        if not set(columns).issubset(table.columns):
            return pd.DataFrame(columns=columns)
        # Summary rows are regenerated after real run rows are updated.
        table = table[~table["scenario"].astype(str).isin(["mean", "std"])]
        table = table[~table["run"].astype(str).isin(["mean", "std", "-"])]
        return table[columns]

    def _run_complete(self, table_results, src_id, trg_id, run_id):
        scenario = f"{src_id}_to_{trg_id}"
        rows = table_results[
            (table_results["scenario"].astype(str) == scenario)
            & (table_results["run"].astype(str) == str(run_id))
            & (table_results["status"].astype(str) == "success")
        ]
        if rows.empty:
            return False
        log_dir = os.path.join(self.exp_log_dir, f"{scenario}_run_{run_id}")
        try:
            checkpoint = load_torch_file(os.path.join(log_dir, "checkpoint.pt"))
            metadata = checkpoint.get("metadata", {})
            return (
                checkpoint.get("last") is not None
                and checkpoint.get("best_source") is not None
                and validate_checkpoint_metadata(
                    metadata, self.policy,
                    self.protocol_audit["fingerprint_sha256"],
                ) is metadata
            )
        except Exception:
            return False

    def fit(self):

        # table with metrics
        table_results = self._load_existing_table('results', self.results_columns)

        # table with risks
        table_risks = self._load_existing_table('risks', self.risks_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in self.run_ids:
                if self._run_complete(table_results, src_id, trg_id, run_id):
                    print(f"skip complete run {src_id}->{trg_id} run={run_id}")
                    continue
                # fixing random seed
                fix_randomness(run_id)
                self.target_test_reads_during_training = 0

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                try:
                    started_at = time.perf_counter()
                    if self.device.type == "cuda":
                        # PyTorch 2.3 can reject a torch.device passed to the
                        # memory API before the CUDA context has been selected.
                        torch.cuda.set_device(self.device)
                        torch.cuda.reset_peak_memory_stats()

                    include_target_test = self.policy.evaluate_target_during_training
                    self.load_data(
                        src_id,
                        trg_id,
                        include_target_test=include_target_test,
                    )
                    self.initialize_algorithm()
                    self.training_active = True
                    try:
                        self.last_model, self.best_model = self.algorithm.update(
                            self.src_train_dl, self.trg_train_dl,
                            self.loss_avg_meters, self.logger,
                            self.calculate_metrics,
                        )
                    finally:
                        self.training_active = False

                    # Save only after update completed successfully.
                    scenario = f"{src_id}_to_{trg_id}"
                    run_metadata = {
                        "protocol": self.protocol,
                        "upstream_commit": "45c091fca909ac13675c6ddac7e0464f0a186355",
                        "current_commit": self._git_commit(),
                        "scenario": scenario,
                        "run_id": run_id,
                        "seed": run_id,
                        "epoch": self.algorithm.last_epoch,
                        "model_training_mode": self.algorithm.training,
                        "epoch_mode_trace": self.algorithm.epoch_mode_trace,
                        "classifier_output_type": self.policy.classifier_output_type,
                        "cross_entropy_input_type": self.policy.cross_entropy_input_type,
                        "target_test_reads_during_training": self.target_test_reads_during_training,
                        "checkpoint_selection_rule": self.policy.checkpoint_selection_rule,
                        "primary_checkpoint": self.policy.primary_checkpoint,
                        "get_features_returns_z": self.protocol_audit["checks"][
                            "get_features_returns_z"
                        ],
                        "base_dist_mean_registered": (
                            "base_dist_mean" in dict(self.algorithm.named_buffers())
                        ),
                        "prior_enabled": not self.algorithm.config.No_prior,
                        "loss_weights": {
                            "class": self.algorithm.config.class_weight,
                            "reconstruction": self.algorithm.config.rec_weight,
                            "sparsity": self.algorithm.config.sparsity_weight,
                            "kl": self.algorithm.config.z_kl_weight,
                            "structure": self.algorithm.config.structure_weight,
                        },
                        "pseudo_threshold": self.algorithm.config.tar_psuedo_thre,
                        "pseudo_start_epoch": self.algorithm.config.start_psuedo_step,
                        "python_version": platform.python_version(),
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda or "none",
                        "gpu_name": (
                            torch.cuda.get_device_name(self.device)
                            if self.device.type == "cuda" else "cpu"
                        ),
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "deterministic_algorithms_enabled": (
                            torch.are_deterministic_algorithms_enabled()
                        ),
                        "metric_backend": self.metric_backend,
                        "torchmetrics_version": self.torchmetrics_version,
                        "best_source_epoch": self.algorithm.best_epoch,
                        "last_epoch": self.algorithm.last_epoch,
                        "protocol_fingerprint_sha256": self.protocol_audit[
                            "fingerprint_sha256"
                        ],
                    }
                    validate_checkpoint_metadata(
                        run_metadata, self.policy,
                        self.protocol_audit["fingerprint_sha256"],
                    )
                    self.save_checkpoint(self.home_path, self.scenario_log_dir,
                                         self.last_model, self.best_model,
                                         run_metadata)

                    # The one and only default target-test evaluation happens
                    # after training and checkpoint selection have finished.
                    if self.trg_test_dl is None:
                        self.load_target_evaluation_data(trg_id)
                    primary_state = (
                        self.last_model
                        if self.policy.primary_checkpoint == "last"
                        else self.best_model
                    )
                    self.algorithm.load_state_dict(primary_state)
                    metrics = self.calculate_metrics(is_train=False)
                    risks = self.calculate_risks()
                    runtime_seconds = time.perf_counter() - started_at
                    peak_gpu_memory_mb = (
                        torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                        if self.device.type == "cuda" else 0.0
                    )

                    checkpoint_path = os.path.abspath(
                        os.path.join(self.scenario_log_dir, "checkpoint.pt")
                    )
                    result_values = (
                        *metrics,
                        "success",
                        runtime_seconds,
                        peak_gpu_memory_mb,
                        checkpoint_path,
                        self._git_commit(),
                        platform.python_version(),
                        torch.__version__,
                        torch.version.cuda or "none",
                        self.protocol,
                        self.official_forward_f1,
                        self.official_accumulated_compute_f1_audit,
                        run_metadata["target_test_reads_during_training"],
                        run_metadata["classifier_output_type"],
                        run_metadata["cross_entropy_input_type"],
                        run_metadata["best_source_epoch"],
                        run_metadata["last_epoch"],
                        run_metadata["protocol_fingerprint_sha256"],
                    )
                    table_results = self.append_results_to_tables(
                        table_results, scenario, run_id, result_values
                    )
                    table_risks = self.append_results_to_tables(
                        table_risks, scenario, run_id, risks
                    )
                    self.save_tables_to_file(table_results, 'results')
                    self.save_tables_to_file(table_risks, 'risks')
                except Exception as e:
                    failure = self._record_failure(src_id, trg_id, run_id, e)
                    print(failure["traceback"], file=sys.stderr)
                    raise
        print(table_results)
        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, self.results_columns)
        table_risks = self.add_mean_std_table(table_risks, self.risks_columns)

        # Save tables to file if needed
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')

    def test(self):
        # Results dataframes
        last_results = pd.DataFrame(columns=self.evaluation_columns)
        best_results = pd.DataFrame(columns=self.evaluation_columns)

        # Cross-domain scenarios
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in self.run_ids:
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))

                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id, include_target_test=True)

                # Build model
                self.initialize_algorithm()

                # Load chechpoint 
                last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)

                # Testing the last model
                if self.args.da_method == "LCA":
                    self.algorithm.load_state_dict(last_chk)
                else:
                    self.algorithm.network.load_state_dict(last_chk)
                self.evaluate(self.trg_test_dl,is_train=False)
                last_metrics = self.calculate_metrics(is_train=False)
                last_results = self.append_results_to_tables(last_results, f"{src_id}_to_{trg_id}", run_id,
                                                             last_metrics)

                # Testing the best model
                if self.args.da_method == "LCA":
                    self.algorithm.load_state_dict(best_chk)
                else:
                    self.algorithm.network.load_state_dict(best_chk)
                self.evaluate(self.trg_test_dl,is_train=False)
                best_metrics = self.calculate_metrics(is_train=False)
                # Append results to tables
                best_results = self.append_results_to_tables(best_results, f"{src_id}_to_{trg_id}", run_id,
                                                             best_metrics)

        last_scenario_mean_std = last_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])
        best_scenario_mean_std = best_results.groupby('scenario')[['acc', 'f1_score', 'auroc']].agg(['mean', 'std'])

        # Save tables to file if needed
        self.save_tables_to_file(last_scenario_mean_std, 'last_results')
        self.save_tables_to_file(best_scenario_mean_std, 'best_results')

        # printing summary 
        summary_last = {metric: np.mean(last_results[metric]) for metric in self.evaluation_columns[2:]}
        summary_best = {metric: np.mean(best_results[metric]) for metric in self.evaluation_columns[2:]}
        for summary_name, summary in [('Last', summary_last), ('Best', summary_best)]:
            for key, val in summary.items():
                print(f'{summary_name}: {key}\t: {val:2.4f}')
