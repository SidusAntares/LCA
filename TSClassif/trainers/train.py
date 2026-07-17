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
            "training_protocol", "metric_protocol", "get_features_returns_z",
            "metric_reset_fixed", "best_epoch", "last_epoch",
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
            "training_protocol": self.training_protocol,
            "metric_protocol": self.metric_protocol,
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
                and checkpoint.get("best") is not None
                and metadata.get("training_protocol") == self.training_protocol
                and metadata.get("metric_protocol") == self.metric_protocol
                and metadata.get("protocol_fingerprint_sha256")
                == self.protocol_audit["fingerprint_sha256"]
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

                    include_target_test = (
                        self.training_protocol == "paper_code_protocol"
                    )
                    self.load_data(
                        src_id,
                        trg_id,
                        include_target_test=include_target_test,
                    )
                    self.initialize_algorithm()
                    self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl,
                                                                             self.loss_avg_meters, self.logger,
                                                                             self.calculate_metrics,
                                                                             self.training_protocol)

                    # Save only after update completed successfully.
                    scenario = f"{src_id}_to_{trg_id}"
                    run_metadata = {
                        "scenario": scenario,
                        "run_id": run_id,
                        "seed": run_id,
                        "training_protocol": self.training_protocol,
                        "metric_protocol": self.metric_protocol,
                        "get_features_returns_z": self.protocol_audit["checks"][
                            "get_features_returns_z"
                        ],
                        "metric_reset_fixed": (
                            self.metric_protocol != "official_stateful_no_reset"
                        ),
                        "best_epoch": self.algorithm.best_epoch,
                        "last_epoch": self.algorithm.last_epoch,
                        "protocol_fingerprint_sha256": self.protocol_audit[
                            "fingerprint_sha256"
                        ],
                    }
                    self.save_checkpoint(self.home_path, self.scenario_log_dir,
                                         self.last_model, self.best_model,
                                         run_metadata)

                    # The one and only default target-test evaluation happens
                    # after training and checkpoint selection have finished.
                    if self.trg_test_dl is None:
                        self.load_target_evaluation_data(trg_id)
                    metrics = self.calculate_metrics()
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
                        self.training_protocol,
                        self.metric_protocol,
                        run_metadata["get_features_returns_z"],
                        run_metadata["metric_reset_fixed"],
                        run_metadata["best_epoch"],
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
