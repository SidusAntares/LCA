import sys

sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    import wandb
except ImportError:  # wandb is not required by the local training flow.
    wandb = None

from compat import load_torch_file
from dataloader.dataloader import data_generator, few_shot_data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import fix_randomness, starting_logs, DictAsObject, AverageMeter
from algorithms.algorithms import get_algorithm_class, LCA
from models.models import get_backbone_class
from run_config import parse_scenario, resolve_run_ids
from reproduction.UCIHAR.protocol_audit import audit_corrected_public_implementation

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.experiment_description = args.dataset
        self.run_description = f"{args.da_method}_{args.exp_name}"

        # paths
        self.home_path = os.getcwd()  # os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        # self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description,
                                        f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # Specify runs
        self.num_runs = args.num_runs
        self.run_ids = resolve_run_ids(args.run_ids, args.num_runs)
        self.training_protocol = args.training_protocol
        self.metric_protocol = args.metric_protocol
        expected_metric_protocol = {
            "paper_code_protocol": "official_stateful_no_reset",
            "baseline_clean_protocol": "stateless_current",
        }[self.training_protocol]
        if self.metric_protocol != expected_metric_protocol:
            raise ValueError(
                f"{self.training_protocol} requires metric protocol "
                f"{expected_metric_protocol}, got {self.metric_protocol}"
            )
        self.protocol_audit = audit_corrected_public_implementation()
        if not self.protocol_audit["passed"]:
            raise RuntimeError(
                "corrected public implementation audit failed: "
                f"{self.protocol_audit['checks']}"
            )

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams

        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                        **self.hparams_class.train_params}
        if args.num_epochs is not None:
            if args.num_epochs < 1:
                raise ValueError("--num_epochs must be at least 1")
            self.hparams["num_epochs"] = args.num_epochs
        if args.scenario is not None:
            self.dataset_configs.scenarios = parse_scenario(args.scenario)

        # metrics
        self.num_classes = self.dataset_configs.num_classes
        # Reproduce the public torchmetrics objects' no-reset behavior without
        # requiring torchmetrics in the offline environment.  This history is
        # used only for the training-reported diagnostic value; the reproduction
        # evaluator uses fresh local arrays for every checkpoint.
        self._official_scores = []
        self._official_labels = []
        self.args = args

        # metrics

    def sweep(self):
        # sweep configurations
        pass

    def initialize_algorithm(self):
        # get algorithm class
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        if self.args.da_method == "LCA":
            self.algorithm = algorithm_class(self.LCA_config, self.dataset_configs, self.hparams,
                                             self.device)
        else:
            self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

    def load_checkpoint(self, model_dir):
        checkpoint = load_torch_file(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        last_model = checkpoint['last']
        best_model = checkpoint['best']
        return last_model, best_model

    def train_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # Training the model
        self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl,
                                                                 self.loss_avg_meters, self.logger,
                                                                 self.calculate_metrics,
                                                                 self.training_protocol)
        return self.last_model, self.best_model

    def evaluate(self, test_loader, is_train=True):
        if isinstance(self.algorithm, LCA):
            self.algorithm.eval()
            if is_train:
                if isinstance(self.algorithm, LCA):
                    self.algorithm.load_state_dict(self.best_model)
                else:
                    self.algorithm.network.load_state_dict(self.best_model)

        else:
            feature_extractor = self.algorithm.feature_extractor.to(self.device)
            classifier = self.algorithm.classifier.to(self.device)
            feature_extractor.eval()
            classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                if self.args.da_method == "LCA":
                    predictions = self.algorithm.inference(data)
                else:
                    features = feature_extractor(data)
                    predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id, include_target_test=False):
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_dl = None
        self.few_shot_dl_5 = None
        if include_target_test:
            self.load_target_evaluation_data(trg_id)

    def load_target_evaluation_data(self, trg_id):
        """Load target test labels only after training (or explicitly for test phase)."""
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")
        self.few_shot_dl_5 = few_shot_data_generator(
            self.trg_test_dl, self.dataset_configs, 5
        )

    def _compute_metrics(self, accumulate=False):
        scores = self.full_preds.detach().cpu().numpy()
        labels = self.full_labels.detach().cpu().numpy()
        if accumulate:
            self._official_scores.append(scores.copy())
            self._official_labels.append(labels.copy())
            scores = np.concatenate(self._official_scores, axis=0)
            labels = np.concatenate(self._official_labels, axis=0)
        predicted = scores.argmax(axis=1)
        acc = accuracy_score(labels, predicted)
        f1 = f1_score(
            labels,
            predicted,
            average="macro",
            labels=list(range(self.num_classes)),
            zero_division=0,
        )
        missing = sorted(set(range(self.num_classes)) - set(np.unique(labels).tolist()))
        if missing:
            warnings.warn(
                f"AUROC is undefined because evaluation labels omit classes {missing}; returning NaN",
                RuntimeWarning,
            )
            auroc = float("nan")
        else:
            try:
                auroc = roc_auc_score(
                    labels,
                    scores,
                    labels=list(range(self.num_classes)),
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError as exc:
                warnings.warn(f"AUROC could not be computed ({exc}); returning NaN", RuntimeWarning)
                auroc = float("nan")
        return acc, f1, auroc

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate(self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc, f1, auroc = self._compute_metrics()

        risks = src_risk, fst_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

    def save_tables_to_file(self, table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir, f"{name}.csv"), index=False)

    def save_checkpoint(self, home_path, log_dir, last_model, best_model, metadata):
        save_dict = {
            "last": last_model,
            "best": best_model,
            "metadata": metadata,
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def calculate_avg_std_wandb_table(self, results):

        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks):

        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}

        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks 
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        if wandb is None:
            raise RuntimeError("wandb is not installed; local training does not require wandb logging")
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self, is_train=True):

        self.evaluate(self.trg_test_dl, is_train)
        return self._compute_metrics(
            accumulate=self.metric_protocol == "official_stateful_no_reset"
        )

    def calculate_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate(self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        return src_risk, fst_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table

    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        numeric_series = {}
        for metric in columns[2:]:
            converted = pd.to_numeric(table[metric], errors="coerce")
            if converted.notna().any():
                numeric_series[metric] = converted
        avg_values = {metric: values.mean() for metric, values in numeric_series.items()}
        std_values = {metric: values.std() for metric, values in numeric_series.items()}

        # Create dataframes for mean and std values
        mean_row = {column: "" for column in columns}
        std_row = {column: "" for column in columns}
        mean_row.update({columns[0]: "mean", columns[1]: "-", **avg_values})
        std_row.update({columns[0]: "std", columns[1]: "-", **std_values})
        mean_metrics_df = pd.DataFrame([mean_row], columns=columns)
        std_metrics_df = pd.DataFrame([std_row], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table
