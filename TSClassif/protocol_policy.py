"""Central authority for all allowed LCA baseline protocol differences."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ProtocolPolicy:
    name: str
    short_name: str
    classifier_output_type: str
    cross_entropy_input_type: str
    evaluate_target_during_training: bool
    restore_mode_after_evaluation: bool
    cumulative_epoch_meters: bool
    checkpoint_selection_rule: str
    primary_checkpoint: str
    pseudo_probability_rule: str
    metric_protocol: str
    required_target_test_reads_during_training: Optional[int]

    @property
    def output_namespace(self):
        return self.name

    def probabilities(self, outputs):
        if self.classifier_output_type == "logits":
            return torch.softmax(outputs, dim=1)
        return outputs

    def pseudo_probabilities(self, outputs):
        if self.pseudo_probability_rule == "upstream_softmax_on_probabilities":
            return torch.softmax(outputs, dim=-1)
        return torch.softmax(outputs, dim=-1)

    def cross_entropy_input(self, outputs):
        return outputs

    def begin_epoch(self, model, meters):
        if not self.cumulative_epoch_meters:
            model.train()
            meters.clear()

    def checkpoint_candidate(self, epoch, meters):
        if self.checkpoint_selection_rule == "upstream_cumulative_src_cls_every_10":
            return (epoch + 1) % 10 == 0, meters["Src_cls_loss"].avg
        return True, meters["source_supervised_loss"].avg


_POLICIES = {
    "corrected_public": ProtocolPolicy(
        name="corrected_public",
        short_name="LCA-CP",
        classifier_output_type="probabilities",
        cross_entropy_input_type="probabilities",
        evaluate_target_during_training=True,
        restore_mode_after_evaluation=False,
        cumulative_epoch_meters=True,
        checkpoint_selection_rule="upstream_cumulative_src_cls_every_10",
        primary_checkpoint="best_source",
        pseudo_probability_rule="upstream_softmax_on_probabilities",
        metric_protocol="torchmetrics_1_3_2_forward_persistent_state",
        required_target_test_reads_during_training=None,
    ),
    "clean_baseline": ProtocolPolicy(
        name="clean_baseline",
        short_name="LCA-Clean",
        classifier_output_type="logits",
        cross_entropy_input_type="logits",
        evaluate_target_during_training=False,
        restore_mode_after_evaluation=True,
        cumulative_epoch_meters=False,
        checkpoint_selection_rule="current_epoch_source_supervised",
        primary_checkpoint="last",
        pseudo_probability_rule="softmax_on_logits",
        metric_protocol="sklearn_stateless_macro_f1",
        required_target_test_reads_during_training=0,
    ),
}


def get_protocol_policy(name):
    try:
        return _POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown protocol: {name}") from exc


PROTOCOL_NAMES = tuple(_POLICIES)
