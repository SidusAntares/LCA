"""TorchMetrics 1.3.2 F1 forward/compute compatibility for offline runs."""

from importlib import metadata

import torch


def _confusion_matrix(preds, labels, num_classes):
    preds = preds.detach().view(-1).long().cpu()
    labels = labels.detach().view(-1).long().cpu()
    valid = (labels >= 0) & (labels < num_classes)
    indices = labels[valid] * num_classes + preds[valid]
    return torch.bincount(indices, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def _macro_f1(confusion):
    confusion = confusion.to(torch.float64)
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    denominator = 2 * tp + fp + fn
    per_class = torch.where(denominator > 0, 2 * tp / denominator, 0.0)
    return float(per_class.mean().item())


class TorchMetrics132ForwardCompat:
    """Persistent state with TorchMetrics-1.3.2 current-call forward return."""

    backend = "compat_1_3_2"

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._accumulated = torch.zeros(
            (num_classes, num_classes), dtype=torch.long
        )

    def __call__(self, preds, labels):
        current = _confusion_matrix(preds, labels, self.num_classes)
        self._accumulated += current
        return _macro_f1(current)

    def compute(self):
        return _macro_f1(self._accumulated)

    def reset(self):
        self._accumulated.zero_()


def installed_torchmetrics_version():
    try:
        return metadata.version("torchmetrics")
    except metadata.PackageNotFoundError:
        return "unavailable"


def create_official_f1_metric(num_classes):
    version = installed_torchmetrics_version()
    if version == "1.3.2":
        from torchmetrics import F1Score

        metric = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        metric.backend = "torchmetrics_1.3.2"
        return metric, version
    return TorchMetrics132ForwardCompat(num_classes), version

