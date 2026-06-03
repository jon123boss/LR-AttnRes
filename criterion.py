# criterion.py
import torch.nn as nn
from dataclasses import dataclass

from liger_kernel.transformers import (
    LigerCrossEntropyLoss,
    LigerFusedLinearCrossEntropyLoss,
)


@dataclass
class CriterionConfig:
    ignore_index: int = -100
    reduction: str = "mean"
    z_loss: bool = False
    z_loss_weight: float = 1e-4


class CrossEntropyLoss(nn.Module):
    def __init__(self, config: CriterionConfig):
        super().__init__()
        self.config = config
        lse_square_scale = config.z_loss_weight if config.z_loss else 0.0
        self.cross_entropy = LigerCrossEntropyLoss(
            ignore_index=config.ignore_index,
            lse_square_scale=lse_square_scale,
            reduction=config.reduction,
        )
        self.fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(
            ignore_index=config.ignore_index,
            lse_square_scale=lse_square_scale,
            reduction=config.reduction,
        )

    def forward(self, logits_or_hidden, labels, linear_weight=None, linear_bias=None):
        labels = labels.reshape(-1)
        logits_or_hidden = logits_or_hidden.reshape(-1, logits_or_hidden.size(-1))

        if linear_weight is not None:
            return self.fused_linear_cross_entropy(
                linear_weight,
                logits_or_hidden,
                labels,
                bias=linear_bias,
            )

        return self.cross_entropy(logits_or_hidden, labels)


def get_criterion(config):
    return CrossEntropyLoss(
        CriterionConfig(
            ignore_index=config["ignore_index"],
            reduction=config["reduction"],
            z_loss=config["z_loss"],
            z_loss_weight=config["z_loss_weight"],
        )
    )
