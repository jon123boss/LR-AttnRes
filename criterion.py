# criterion.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class CriterionConfig:
    ignore_index: int = -100
    reduction: str = "mean"
    z_loss: bool = False
    z_loss_weight: float = 1e-4


class CrossEntropyLoss(nn.Module):
    def __init__(self, config: CriterionConfig, flash_attention = False):
        super().__init__()
        self.config = config
        
        self.flash_attention = False
        self._flash_ce = None

        if flash_attention:
            try:
                import flash_attn  # type: ignore
                from flash_attn.ops.triton.cross_entropy import (  # type: ignore
                    cross_entropy_loss as flash_cross_entropy_loss,
                )
                from packaging import version

                self._flash_ce = flash_cross_entropy_loss

                self.flash_attention = True
            except Exception:
                print("Flash attention not installed, using pytorch Cross Entropy Loss")

    def _fused_cel(self, logits, labels, compute_z_loss, z_loss_weight, mask):
        loss, z_loss = self._flash_ce(
            logits,
            labels,
            label_smoothing=0.0,
            logit_scale=1.0,
            lse_square_scale=z_loss_weight,
            inplace_backward=False,
            process_group=None,
            ignore_index=self.config.ignore_index
        )

        if self.config.reduction == "mean": loss = loss.sum() / mask.sum()
        if self.config.reduction == "sum": loss = loss.sum()

        if not compute_z_loss: return loss, None

        if self.config.reduction == "mean": z_loss = z_loss.sum() / mask.sum()
        if self.config.reduction == "sum": z_loss = z_loss.sum()

        return loss, z_loss

    def forward(self, logits, labels):
        mask = (labels != self.config.ignore_index)
        
        if self.flash_attention:
            loss, z_loss = self._fused_cel(
                logits,
                labels,
                compute_z_loss=self.config.z_loss,
                z_loss_weight=self.config.z_loss_weight,
                mask=mask
            )

            if self.config.z_loss: return loss + z_loss
            return loss

        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=self.config.ignore_index,
            reduction=self.config.reduction,
        )

        if not self.config.z_loss: return loss

        z_squared = logits.logsumexp(dim=-1).pow(2)

        if self.config.reduction == "mean": z_squared = (z_squared * mask).sum() / mask.sum()
        elif self.config.reduction == "sum": z_squared = (z_squared * mask).sum()
        else: z_squared = z_squared * mask

        z_loss = self.config.z_loss_weight * z_squared
        return loss + z_loss

def get_criterion(config):
    criterion = CrossEntropyLoss(
        CriterionConfig(
            ignore_index=config["ignore_index"],
            reduction=config["reduction"],
            z_loss=config["z_loss"],
            z_loss_weight=config["z_loss_weight"],
        ), 
        flash_attention=config["flash_attention"]
        )
    return criterion