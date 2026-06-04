# criterion.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from liger_utils import get_liger_kernel, tensor_supports_liger


@dataclass
class CriterionConfig:
    ignore_index: int = -100
    reduction: str = "mean"
    z_loss: bool = False
    z_loss_weight: float = 1e-4
    liger_cross_entropy: bool = False
    liger_fused_linear_cross_entropy: bool = False
    liger_strict: bool = False


class CrossEntropyLoss(nn.Module):
    def __init__(self, config: CriterionConfig, flash_attention=False):
        super().__init__()
        self.config = config
        self.uses_fused_linear = config.liger_fused_linear_cross_entropy

        self.flash_attention = False
        self._flash_ce = None
        self._liger_ce = None
        self._liger_fused_ce = None

        if config.liger_cross_entropy:
            liger_ce_cls = get_liger_kernel("cross_entropy")
            if liger_ce_cls is None:
                if config.liger_strict:
                    raise RuntimeError("liger_cross_entropy=True but LigerCrossEntropyLoss is unavailable")
            else:
                self._liger_ce = liger_ce_cls(
                    ignore_index=config.ignore_index,
                    lse_square_scale=config.z_loss_weight if config.z_loss else 0.0,
                    reduction=config.reduction,
                )

        if config.liger_fused_linear_cross_entropy:
            liger_fused_ce_cls = get_liger_kernel("fused_linear_cross_entropy")
            if liger_fused_ce_cls is None:
                if config.liger_strict:
                    raise RuntimeError(
                        "liger_fused_linear_cross_entropy=True but "
                        "LigerFusedLinearCrossEntropyLoss is unavailable"
                    )
            else:
                self._liger_fused_ce = liger_fused_ce_cls(
                    ignore_index=config.ignore_index,
                    lse_square_scale=config.z_loss_weight if config.z_loss else 0.0,
                    reduction=config.reduction,
                )

        if flash_attention and self._liger_ce is None and self._liger_fused_ce is None:
            try:
                from flash_attn.ops.triton.cross_entropy import (  # type: ignore
                    cross_entropy_loss as flash_cross_entropy_loss,
                )

                self._flash_ce = flash_cross_entropy_loss
                self.flash_attention = True
            except Exception:
                print("Flash attention not installed, using pytorch Cross Entropy Loss")

    def _standard_ce(self, logits, labels, mask):
        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=self.config.ignore_index,
            reduction=self.config.reduction,
        )

        if not self.config.z_loss:
            return loss

        z_squared = logits.logsumexp(dim=-1).pow(2)

        if self.config.reduction == "mean":
            z_squared = (z_squared * mask).sum() / mask.sum()
        elif self.config.reduction == "sum":
            z_squared = (z_squared * mask).sum()
        else:
            z_squared = z_squared * mask

        z_loss = self.config.z_loss_weight * z_squared
        return loss + z_loss

    def _fused_cel(self, logits, labels, compute_z_loss, z_loss_weight, mask):
        loss, z_loss = self._flash_ce(
            logits,
            labels,
            label_smoothing=0.0,
            logit_scale=1.0,
            lse_square_scale=z_loss_weight,
            inplace_backward=False,
            process_group=None,
            ignore_index=self.config.ignore_index,
        )

        if self.config.reduction == "mean":
            loss = loss.sum() / mask.sum()
        if self.config.reduction == "sum":
            loss = loss.sum()

        if not compute_z_loss:
            return loss, None

        if self.config.reduction == "mean":
            z_loss = z_loss.sum() / mask.sum()
        if self.config.reduction == "sum":
            z_loss = z_loss.sum()

        return loss, z_loss

    def forward(
        self,
        logits,
        labels,
        lm_head_weight=None,
        lm_head_bias=None,
        fallback_logits_dtype=None,
    ):
        if labels.dim() != 1:
            labels = labels.reshape(-1)
        if logits.dim() != 2:
            logits = logits.reshape(-1, logits.size(-1))

        mask = labels != self.config.ignore_index

        if self.uses_fused_linear:
            if lm_head_weight is None:
                raise ValueError("lm_head_weight is required for fused linear cross entropy")

            if self._liger_fused_ce is not None and tensor_supports_liger(logits):
                return self._liger_fused_ce(lm_head_weight, logits, labels, bias=lm_head_bias)

            materialized_logits = F.linear(logits, lm_head_weight, lm_head_bias)
            if fallback_logits_dtype is not None:
                materialized_logits = materialized_logits.to(fallback_logits_dtype)
            return self._standard_ce(materialized_logits, labels, mask)

        if self._liger_ce is not None and tensor_supports_liger(logits):
            return self._liger_ce(logits, labels)

        if self.flash_attention:
            loss, z_loss = self._fused_cel(
                logits,
                labels,
                compute_z_loss=self.config.z_loss,
                z_loss_weight=self.config.z_loss_weight,
                mask=mask,
            )

            if self.config.z_loss:
                return loss + z_loss
            return loss

        return self._standard_ce(logits, labels, mask)


def get_criterion(config):
    return CrossEntropyLoss(
        CriterionConfig(
            ignore_index=config["ignore_index"],
            reduction=config["reduction"],
            z_loss=config["z_loss"],
            z_loss_weight=config["z_loss_weight"],
            liger_cross_entropy=config.get("liger_cross_entropy", False),
            liger_fused_linear_cross_entropy=config.get("liger_fused_linear_cross_entropy", False),
            liger_strict=config.get("liger_strict", False),
        ),
        flash_attention=config["flash_attention"],
    )
