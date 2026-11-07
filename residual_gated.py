"""Qwen3.8-Flash-Next Gated Residuals (arXiv:2608.30320, Eqs. 30–34).

The read includes branch RMSNorm and gating; callers must not normalize it again.
Reference: Transformers v5.16.0, modeling_qwen4_exp.py (Apache-2.0),
Copyright 2026 The Qwen Team and The HuggingFace Inc. team.
https://github.com/huggingface/transformers/blob/v5.16.0/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py

Optional grouped RMSNorm: FLA (MIT), commit
fbe2c383c191cdbdd7656864305c9eadee1a6756, fla/modules/norm/layernorm.py.
"""

import torch
from torch import nn
from torch.nn import functional as F


class GatedResidualHead(nn.Module):
    """Read a [..., streams, dim] state through Qwen's normalized gated mean."""

    def __init__(self, dim, num_streams=4, rank=None, eps=1e-6, init_std=0.02, norm_backend="torch"):
        super().__init__()
        rank = max(1, dim // 8) if rank is None else rank
        if dim < 1 or num_streams < 1 or rank < 1 or eps <= 0:
            raise ValueError("dim, num_streams, rank and eps must be positive")
        if norm_backend not in {"torch", "fla"}:
            raise ValueError("norm_backend must be 'torch' or 'fla'")
        self.norm_backend = norm_backend
        if norm_backend == "fla":
            try:
                from fla.modules.norm.layernorm import group_norm
            except ImportError as error:
                raise ImportError("FLA norm backend requires flash-linear-attention with grouped norm support") from error
            self._fla_group_norm = group_norm
        self.dim = dim
        self.num_streams = num_streams
        self.eps = eps
        self.norm_weight = nn.Parameter(torch.zeros(num_streams * dim))
        self.mix_down = nn.Linear(num_streams * dim, rank, bias=False)
        self.mix_up = nn.Linear(rank, num_streams * dim, bias=False)
        nn.init.normal_(self.mix_down.weight, std=init_std)
        nn.init.normal_(self.mix_up.weight, std=init_std)

    def _normalize(self, state):
        if state.shape[-2:] != (self.num_streams, self.dim):
            raise ValueError(f"Expected trailing state shape {(self.num_streams, self.dim)}, got {state.shape[-2:]}")
        if self.norm_backend == "fla":
            if not state.is_cuda:
                raise RuntimeError("FLA norm backend requires CUDA tensors")
            return self._fla_group_norm(
                state.flatten(-2), 1 + self.norm_weight.float(), None,
                eps=self.eps, is_rms_norm=True, num_groups=self.num_streams,
            )
        normalized = state.flatten(-2).float().unflatten(-1, (self.num_streams, self.dim))
        normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized.flatten(-2) * (1 + self.norm_weight.float())).to(state.dtype)

    def _mix(self, normalized):
        gate = F.silu(self.mix_down(normalized) / self.num_streams)
        gate = self.mix_up(gate).sigmoid().unflatten(-1, (self.num_streams, self.dim))
        return (gate * normalized.unflatten(-1, (self.num_streams, self.dim))).mean(-2)

    def forward(self, state):
        return self._mix(self._normalize(state))


class GatedResidual(GatedResidualHead):
    """Elementwise gated read, identity bypass, and scalar write per stream."""

    def __init__(self, dim, num_streams=4, rank=None, eps=1e-6, init_std=0.02, norm_backend="torch"):
        super().__init__(dim, num_streams, rank, eps, init_std, norm_backend)
        self.inject = nn.Linear(num_streams * dim, num_streams, bias=False)
        nn.init.normal_(self.inject.weight, std=init_std)

    def read(self, state):
        normalized = self._normalize(state)
        mixed = self._mix(normalized)
        weights = 2 * (self.inject(normalized) / self.num_streams).sigmoid()
        return mixed, weights

    def write(self, state, branch_output, weights):
        return state + weights.unsqueeze(-1) * branch_output.unsqueeze(-2)
