# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0:
# https://www.apache.org/licenses/LICENSE-2.0
"""Hy4 identity Hyper-Connections, adapted to the residual-read interface.

Source: huggingface/transformers@0f53828e83f2212060233d2ea552cc76617b8712,
src/transformers/models/hy_v4/modeling_hy_v4.py (HYV4HyperConnection/Head).
Controllers use the Hy4 recipe: four streams, weight std .006, scale .01,
read bias -log(N-1), write bias zero, and write magnitude two. The caller
replicates embeddings across streams and applies its final norm after IHCHead.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class IHCConnection(nn.Module):
    def __init__(self, dim, num_streams=4, norm_eps=1e-5, gate_eps=1e-6):
        super().__init__()
        if dim < 1 or num_streams < 1:
            raise ValueError("dim and num_streams must be positive")
        self.num_streams = num_streams
        self.norm_eps = norm_eps
        self.gate_eps = gate_eps
        self.fn = nn.Parameter(torch.empty(2 * num_streams, num_streams * dim, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(2 * num_streams, dtype=torch.float32))
        self.scale = nn.Parameter(torch.full((2,), 0.01, dtype=torch.float32))
        nn.init.normal_(self.fn, std=0.006)
        nn.init.constant_(self.base[:num_streams], -math.log(max(num_streams - 1, 1)))

    def read(self, state):
        with torch.autocast(device_type=state.device.type, enabled=False):
            flat = state.flatten(-2).float()
            mixes = F.linear(flat, self.fn.float()) * torch.rsqrt(
                flat.square().mean(-1, keepdim=True) + self.norm_eps
            )
            pre_logits, post_logits = mixes.split(self.num_streams, dim=-1)
            pre_bias, post_bias = self.base.float().split(self.num_streams)
            pre_scale, post_scale = self.scale.float().unbind()
            pre = torch.sigmoid(pre_logits * pre_scale + pre_bias) + self.gate_eps
            post = 2 * torch.sigmoid(post_logits * post_scale + post_bias) + self.gate_eps
            branch_input = (pre.unsqueeze(-1) * state.float()).sum(-2)
        return branch_input.to(state.dtype), post

    @staticmethod
    def write(state, branch, post):
        with torch.autocast(device_type=state.device.type, enabled=False):
            updated = state.float() + post.float().unsqueeze(-1) * branch.float().unsqueeze(-2)
        return updated.to(state.dtype)


class IHCHead(nn.Module):
    """Learned stream readout, before the backbone's final RMSNorm."""

    def __init__(self, dim, num_streams=4, norm_eps=1e-5, gate_eps=1e-6):
        super().__init__()
        if dim < 1 or num_streams < 1:
            raise ValueError("dim and num_streams must be positive")
        self.norm_eps = norm_eps
        self.gate_eps = gate_eps
        self.fn = nn.Parameter(torch.empty(num_streams, num_streams * dim, dtype=torch.float32))
        self.base = nn.Parameter(torch.full(
            (num_streams,), -math.log(max(num_streams - 1, 1)), dtype=torch.float32
        ))
        self.scale = nn.Parameter(torch.full((1,), 0.01, dtype=torch.float32))
        nn.init.normal_(self.fn, std=0.006)

    def forward(self, state):
        with torch.autocast(device_type=state.device.type, enabled=False):
            flat = state.flatten(-2).float()
            mixes = F.linear(flat, self.fn.float()) * torch.rsqrt(
                flat.square().mean(-1, keepdim=True) + self.norm_eps
            )
            pre = torch.sigmoid(mixes * self.scale.float() + self.base.float()) + self.gate_eps
            out = (pre.unsqueeze(-1) * state.float()).sum(-2)
        return out.to(state.dtype)
