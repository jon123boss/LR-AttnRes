"""mHC with DeepSeek-V4's FP32 gates, Sinkhorn order, and stream convention.

Reference: arXiv:2512.24880, equations 7--9, and DeepSeek's reference operators:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/
60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py
The same revision's kernel.py defines the epsilon-stabilized Sinkhorn iteration.
The reference operators are independently expressed in PyTorch.
Fresh training uses the paper's scale 0.01; normal projection std 0.02 and zero
static biases are explicit local choices, not a reproduction of V4 training init.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _project_streams(state, fn, eps):
    flat = state.flatten(-2).float()
    return F.linear(flat, fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)


def _tile_sinkhorn(logits, repeat, eps):
    if logits.device.type != "cuda":
        raise RuntimeError("tile Sinkhorn requires a CUDA device; use sinkhorn_backend='torch'")
    if logits.shape[-1] != 4 or torch.cuda.get_device_capability(logits.device)[0] not in (9, 10):
        raise RuntimeError("tile Sinkhorn requires four streams on SM90/SM100")
    if logits.dtype != torch.float32:
        raise TypeError("tile Sinkhorn requires FP32 logits")
    try:
        from tile_kernels.modeling.mhc.ops.sinkhorn import sinkhorn_normalize
    except ImportError as exc:
        raise ImportError(
            "tile Sinkhorn requires TileKernels revision "
            "79461d72d6f97f91b89403705a841c9d4c1eddd6; see RESIDUAL_BASELINES.md"
        ) from exc
    output = sinkhorn_normalize(logits, repeat=repeat, eps=eps)
    if output.requires_grad:
        # The upstream backward kernel assumes contiguous cotangents.
        output.register_hook(lambda grad: grad.contiguous())
    return output


class MHCConnection(nn.Module):
    """One residual site over ``[..., num_streams, dim]`` states.

    ``comb[i, j]`` routes input stream i to output stream j. Finite Sinkhorn
    iterations approximate double stochasticity; they do not impose it exactly.
    """

    def __init__(self, dim, num_streams=4, sinkhorn_iters=20, norm_eps=1e-6,
                 gate_eps=1e-6, scale_init=0.01, init_std=0.02, sinkhorn_backend="torch"):
        super().__init__()
        if sinkhorn_backend not in {"torch", "tile"}:
            raise ValueError("sinkhorn_backend must be 'torch' or 'tile'")
        self.sinkhorn_backend = sinkhorn_backend
        if dim < 1 or num_streams < 1 or sinkhorn_iters < 1:
            raise ValueError("dim, num_streams and sinkhorn_iters must be positive")
        if not (math.isfinite(norm_eps) and norm_eps > 0
                and math.isfinite(gate_eps) and gate_eps > 0):
            raise ValueError("norm_eps and gate_eps must be finite and positive")
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.norm_eps = norm_eps
        self.gate_eps = gate_eps
        size = num_streams * (num_streams + 2)
        self.fn = nn.Parameter(torch.empty(size, num_streams * dim, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(size, dtype=torch.float32))
        self.scale = nn.Parameter(torch.full((3,), scale_init, dtype=torch.float32))
        nn.init.normal_(self.fn, std=init_std)

    def read(self, state):
        n, eps = self.num_streams, self.gate_eps
        with torch.autocast(device_type=state.device.type, enabled=False):
            mixes = _project_streams(state, self.fn, self.norm_eps)
            pre, post, comb = mixes.split((n, n, n * n), dim=-1)
            pre_base, post_base, comb_base = self.base.float().split((n, n, n * n))
            pre = torch.sigmoid(pre * self.scale[0].float() + pre_base) + eps
            post = 2 * torch.sigmoid(post * self.scale[1].float() + post_base)
            logits = (comb * self.scale[2].float() + comb_base).unflatten(-1, (n, n))
            if self.sinkhorn_backend == "tile":
                comb = _tile_sinkhorn(logits, self.sinkhorn_iters, eps)
            else:
                comb = logits.softmax(dim=-1) + eps
                comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
                for _ in range(self.sinkhorn_iters - 1):
                    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
                    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
            collapsed = (pre.unsqueeze(-1) * state.float()).sum(dim=-2)
        return collapsed.to(state.dtype), (post, comb)

    def write(self, state, branch, context):
        post, comb = context
        with torch.autocast(device_type=state.device.type, enabled=False):
            mixed = (comb.unsqueeze(-1) * state.float().unsqueeze(-2)).sum(dim=-3)
            output = mixed + post.unsqueeze(-1) * branch.float().unsqueeze(-2)
        return output.to(state.dtype)


class MHCHead(nn.Module):
    """Learned stream collapse before the backbone's final normalization."""

    def __init__(self, dim, num_streams=4, norm_eps=1e-6, gate_eps=1e-6,
                 scale_init=0.01, init_std=0.02):
        super().__init__()
        if dim < 1 or num_streams < 1:
            raise ValueError("dim and num_streams must be positive")
        if not (math.isfinite(norm_eps) and norm_eps > 0
                and math.isfinite(gate_eps) and gate_eps > 0):
            raise ValueError("norm_eps and gate_eps must be finite and positive")
        self.norm_eps = norm_eps
        self.gate_eps = gate_eps
        self.fn = nn.Parameter(torch.empty(num_streams, num_streams * dim, dtype=torch.float32))
        self.base = nn.Parameter(torch.zeros(num_streams, dtype=torch.float32))
        self.scale = nn.Parameter(torch.full((1,), scale_init, dtype=torch.float32))
        nn.init.normal_(self.fn, std=init_std)

    def forward(self, state):
        with torch.autocast(device_type=state.device.type, enabled=False):
            mixes = _project_streams(state, self.fn, self.norm_eps)
            pre = torch.sigmoid(mixes * self.scale.float() + self.base.float()) + self.gate_eps
            collapsed = (pre.unsqueeze(-1) * state.float()).sum(dim=-2)
        return collapsed.to(state.dtype)
