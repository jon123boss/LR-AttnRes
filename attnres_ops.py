"""Fused Attention Residual read kernels.

This module contains the optional Triton-backed read path used by ``model.py``.
The public functions keep the same tensor semantics as the PyTorch reference
path and fall back to ordinary PyTorch when Triton is unavailable or the input
layout is unsupported.
"""

import math
import os

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

__all__ = [
    "attention_residual_average_read",
    "attention_residual_phase1",
    "attention_residual_phase1_from_logits",
    "attention_residual_phase2",
    "attention_residual_phase2_from_logit",
    "attention_residual_phase2_torch",
    "attention_residual_read",
    "attention_residual_read_torch",
    "is_fused_attnres_available",
    "lrid_attention_residual_phase2",
    "lrid_attention_residual_phase2_torch",
    "lrid_attention_residual_read",
    "lrid_attention_residual_read_torch",
]


def _rms_norm_eps(x: Tensor, eps: float = None) -> float:
    if eps is not None:
        return eps
    return torch.finfo(x.dtype).eps


def _norm(x: Tensor, eps: float = None):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),), eps=eps)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + _rms_norm_eps(x, eps))


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _as_stacked(values):
    if isinstance(values, Tensor):
        return values
    return torch.stack(values, dim=0)


def _source_log_count_bias(source_counts, n_sources: int, device) -> Tensor:
    if source_counts is None:
        return None
    if isinstance(source_counts, Tensor):
        counts = source_counts.reshape(-1).to(device=device, dtype=torch.float32)
    elif isinstance(source_counts, (int, float)):
        counts = torch.full((n_sources,), float(source_counts), device=device, dtype=torch.float32)
    else:
        if len(source_counts) != n_sources:
            raise RuntimeError("source_counts length must match the number of sources")
        count_values = [float(count) for count in source_counts]
        if any(count <= 0.0 for count in count_values):
            raise RuntimeError("source_counts must be positive")
        counts = torch.tensor(count_values, device=device, dtype=torch.float32)
    if counts.numel() != n_sources:
        raise RuntimeError("source_counts length must match the number of sources")
    return counts.log()


def is_fused_attnres_available() -> bool:
    return _TRITON_AVAILABLE


def _use_triton_training_kernel() -> bool:
    return _training_kernel_mode() in {"auto", "triton"}


def _training_kernel_mode() -> str:
    return os.environ.get("ATTNRES_TRAIN_KERNEL", "auto").lower()


if _TRITON_AVAILABLE:

    @triton.jit
    def _attnres_read_kernel(
        values,
        query,
        output,
        n_tokens,
        n_sources,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        source_offsets = tl.arange(0, S_BLOCK)
        d_mask = d_offsets < D

        q = tl.load(query + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        logits = tl.full((S_BLOCK,), -float("inf"), tl.float32)

        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            v = tl.load(
                values + (s * n_tokens + token) * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            k = v
            if KEY_NORM:
                sumsq = tl.sum(k * k, axis=0)
                k = k * tl.rsqrt(sumsq / D + NORM_EPS)
            logit = tl.sum(k * q, axis=0)
            logits = tl.where((source_offsets == s) & valid_source, logit, logits)

        max_logit = tl.max(logits, axis=0)
        weights = tl.exp(logits - max_logit)
        weights = tl.where(source_offsets < n_sources, weights, 0.0)
        weights = weights / tl.sum(weights, axis=0)

        acc = tl.zeros((BLOCK_D,), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            v = tl.load(
                values + (s * n_tokens + token) * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            w = tl.sum(tl.where(source_offsets == s, weights, 0.0), axis=0)
            acc += w * v

        if OUTPUT_NORM:
            sumsq = tl.sum(acc * acc, axis=0)
            acc = acc * tl.rsqrt(sumsq / D + NORM_EPS)

        tl.store(output + token * D + d_offsets, acc, mask=d_mask)



    @triton.jit
    def _select_ptr16(idx: tl.constexpr, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15):
        if idx == 0:
            return p0
        if idx == 1:
            return p1
        if idx == 2:
            return p2
        if idx == 3:
            return p3
        if idx == 4:
            return p4
        if idx == 5:
            return p5
        if idx == 6:
            return p6
        if idx == 7:
            return p7
        if idx == 8:
            return p8
        if idx == 9:
            return p9
        if idx == 10:
            return p10
        if idx == 11:
            return p11
        if idx == 12:
            return p12
        if idx == 13:
            return p13
        if idx == 14:
            return p14
        return p15


    @triton.jit
    def _select_i64_16(idx: tl.constexpr, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15):
        if idx == 0:
            return p0
        if idx == 1:
            return p1
        if idx == 2:
            return p2
        if idx == 3:
            return p3
        if idx == 4:
            return p4
        if idx == 5:
            return p5
        if idx == 6:
            return p6
        if idx == 7:
            return p7
        if idx == 8:
            return p8
        if idx == 9:
            return p9
        if idx == 10:
            return p10
        if idx == 11:
            return p11
        if idx == 12:
            return p12
        if idx == 13:
            return p13
        if idx == 14:
            return p14
        return p15


    @triton.jit
    def _attnres_list_read_kernel(
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        query,
        output,
        output_inv_rms,
        n_tokens,
        n_sources,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        source_offsets = tl.arange(0, S_BLOCK)
        d_mask = d_offsets < D

        q = tl.load(query + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        logits = tl.full((S_BLOCK,), -float("inf"), tl.float32)

        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            v_base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            v = tl.load(
                v_base + token * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            k = v
            if KEY_NORM:
                sumsq = tl.sum(k * k, axis=0)
                k = k * tl.rsqrt(sumsq / D + NORM_EPS)
            logit = tl.sum(k * q, axis=0)
            logits = tl.where((source_offsets == s) & valid_source, logit, logits)

        max_logit = tl.max(logits, axis=0)
        weights = tl.exp(logits - max_logit)
        weights = tl.where(source_offsets < n_sources, weights, 0.0)
        weights = weights / tl.sum(weights, axis=0)

        acc = tl.zeros((BLOCK_D,), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            v_base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            v = tl.load(
                v_base + token * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            w = tl.sum(tl.where(source_offsets == s, weights, 0.0), axis=0)
            acc += w * v

        if OUTPUT_NORM:
            sumsq = tl.sum(acc * acc, axis=0)
            inv_rms = tl.rsqrt(sumsq / D + NORM_EPS)
            acc = acc * inv_rms
            tl.store(output_inv_rms + token, inv_rms)

        tl.store(output + token * D + d_offsets, acc, mask=d_mask)


    @triton.jit
    def _lrid_attnres_list_read_kernel(
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15,
        vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7, vs8, vs9, vs10, vs11, vs12, vs13, vs14, vs15,
        ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9, ks10, ks11, ks12, ks13, ks14, ks15,
        query,
        output,
        output_inv_rms,
        alpha_out,
        key_inv_rms_out,
        n_tokens,
        n_sources,
        logit_scale,
        H: tl.constexpr,
        VALUE_HEAD_DIM: tl.constexpr,
        KEY_HEAD_DIM: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        KEY_NORM: tl.constexpr,
        QUERY_DYNAMIC: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        SAVE_AUX: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token_block = tl.program_id(0)
        head = tl.program_id(1)
        value_block = tl.program_id(2)

        token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
        value_offsets = value_block * BLOCK_D + tl.arange(0, BLOCK_D)
        key_offsets = tl.arange(0, BLOCK_K)
        source_offsets = tl.arange(0, S_BLOCK)
        token_mask = token_offsets < n_tokens
        value_mask = value_offsets < VALUE_HEAD_DIM
        key_mask = key_offsets < KEY_HEAD_DIM
        value_dim = H * VALUE_HEAD_DIM
        key_dim = H * KEY_HEAD_DIM

        logits = tl.full((BLOCK_M, S_BLOCK), -float("inf"), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            k_base = _select_ptr16(s, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15)
            k_stride = _select_i64_16(s, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9, ks10, ks11, ks12, ks13, ks14, ks15)
            k = tl.load(
                k_base + token_offsets[:, None] * k_stride + head * KEY_HEAD_DIM + key_offsets[None, :],
                mask=token_mask[:, None] & key_mask[None, :] & valid_source,
                other=0.0,
            ).to(tl.float32)
            key_inv_rms = tl.full((BLOCK_M,), 1.0, tl.float32)
            if KEY_NORM:
                sumsq = tl.sum(k * k, axis=1)
                key_inv_rms = tl.rsqrt(sumsq / KEY_HEAD_DIM + NORM_EPS)
                k = k * key_inv_rms[:, None]
            if SAVE_AUX:
                tl.store(
                    key_inv_rms_out + (token_offsets * H + head) * n_sources + s,
                    key_inv_rms,
                    mask=token_mask & valid_source,
                )

            if QUERY_DYNAMIC:
                q = tl.load(
                    query + (token_offsets[:, None] * H + head) * KEY_HEAD_DIM + key_offsets[None, :],
                    mask=token_mask[:, None] & key_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                logit = tl.sum(k * q, axis=1) * logit_scale
            else:
                q = tl.load(
                    query + head * KEY_HEAD_DIM + key_offsets,
                    mask=key_mask,
                    other=0.0,
                ).to(tl.float32)
                logit = tl.sum(k * q[None, :], axis=1) * logit_scale

            logits = tl.where((source_offsets[None, :] == s) & valid_source, logit[:, None], logits)

        max_logits = tl.max(logits, axis=1)
        weights = tl.exp(logits - max_logits[:, None])
        weights = tl.where(source_offsets[None, :] < n_sources, weights, 0.0)
        weights = weights / tl.sum(weights, axis=1)[:, None]

        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            w = tl.sum(tl.where(source_offsets[None, :] == s, weights, 0.0), axis=1)
            if SAVE_AUX:
                tl.store(
                    alpha_out + (token_offsets * H + head) * n_sources + s,
                    w,
                    mask=token_mask & valid_source,
                )
            v_base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            v_stride = _select_i64_16(s, vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7, vs8, vs9, vs10, vs11, vs12, vs13, vs14, vs15)
            v = tl.load(
                v_base + token_offsets[:, None] * v_stride + head * VALUE_HEAD_DIM + value_offsets[None, :],
                mask=token_mask[:, None] & value_mask[None, :] & valid_source,
                other=0.0,
            ).to(tl.float32)
            acc += w[:, None] * v

        if OUTPUT_NORM:
            sumsq = tl.sum(acc * acc, axis=1)
            inv_rms = tl.rsqrt(sumsq / VALUE_HEAD_DIM + NORM_EPS)
            acc = acc * inv_rms[:, None]
            tl.store(output_inv_rms + token_offsets * H + head, inv_rms, mask=token_mask)

        tl.store(
            output + (token_offsets[:, None] * H + head) * VALUE_HEAD_DIM + value_offsets[None, :],
            acc,
            mask=token_mask[:, None] & value_mask[None, :],
        )


    @triton.jit
    def _attnres_list_backward_kernel(
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        query,
        grad_output,
        normed_output,
        output_inv_rms,
        gv0, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15,
        grad_query_out,
        n_tokens,
        n_sources,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        source_offsets = tl.arange(0, S_BLOCK)
        d_mask = d_offsets < D

        q = tl.load(query + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        if OUTPUT_NORM:
            y = tl.load(normed_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            inv_rms = tl.load(output_inv_rms + token).to(tl.float32)
            norm_dot = tl.sum(grad * y, axis=0) / D
            grad = inv_rms * (grad - y * norm_dot)
        logits = tl.full((S_BLOCK,), -float("inf"), tl.float32)
        dweights = tl.zeros((S_BLOCK,), tl.float32)

        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            v_base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            v = tl.load(
                v_base + token * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            k = v
            if KEY_NORM:
                sumsq = tl.sum(k * k, axis=0)
                k = k * tl.rsqrt(sumsq / D + NORM_EPS)
            logit = tl.sum(k * q, axis=0)
            dweight = tl.sum(grad * v, axis=0)
            logits = tl.where((source_offsets == s) & valid_source, logit, logits)
            dweights = tl.where((source_offsets == s) & valid_source, dweight, dweights)

        max_logit = tl.max(logits, axis=0)
        weights = tl.exp(logits - max_logit)
        weights = tl.where(source_offsets < n_sources, weights, 0.0)
        weights = weights / tl.sum(weights, axis=0)
        weight_dot = tl.sum(weights * dweights, axis=0)

        dquery = tl.zeros((BLOCK_D,), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            v_base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            gv_base = _select_ptr16(s, gv0, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15)
            v = tl.load(
                v_base + token * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            k = v
            inv_rms = 1.0
            if KEY_NORM:
                sumsq = tl.sum(k * k, axis=0)
                inv_rms = tl.rsqrt(sumsq / D + NORM_EPS)
                k = k * inv_rms
            weight = tl.sum(tl.where(source_offsets == s, weights, 0.0), axis=0)
            dweight = tl.sum(tl.where(source_offsets == s, dweights, 0.0), axis=0)
            dlogit = weight * (dweight - weight_dot)
            grad_value = weight * grad
            grad_key = dlogit * q
            if KEY_NORM:
                grad_key = inv_rms * (grad_key - k * tl.sum(grad_key * k, axis=0) / D)
            grad_value += grad_key
            dquery += dlogit * k
            tl.store(
                gv_base + token * D + d_offsets,
                grad_value,
                mask=d_mask & valid_source,
            )

        tl.atomic_add(grad_query_out + d_offsets, dquery, sem="relaxed", mask=d_mask)


    @triton.jit
    def _attnres_stacked_backward_kernel(
        values,
        query,
        grad_output,
        grad_values,
        grad_query_out,
        n_tokens,
        n_sources,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        source_offsets = tl.arange(0, S_BLOCK)
        d_mask = d_offsets < D
        source_mask = source_offsets < n_sources
        source_mask_2d = source_mask[:, None] & d_mask[None, :]

        q = tl.load(query + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        source_values = tl.load(
            values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            mask=source_mask_2d,
            other=0.0,
        ).to(tl.float32)

        inv_rms = tl.full((S_BLOCK,), 1.0, tl.float32)
        keys = source_values
        if KEY_NORM:
            inv_rms = tl.rsqrt(tl.sum(source_values * source_values, axis=1) / D + NORM_EPS)
            keys = source_values * inv_rms[:, None]

        logits = tl.sum(keys * q[None, :], axis=1)
        logits = tl.where(source_mask, logits, -float("inf"))
        max_logit = tl.max(logits, axis=0)
        weights = tl.exp(logits - max_logit)
        weights = tl.where(source_mask, weights, 0.0)
        weights = weights / tl.sum(weights, axis=0)

        dweights = tl.sum(source_values * grad[None, :], axis=1)
        weight_dot = tl.sum(weights * dweights, axis=0)
        dlogits = weights * (dweights - weight_dot)

        grad_source = weights[:, None] * grad[None, :]
        grad_key = dlogits[:, None] * q[None, :]
        if KEY_NORM:
            grad_key = inv_rms[:, None] * (
                grad_key - keys * tl.sum(grad_key * keys, axis=1)[:, None] / D
            )
        grad_source += grad_key

        tl.store(
            grad_values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            grad_source,
            mask=source_mask_2d,
        )
        dquery = tl.sum(dlogits[:, None] * keys, axis=0)
        tl.atomic_add(grad_query_out + d_offsets, dquery, sem="relaxed", mask=d_mask)


    @triton.jit
    def _lrid_attnres_list_backward_kernel(
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15,
        vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7, vs8, vs9, vs10, vs11, vs12, vs13, vs14, vs15,
        ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9, ks10, ks11, ks12, ks13, ks14, ks15,
        query,
        grad_output,
        normed_output,
        output_inv_rms,
        alpha_saved,
        key_inv_rms_saved,
        gv0, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15,
        gk0, gk1, gk2, gk3, gk4, gk5, gk6, gk7, gk8, gk9, gk10, gk11, gk12, gk13, gk14, gk15,
        grad_query,
        n_tokens,
        n_sources,
        logit_scale,
        H: tl.constexpr,
        VALUE_HEAD_DIM: tl.constexpr,
        KEY_HEAD_DIM: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        KEY_NORM: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        USE_AUX: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        value_offsets = tl.arange(0, BLOCK_D)
        key_offsets = tl.arange(0, BLOCK_K)
        source_offsets = tl.arange(0, S_BLOCK)
        value_mask = value_offsets < VALUE_HEAD_DIM
        key_mask = key_offsets < KEY_HEAD_DIM
        value_dim = H * VALUE_HEAD_DIM
        key_dim = H * KEY_HEAD_DIM

        q = tl.load(
            query + head * KEY_HEAD_DIM + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        grad = tl.load(
            grad_output + (token * H + head) * VALUE_HEAD_DIM + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)
        if OUTPUT_NORM:
            y = tl.load(
                normed_output + (token * H + head) * VALUE_HEAD_DIM + value_offsets,
                mask=value_mask,
                other=0.0,
            ).to(tl.float32)
            inv_rms = tl.load(output_inv_rms + token * H + head).to(tl.float32)
            norm_dot = tl.sum(grad * y, axis=0) / VALUE_HEAD_DIM
            grad = inv_rms * (grad - y * norm_dot)

        logits = tl.full((S_BLOCK,), -float("inf"), tl.float32)
        dweights = tl.zeros((S_BLOCK,), tl.float32)
        weights = tl.zeros((S_BLOCK,), tl.float32)

        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            k_base = _select_ptr16(s, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15)
            v_base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            k_stride = _select_i64_16(s, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9, ks10, ks11, ks12, ks13, ks14, ks15)
            v_stride = _select_i64_16(s, vs0, vs1, vs2, vs3, vs4, vs5, vs6, vs7, vs8, vs9, vs10, vs11, vs12, vs13, vs14, vs15)
            v = tl.load(
                v_base + token * v_stride + head * VALUE_HEAD_DIM + value_offsets,
                mask=value_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            dweight = tl.sum(grad * v, axis=0)
            dweights = tl.where((source_offsets == s) & valid_source, dweight, dweights)
            if USE_AUX:
                weight = tl.load(alpha_saved + (token * H + head) * n_sources + s, mask=valid_source, other=0.0).to(tl.float32)
                weights = tl.where((source_offsets == s) & valid_source, weight, weights)
            else:
                k = tl.load(
                    k_base + token * k_stride + head * KEY_HEAD_DIM + key_offsets,
                    mask=key_mask & valid_source,
                    other=0.0,
                ).to(tl.float32)
                if KEY_NORM:
                    sumsq = tl.sum(k * k, axis=0)
                    k = k * tl.rsqrt(sumsq / KEY_HEAD_DIM + NORM_EPS)
                logit = tl.sum(k * q, axis=0) * logit_scale
                logits = tl.where((source_offsets == s) & valid_source, logit, logits)

        if not USE_AUX:
            max_logit = tl.max(logits, axis=0)
            weights = tl.exp(logits - max_logit)
            weights = tl.where(source_offsets < n_sources, weights, 0.0)
            weights = weights / tl.sum(weights, axis=0)
        weight_dot = tl.sum(weights * dweights, axis=0)

        dquery = tl.zeros((BLOCK_K,), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            k_base = _select_ptr16(s, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15)
            v_base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            k_stride = _select_i64_16(s, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9, ks10, ks11, ks12, ks13, ks14, ks15)
            gk_base = _select_ptr16(s, gk0, gk1, gk2, gk3, gk4, gk5, gk6, gk7, gk8, gk9, gk10, gk11, gk12, gk13, gk14, gk15)
            gv_base = _select_ptr16(s, gv0, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15)
            k = tl.load(
                k_base + token * k_stride + head * KEY_HEAD_DIM + key_offsets,
                mask=key_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            inv_rms = 1.0
            if KEY_NORM:
                if USE_AUX:
                    inv_rms = tl.load(key_inv_rms_saved + (token * H + head) * n_sources + s, mask=valid_source, other=1.0).to(tl.float32)
                else:
                    sumsq = tl.sum(k * k, axis=0)
                    inv_rms = tl.rsqrt(sumsq / KEY_HEAD_DIM + NORM_EPS)
                k = k * inv_rms
            weight = tl.sum(tl.where(source_offsets == s, weights, 0.0), axis=0)
            dweight = tl.sum(tl.where(source_offsets == s, dweights, 0.0), axis=0)
            dlogit = weight * (dweight - weight_dot)

            tl.store(
                gv_base + token * value_dim + head * VALUE_HEAD_DIM + value_offsets,
                weight * grad,
                mask=value_mask & valid_source,
            )

            grad_key = dlogit * logit_scale * q
            if KEY_NORM:
                grad_key = inv_rms * (grad_key - k * tl.sum(grad_key * k, axis=0) / KEY_HEAD_DIM)
            tl.store(
                gk_base + token * key_dim + head * KEY_HEAD_DIM + key_offsets,
                grad_key,
                mask=key_mask & valid_source,
            )
            dquery += dlogit * logit_scale * k

        tl.atomic_add(
            grad_query + head * KEY_HEAD_DIM + key_offsets,
            dquery,
            sem="relaxed",
            mask=key_mask,
        )


    @triton.jit
    def _attnres_list_average_kernel(
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        output,
        n_tokens,
        n_sources,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        acc = tl.zeros((BLOCK_D,), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            v_base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            v = tl.load(
                v_base + token * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            acc += v
        acc = acc / n_sources

        if OUTPUT_NORM:
            sumsq = tl.sum(acc * acc, axis=0)
            acc = acc * tl.rsqrt(sumsq / D + NORM_EPS)

        tl.store(output + token * D + d_offsets, acc, mask=d_mask)


    @triton.jit
    def _attnres_phase1_forward_kernel(
        values,
        queries,
        output,
        lse,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        source_values = tl.load(
            values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            mask=(source_offsets[:, None] < n_sources) & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        inv_rms = tl.full((S_BLOCK,), 1.0, tl.float32)
        if KEY_NORM:
            sumsq = tl.sum(source_values * source_values, axis=1)
            inv_rms = tl.rsqrt(sumsq / D + NORM_EPS)

        for q_idx in tl.static_range(0, N_QUERIES):
            query = tl.load(
                queries + q_idx * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)

            logits = tl.sum(source_values * query[None, :], axis=1)
            if KEY_NORM:
                logits *= inv_rms
            logits = tl.where(source_offsets < n_sources, logits, -float("inf"))

            max_logit = tl.max(logits, axis=0)
            weights = tl.exp(logits - max_logit)
            weights = tl.where(source_offsets < n_sources, weights, 0.0)
            weight_sum = tl.sum(weights, axis=0)
            acc = tl.sum(weights[:, None] * source_values, axis=0) / weight_sum

            tl.store(
                output + (q_idx * n_tokens + token) * D + d_offsets,
                acc,
                mask=d_mask,
            )
            tl.store(lse + q_idx * n_tokens + token, max_logit + tl.log(weight_sum))


    @triton.jit
    def _attnres_phase1_backward_kernel(
        values,
        queries,
        lse,
        grad_output,
        grad_lse,
        grad_values,
        grad_queries_partial,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        source_mask_2d = (source_offsets[:, None] < n_sources) & d_mask[None, :]
        source_mask_1d = source_offsets < n_sources

        source_values = tl.load(
            values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            mask=source_mask_2d,
            other=0.0,
        ).to(tl.float32)

        inv_rms = tl.full((S_BLOCK,), 1.0, tl.float32)
        if KEY_NORM:
            sumsq = tl.sum(source_values * source_values, axis=1)
            inv_rms = tl.rsqrt(sumsq / D + NORM_EPS)

        grad_source_acc = tl.zeros((S_BLOCK, BLOCK_D), tl.float32)

        for q_idx in tl.static_range(0, N_QUERIES):
            query = tl.load(
                queries + q_idx * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            grad = tl.load(
                grad_output + (q_idx * n_tokens + token) * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)

            logits = tl.sum(source_values * query[None, :], axis=1)
            if KEY_NORM:
                logits *= inv_rms
            logits = tl.where(source_mask_1d, logits, -float("inf"))
            logsumexp = tl.load(lse + q_idx * n_tokens + token).to(tl.float32)
            weights = tl.exp(logits - logsumexp)
            weights = tl.where(source_mask_1d, weights, 0.0)

            dweights = tl.sum(source_values * grad[None, :], axis=1)
            expected_dweight = tl.sum(weights * dweights, axis=0)
            grad_logsumexp = 0.0
            if HAS_GRAD_LSE:
                grad_logsumexp = tl.load(grad_lse + q_idx * n_tokens + token).to(tl.float32)
            dlogits = weights * (grad_logsumexp + dweights - expected_dweight)

            grad_source = weights[:, None] * grad[None, :]
            grad_key = dlogits[:, None] * query[None, :]
            if KEY_NORM:
                key = source_values * inv_rms[:, None]
                grad_key = inv_rms[:, None] * (
                    grad_key - key * tl.sum(grad_key * key, axis=1)[:, None] / D
                )
                grad_query = tl.sum(dlogits[:, None] * key, axis=0)
            else:
                grad_query = tl.sum(dlogits[:, None] * source_values, axis=0)
            grad_source += grad_key
            grad_source_acc += tl.where(source_mask_2d, grad_source, 0.0)

            tl.store(
                grad_queries_partial + (q_idx * n_tokens + token) * D + d_offsets,
                grad_query,
                mask=d_mask,
            )

        tl.store(
            grad_values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            grad_source_acc,
            mask=source_mask_2d,
        )


    @triton.jit
    def _attnres_phase1_logits_forward_kernel(
        values,
        logits,
        output,
        lse,
        output_inv_rms,
        saved_weights,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < n_sources
        d_mask = d_offsets < D

        source_values = tl.load(
            values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            mask=source_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        for q_idx in tl.static_range(0, N_QUERIES):
            q_logits = tl.load(
                logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
                mask=source_mask,
                other=-float("inf"),
            ).to(tl.float32)
            q_logits = tl.where(source_mask, q_logits, -float("inf"))
            max_logit = tl.max(q_logits, axis=0)
            weights = tl.exp(q_logits - max_logit)
            weights = tl.where(source_mask, weights, 0.0)
            weight_sum = tl.sum(weights, axis=0)
            probs = weights / weight_sum
            acc = tl.sum(probs[:, None] * source_values, axis=0)
            tl.store(
                saved_weights + (q_idx * n_sources + source_offsets) * n_tokens + token,
                probs,
                mask=source_mask,
            )
            if OUTPUT_NORM:
                inv_rms = tl.rsqrt(tl.sum(acc * acc, axis=0) / D + NORM_EPS)
                acc = acc * inv_rms
                tl.store(output_inv_rms + q_idx * n_tokens + token, inv_rms)
            tl.store(
                output + (q_idx * n_tokens + token) * D + d_offsets,
                acc,
                mask=d_mask,
            )
            tl.store(lse + q_idx * n_tokens + token, max_logit + tl.log(weight_sum))


    @triton.jit
    def _attnres_phase1_logits_backward_kernel(
        values,
        logits,
        lse,
        output,
        output_inv_rms,
        grad_output,
        grad_lse,
        grad_values,
        grad_logits,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
    ):
        token = tl.program_id(0)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < n_sources
        source_mask_2d = source_mask[:, None]
        d_mask = d_offsets < D

        source_values = tl.load(
            values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            mask=source_mask_2d & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_source_acc = tl.zeros((S_BLOCK, BLOCK_D), tl.float32)

        for q_idx in tl.static_range(0, N_QUERIES):
            grad = tl.load(
                grad_output + (q_idx * n_tokens + token) * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            if OUTPUT_NORM:
                y = tl.load(
                    output + (q_idx * n_tokens + token) * D + d_offsets,
                    mask=d_mask,
                    other=0.0,
                ).to(tl.float32)
                inv_rms = tl.load(output_inv_rms + q_idx * n_tokens + token).to(tl.float32)
                grad = inv_rms * (grad - y * tl.sum(grad * y, axis=0) / D)
            q_logits = tl.load(
                logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
                mask=source_mask,
                other=-float("inf"),
            ).to(tl.float32)
            q_logits = tl.where(source_mask, q_logits, -float("inf"))
            logsumexp = tl.load(lse + q_idx * n_tokens + token).to(tl.float32)
            weights = tl.exp(q_logits - logsumexp)
            weights = tl.where(source_mask, weights, 0.0)
            dweights = tl.sum(source_values * grad[None, :], axis=1)
            expected_dweight = tl.sum(weights * dweights, axis=0)
            grad_logsumexp = 0.0
            if HAS_GRAD_LSE:
                grad_logsumexp = tl.load(grad_lse + q_idx * n_tokens + token).to(tl.float32)
            dlogits = weights * (grad_logsumexp + dweights - expected_dweight)
            grad_source_acc += tl.where(source_mask_2d, weights[:, None] * grad[None, :], 0.0)
            tl.store(
                grad_logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
                dlogits,
                mask=source_mask,
            )

        tl.store(
            grad_values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            grad_source_acc,
            mask=source_mask_2d & d_mask[None, :],
        )


    @triton.jit
    def _attnres_phase1_logits_backward_weights_kernel(
        values,
        saved_weights,
        output,
        output_inv_rms,
        grad_output,
        grad_lse,
        grad_values,
        grad_logits,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
    ):
        token = tl.program_id(0)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < n_sources
        source_mask_2d = source_mask[:, None]
        d_mask = d_offsets < D

        source_values = tl.load(
            values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            mask=source_mask_2d & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        grad_source_acc = tl.zeros((S_BLOCK, BLOCK_D), tl.float32)

        for q_idx in tl.static_range(0, N_QUERIES):
            grad = tl.load(
                grad_output + (q_idx * n_tokens + token) * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            if OUTPUT_NORM:
                y = tl.load(
                    output + (q_idx * n_tokens + token) * D + d_offsets,
                    mask=d_mask,
                    other=0.0,
                ).to(tl.float32)
                inv_rms = tl.load(output_inv_rms + q_idx * n_tokens + token).to(tl.float32)
                grad = inv_rms * (grad - y * tl.sum(grad * y, axis=0) / D)

            weights = tl.load(
                saved_weights + (q_idx * n_sources + source_offsets) * n_tokens + token,
                mask=source_mask,
                other=0.0,
            ).to(tl.float32)
            weights = tl.where(source_mask, weights, 0.0)
            dweights = tl.sum(source_values * grad[None, :], axis=1)
            expected_dweight = tl.sum(weights * dweights, axis=0)
            grad_logsumexp = 0.0
            if HAS_GRAD_LSE:
                grad_logsumexp = tl.load(grad_lse + q_idx * n_tokens + token).to(tl.float32)
            dlogits = weights * (grad_logsumexp + dweights - expected_dweight)
            grad_source_acc += tl.where(source_mask_2d, weights[:, None] * grad[None, :], 0.0)
            tl.store(
                grad_logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
                dlogits,
                mask=source_mask,
            )

        tl.store(
            grad_values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            grad_source_acc,
            mask=source_mask_2d & d_mask[None, :],
        )


    @triton.jit
    def _attnres_phase1_logits_list_forward_kernel(
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        logits,
        output,
        lse,
        output_inv_rms,
        saved_weights,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < n_sources
        d_mask = d_offsets < D

        source_values = tl.zeros((S_BLOCK, BLOCK_D), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            value = tl.load(
                base + token * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            source_values += tl.where((source_offsets[:, None] == s) & valid_source, value[None, :], 0.0)

        for q_idx in tl.static_range(0, N_QUERIES):
            q_logits = tl.load(
                logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
                mask=source_mask,
                other=-float("inf"),
            ).to(tl.float32)
            q_logits = tl.where(source_mask, q_logits, -float("inf"))
            max_logit = tl.max(q_logits, axis=0)
            weights = tl.exp(q_logits - max_logit)
            weights = tl.where(source_mask, weights, 0.0)
            weight_sum = tl.sum(weights, axis=0)
            probs = weights / weight_sum
            acc = tl.sum(probs[:, None] * source_values, axis=0)
            tl.store(
                saved_weights + (q_idx * n_sources + source_offsets) * n_tokens + token,
                probs,
                mask=source_mask,
            )
            if OUTPUT_NORM:
                inv_rms = tl.rsqrt(tl.sum(acc * acc, axis=0) / D + NORM_EPS)
                acc = acc * inv_rms
                tl.store(output_inv_rms + q_idx * n_tokens + token, inv_rms)
            tl.store(
                output + (q_idx * n_tokens + token) * D + d_offsets,
                acc,
                mask=d_mask,
            )
            tl.store(lse + q_idx * n_tokens + token, max_logit + tl.log(weight_sum))


    @triton.jit
    def _attnres_phase1_logits_list_backward_weights_kernel(
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        saved_weights,
        output,
        output_inv_rms,
        grad_output,
        grad_lse,
        gv0, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15,
        grad_logits,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
    ):
        token = tl.program_id(0)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < n_sources
        source_mask_2d = source_mask[:, None]
        d_mask = d_offsets < D

        source_values = tl.zeros((S_BLOCK, BLOCK_D), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            value = tl.load(
                base + token * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            source_values += tl.where((source_offsets[:, None] == s) & valid_source, value[None, :], 0.0)

        grad_source_acc = tl.zeros((S_BLOCK, BLOCK_D), tl.float32)
        for q_idx in tl.static_range(0, N_QUERIES):
            grad = tl.load(
                grad_output + (q_idx * n_tokens + token) * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            if OUTPUT_NORM:
                y = tl.load(
                    output + (q_idx * n_tokens + token) * D + d_offsets,
                    mask=d_mask,
                    other=0.0,
                ).to(tl.float32)
                inv_rms = tl.load(output_inv_rms + q_idx * n_tokens + token).to(tl.float32)
                grad = inv_rms * (grad - y * tl.sum(grad * y, axis=0) / D)

            weights = tl.load(
                saved_weights + (q_idx * n_sources + source_offsets) * n_tokens + token,
                mask=source_mask,
                other=0.0,
            ).to(tl.float32)
            weights = tl.where(source_mask, weights, 0.0)
            dweights = tl.sum(source_values * grad[None, :], axis=1)
            expected_dweight = tl.sum(weights * dweights, axis=0)
            grad_logsumexp = 0.0
            if HAS_GRAD_LSE:
                grad_logsumexp = tl.load(grad_lse + q_idx * n_tokens + token).to(tl.float32)
            dlogits = weights * (grad_logsumexp + dweights - expected_dweight)
            grad_source_acc += tl.where(source_mask_2d, weights[:, None] * grad[None, :], 0.0)
            tl.store(
                grad_logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
                dlogits,
                mask=source_mask,
            )

        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            grad_base = _select_ptr16(s, gv0, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15)
            grad_value = tl.sum(tl.where(source_offsets[:, None] == s, grad_source_acc, 0.0), axis=0)
            tl.store(
                grad_base + token * D + d_offsets,
                grad_value,
                mask=d_mask & valid_source,
            )


    @triton.jit
    def _attnres_phase1_logits_forward_qsplit_kernel(
        values,
        logits,
        output,
        lse,
        n_tokens,
        n_sources,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        q_idx = tl.program_id(1)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < n_sources
        d_mask = d_offsets < D

        q_logits = tl.load(
            logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
            mask=source_mask,
            other=-float("inf"),
        ).to(tl.float32)
        q_logits = tl.where(source_mask, q_logits, -float("inf"))
        max_logit = tl.max(q_logits, axis=0)
        weights = tl.exp(q_logits - max_logit)
        weights = tl.where(source_mask, weights, 0.0)
        weight_sum = tl.sum(weights, axis=0)
        probs = weights / weight_sum

        acc = tl.zeros((BLOCK_D,), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            v = tl.load(
                values + (s * n_tokens + token) * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            p = tl.sum(tl.where(source_offsets == s, probs, 0.0), axis=0)
            acc += p * v

        tl.store(
            output + (q_idx * n_tokens + token) * D + d_offsets,
            acc,
            mask=d_mask,
        )
        tl.store(lse + q_idx * n_tokens + token, max_logit + tl.log(weight_sum))


    @triton.jit
    def _attnres_phase1_logits_backward_qsplit_kernel(
        values,
        logits,
        lse,
        grad_output,
        grad_lse,
        grad_values,
        grad_logits,
        n_tokens,
        n_sources,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
    ):
        token = tl.program_id(0)
        q_idx = tl.program_id(1)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        source_mask = source_offsets < n_sources
        d_mask = d_offsets < D

        grad = tl.load(
            grad_output + (q_idx * n_tokens + token) * D + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        q_logits = tl.load(
            logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
            mask=source_mask,
            other=-float("inf"),
        ).to(tl.float32)
        logsumexp = tl.load(lse + q_idx * n_tokens + token).to(tl.float32)
        weights = tl.exp(q_logits - logsumexp)
        weights = tl.where(source_mask, weights, 0.0)

        dweights = tl.zeros((S_BLOCK,), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            v = tl.load(
                values + (s * n_tokens + token) * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            dweight = tl.sum(grad * v, axis=0)
            dweights = tl.where((source_offsets == s) & valid_source, dweight, dweights)

        expected = tl.sum(weights * dweights, axis=0)
        grad_logsumexp = 0.0
        if HAS_GRAD_LSE:
            grad_logsumexp = tl.load(grad_lse + q_idx * n_tokens + token).to(tl.float32)
        dlogits = weights * (grad_logsumexp + dweights - expected)
        tl.store(
            grad_logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
            dlogits,
            mask=source_mask,
        )

        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            weight = tl.sum(tl.where(source_offsets == s, weights, 0.0), axis=0)
            tl.atomic_add(
                grad_values + (s * n_tokens + token) * D + d_offsets,
                weight * grad,
                sem="relaxed",
                mask=d_mask & valid_source,
            )


    @triton.jit
    def _attnres_phase1_logits_forward_tiled_kernel(
        values,
        logits,
        output,
        lse,
        n_tokens,
        n_sources,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D_TILE: tl.constexpr,
    ):
        token = tl.program_id(0)
        q_idx = tl.program_id(1)
        d_block = tl.program_id(2)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = d_block * BLOCK_D_TILE + tl.arange(0, BLOCK_D_TILE)
        source_mask = source_offsets < n_sources
        d_mask = d_offsets < D

        q_logits = tl.load(
            logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
            mask=source_mask,
            other=-float("inf"),
        ).to(tl.float32)
        q_logits = tl.where(source_mask, q_logits, -float("inf"))
        max_logit = tl.max(q_logits, axis=0)
        weights = tl.exp(q_logits - max_logit)
        weights = tl.where(source_mask, weights, 0.0)
        weight_sum = tl.sum(weights, axis=0)
        probs = weights / weight_sum

        source_values = tl.load(
            values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            mask=source_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(probs[:, None] * source_values, axis=0)
        tl.store(
            output + (q_idx * n_tokens + token) * D + d_offsets,
            acc,
            mask=d_mask,
        )
        if d_block == 0:
            tl.store(lse + q_idx * n_tokens + token, max_logit + tl.log(weight_sum))


    @triton.jit
    def _attnres_phase1_logits_dweights_kernel(
        values,
        grad_output,
        partial_dweights,
        n_tokens,
        n_sources,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D_TILE: tl.constexpr,
        N_DBLOCKS: tl.constexpr,
    ):
        token = tl.program_id(0)
        q_idx = tl.program_id(1)
        d_block = tl.program_id(2)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = d_block * BLOCK_D_TILE + tl.arange(0, BLOCK_D_TILE)
        source_mask = source_offsets < n_sources
        d_mask = d_offsets < D

        grad = tl.load(
            grad_output + (q_idx * n_tokens + token) * D + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        source_values = tl.load(
            values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            mask=source_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        partial = tl.sum(source_values * grad[None, :], axis=1)
        tl.store(
            partial_dweights + ((q_idx * n_sources + source_offsets) * n_tokens + token) * N_DBLOCKS + d_block,
            partial,
            mask=source_mask,
        )


    @triton.jit
    def _attnres_phase1_logits_dlogits_kernel(
        logits,
        lse,
        grad_lse,
        partial_dweights,
        grad_logits,
        n_tokens,
        n_sources,
        S_BLOCK: tl.constexpr,
        N_DBLOCKS: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
    ):
        token = tl.program_id(0)
        q_idx = tl.program_id(1)
        source_offsets = tl.arange(0, S_BLOCK)
        dblock_offsets = tl.arange(0, N_DBLOCKS)
        source_mask = source_offsets < n_sources

        q_logits = tl.load(
            logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
            mask=source_mask,
            other=-float("inf"),
        ).to(tl.float32)
        logsumexp = tl.load(lse + q_idx * n_tokens + token).to(tl.float32)
        weights = tl.exp(q_logits - logsumexp)
        weights = tl.where(source_mask, weights, 0.0)
        partial = tl.load(
            partial_dweights
            + ((q_idx * n_sources + source_offsets[:, None]) * n_tokens + token) * N_DBLOCKS
            + dblock_offsets[None, :],
            mask=source_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        dweights = tl.sum(partial, axis=1)
        expected = tl.sum(weights * dweights, axis=0)
        grad_logsumexp = 0.0
        if HAS_GRAD_LSE:
            grad_logsumexp = tl.load(grad_lse + q_idx * n_tokens + token).to(tl.float32)
        dlogits = weights * (grad_logsumexp + dweights - expected)
        tl.store(
            grad_logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
            dlogits,
            mask=source_mask,
        )


    @triton.jit
    def _attnres_phase1_logits_grad_values_kernel(
        values,
        logits,
        lse,
        grad_output,
        grad_logits,
        grad_values,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D_TILE: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_block = tl.program_id(1)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = d_block * BLOCK_D_TILE + tl.arange(0, BLOCK_D_TILE)
        source_mask = source_offsets < n_sources
        d_mask = d_offsets < D

        grad_source_acc = tl.zeros((S_BLOCK, BLOCK_D_TILE), tl.float32)
        for q_idx in tl.static_range(0, N_QUERIES):
            q_logits = tl.load(
                logits + (q_idx * n_sources + source_offsets) * n_tokens + token,
                mask=source_mask,
                other=-float("inf"),
            ).to(tl.float32)
            logsumexp = tl.load(lse + q_idx * n_tokens + token).to(tl.float32)
            weights = tl.exp(q_logits - logsumexp)
            weights = tl.where(source_mask, weights, 0.0)
            grad = tl.load(
                grad_output + (q_idx * n_tokens + token) * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            grad_source_acc += tl.where(
                source_mask[:, None],
                weights[:, None] * grad[None, :],
                0.0,
            )

        tl.store(
            grad_values + (source_offsets[:, None] * n_tokens + token) * D + d_offsets[None, :],
            grad_source_acc,
            mask=source_mask[:, None] & d_mask[None, :],
        )


    @triton.jit
    def _attnres_phase1_list_forward_kernel(
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        queries,
        output,
        lse,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        source_values = tl.zeros((S_BLOCK, BLOCK_D), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            v = tl.load(
                base + token * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            source_values += tl.where((source_offsets[:, None] == s) & valid_source, v[None, :], 0.0)

        inv_rms = tl.full((S_BLOCK,), 1.0, tl.float32)
        if KEY_NORM:
            sumsq = tl.sum(source_values * source_values, axis=1)
            inv_rms = tl.rsqrt(sumsq / D + NORM_EPS)

        for q_idx in tl.static_range(0, N_QUERIES):
            query = tl.load(
                queries + q_idx * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(source_values * query[None, :], axis=1)
            if KEY_NORM:
                logits *= inv_rms
            logits = tl.where(source_offsets < n_sources, logits, -float("inf"))
            max_logit = tl.max(logits, axis=0)
            weights = tl.exp(logits - max_logit)
            weights = tl.where(source_offsets < n_sources, weights, 0.0)
            weight_sum = tl.sum(weights, axis=0)
            acc = tl.sum(weights[:, None] * source_values, axis=0) / weight_sum
            tl.store(
                output + (q_idx * n_tokens + token) * D + d_offsets,
                acc,
                mask=d_mask,
            )
            tl.store(lse + q_idx * n_tokens + token, max_logit + tl.log(weight_sum))


    @triton.jit
    def _attnres_phase1_list_backward_kernel(
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
        queries,
        lse,
        grad_output,
        grad_lse,
        gv0, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15,
        grad_queries_partial,
        n_tokens,
        n_sources,
        N_QUERIES: tl.constexpr,
        D: tl.constexpr,
        S_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        HAS_GRAD_LSE: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        source_offsets = tl.arange(0, S_BLOCK)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        source_mask_1d = source_offsets < n_sources
        source_mask_2d = source_mask_1d[:, None] & d_mask[None, :]

        source_values = tl.zeros((S_BLOCK, BLOCK_D), tl.float32)
        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            base = _select_ptr16(s, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)
            v = tl.load(
                base + token * D + d_offsets,
                mask=d_mask & valid_source,
                other=0.0,
            ).to(tl.float32)
            source_values += tl.where((source_offsets[:, None] == s) & valid_source, v[None, :], 0.0)

        inv_rms = tl.full((S_BLOCK,), 1.0, tl.float32)
        if KEY_NORM:
            sumsq = tl.sum(source_values * source_values, axis=1)
            inv_rms = tl.rsqrt(sumsq / D + NORM_EPS)

        grad_source_acc = tl.zeros((S_BLOCK, BLOCK_D), tl.float32)
        for q_idx in tl.static_range(0, N_QUERIES):
            query = tl.load(
                queries + q_idx * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            grad = tl.load(
                grad_output + (q_idx * n_tokens + token) * D + d_offsets,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(source_values * query[None, :], axis=1)
            if KEY_NORM:
                logits *= inv_rms
            logits = tl.where(source_mask_1d, logits, -float("inf"))
            logsumexp = tl.load(lse + q_idx * n_tokens + token).to(tl.float32)
            weights = tl.exp(logits - logsumexp)
            weights = tl.where(source_mask_1d, weights, 0.0)

            dweights = tl.sum(source_values * grad[None, :], axis=1)
            expected_dweight = tl.sum(weights * dweights, axis=0)
            grad_logsumexp = 0.0
            if HAS_GRAD_LSE:
                grad_logsumexp = tl.load(grad_lse + q_idx * n_tokens + token).to(tl.float32)
            dlogits = weights * (grad_logsumexp + dweights - expected_dweight)

            grad_source = weights[:, None] * grad[None, :]
            grad_key = dlogits[:, None] * query[None, :]
            if KEY_NORM:
                key = source_values * inv_rms[:, None]
                grad_key = inv_rms[:, None] * (
                    grad_key - key * tl.sum(grad_key * key, axis=1)[:, None] / D
                )
                grad_query = tl.sum(dlogits[:, None] * key, axis=0)
            else:
                grad_query = tl.sum(dlogits[:, None] * source_values, axis=0)
            grad_source_acc += tl.where(source_mask_2d, grad_source + grad_key, 0.0)
            tl.store(
                grad_queries_partial + (q_idx * n_tokens + token) * D + d_offsets,
                grad_query,
                mask=d_mask,
            )

        for s in tl.static_range(0, S_BLOCK):
            valid_source = s < n_sources
            gv_base = _select_ptr16(s, gv0, gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8, gv9, gv10, gv11, gv12, gv13, gv14, gv15)
            grad_s = tl.sum(tl.where(source_offsets[:, None] == s, grad_source_acc, 0.0), axis=0)
            tl.store(
                gv_base + token * D + d_offsets,
                grad_s,
                mask=d_mask & valid_source,
            )


    @triton.jit
    def _attnres_phase2_forward_kernel(
        partial_value,
        query,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        n_tokens,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        v = tl.load(partial_value + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        q = tl.load(query + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        prev = tl.load(interblock_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        prev_lse = tl.load(interblock_lse + token).to(tl.float32)

        key = v
        if KEY_NORM:
            inv_rms = tl.rsqrt(tl.sum(v * v, axis=0) / D + NORM_EPS)
            key = v * inv_rms
        logit = tl.sum(key * q, axis=0)

        prob = tl.sigmoid(logit - prev_lse)
        merged = prev + prob * (v - prev)
        if OUTPUT_NORM:
            out_inv_rms = tl.rsqrt(tl.sum(merged * merged, axis=0) / D + NORM_EPS)
            merged = merged * out_inv_rms
            tl.store(output_inv_rms + token, out_inv_rms)
        tl.store(output + token * D + d_offsets, merged, mask=d_mask)


    @triton.jit
    def _attnres_phase2_backward_kernel(
        partial_value,
        query,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        grad_output,
        grad_partial_value,
        grad_query_partial,
        grad_interblock_output,
        grad_interblock_lse,
        n_tokens,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        v = tl.load(partial_value + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        q = tl.load(query + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        prev = tl.load(interblock_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        prev_lse = tl.load(interblock_lse + token).to(tl.float32)
        grad = tl.load(grad_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        if OUTPUT_NORM:
            y = tl.load(output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            out_inv_rms = tl.load(output_inv_rms + token).to(tl.float32)
            grad = out_inv_rms * (grad - y * tl.sum(grad * y, axis=0) / D)

        inv_rms = tl.full((), 1.0, tl.float32)
        key = v
        logit = tl.sum(v * q, axis=0)
        if KEY_NORM:
            inv_rms = tl.rsqrt(tl.sum(v * v, axis=0) / D + NORM_EPS)
            key = v * inv_rms
            logit = tl.sum(key * q, axis=0)

        prob = tl.sigmoid(logit - prev_lse)
        prev_prob = 1.0 - prob
        merge_grad = tl.sum(grad * (v - prev), axis=0)
        dlogit = prob * prev_prob * merge_grad
        grad_lse = -dlogit

        grad_v = prob * grad
        grad_key = dlogit * q
        if KEY_NORM:
            grad_key = inv_rms * (grad_key - key * tl.sum(grad_key * key, axis=0) / D)
        grad_v += grad_key
        grad_query = dlogit * key

        tl.store(grad_partial_value + token * D + d_offsets, grad_v, mask=d_mask)
        tl.store(grad_interblock_output + token * D + d_offsets, prev_prob * grad, mask=d_mask)
        tl.store(grad_query_partial + token * D + d_offsets, grad_query, mask=d_mask)
        tl.store(grad_interblock_lse + token, grad_lse)


    @triton.jit
    def _attnres_phase2_logits_forward_kernel(
        partial_value,
        partial_logit,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        n_tokens,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        v = tl.load(partial_value + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        prev = tl.load(interblock_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        logit = tl.load(partial_logit + token).to(tl.float32)
        prev_lse = tl.load(interblock_lse + token).to(tl.float32)
        prob = tl.sigmoid(logit - prev_lse)
        merged = prev + prob * (v - prev)
        if OUTPUT_NORM:
            inv_rms = tl.rsqrt(tl.sum(merged * merged, axis=0) / D + NORM_EPS)
            merged = merged * inv_rms
            tl.store(output_inv_rms + token, inv_rms)
        tl.store(output + token * D + d_offsets, merged, mask=d_mask)


    @triton.jit
    def _attnres_phase2_logits_backward_kernel(
        partial_value,
        partial_logit,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        grad_output,
        grad_partial_value,
        grad_partial_logit,
        grad_interblock_output,
        grad_interblock_lse,
        n_tokens,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        v = tl.load(partial_value + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        prev = tl.load(interblock_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        if OUTPUT_NORM:
            y = tl.load(output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            inv_rms = tl.load(output_inv_rms + token).to(tl.float32)
            grad = inv_rms * (grad - y * tl.sum(grad * y, axis=0) / D)
        logit = tl.load(partial_logit + token).to(tl.float32)
        prev_lse = tl.load(interblock_lse + token).to(tl.float32)

        prob = tl.sigmoid(logit - prev_lse)
        prev_prob = 1.0 - prob
        merge_grad = tl.sum(grad * (v - prev), axis=0)
        dlogit = prob * prev_prob * merge_grad

        tl.store(grad_partial_value + token * D + d_offsets, prob * grad, mask=d_mask)
        tl.store(grad_interblock_output + token * D + d_offsets, prev_prob * grad, mask=d_mask)
        tl.store(grad_partial_logit + token, dlogit)
        tl.store(grad_interblock_lse + token, -dlogit)


    @triton.jit
    def _attnres_phase2_backward_fused_query_kernel(
        partial_value,
        query,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        grad_output,
        grad_partial,
        grad_query_out,
        grad_interblock_output,
        grad_interblock_lse,
        n_tokens,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        KEY_NORM: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offsets = tl.arange(0, BLOCK_D)
        token_mask = token_offsets < n_tokens
        d_mask = d_offsets < D
        mask = token_mask[:, None] & d_mask[None, :]

        partial = tl.load(
            partial_value + token_offsets[:, None] * D + d_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inter = tl.load(
            interblock_output + token_offsets[:, None] * D + d_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad = tl.load(
            grad_output + token_offsets[:, None] * D + d_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if OUTPUT_NORM:
            y = tl.load(
                output + token_offsets[:, None] * D + d_offsets[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            out_inv_rms = tl.load(output_inv_rms + token_offsets, mask=token_mask, other=0.0).to(tl.float32)
            norm_dot = tl.sum(grad * y, axis=1) / D
            grad = out_inv_rms[:, None] * (grad - y * norm_dot[:, None])
        q = tl.load(query + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        inv_rms = tl.full((BLOCK_M,), 1.0, tl.float32)
        key = partial
        if KEY_NORM:
            inv_rms = tl.rsqrt(tl.sum(partial * partial, axis=1) / D + NORM_EPS)
            key = partial * inv_rms[:, None]

        logit = tl.sum(key * q[None, :], axis=1)
        lse = tl.load(interblock_lse + token_offsets, mask=token_mask, other=0.0).to(tl.float32)
        prob = 1.0 / (1.0 + tl.exp(lse - logit))
        prob = tl.where(token_mask, prob, 0.0)

        dprob = tl.sum(grad * (partial - inter), axis=1)
        dlogit = prob * (1.0 - prob) * dprob

        grad_partial_value = prob[:, None] * grad
        grad_inter = (1.0 - prob)[:, None] * grad
        grad_key = dlogit[:, None] * q[None, :]
        if KEY_NORM:
            grad_key = inv_rms[:, None] * (
                grad_key - key * tl.sum(grad_key * key, axis=1)[:, None] / D
            )
        grad_partial_total = grad_partial_value + grad_key
        grad_query_tile = tl.sum(dlogit[:, None] * key, axis=0)

        tl.store(
            grad_partial + token_offsets[:, None] * D + d_offsets[None, :],
            grad_partial_total,
            mask=mask,
        )
        tl.store(
            grad_interblock_output + token_offsets[:, None] * D + d_offsets[None, :],
            grad_inter,
            mask=mask,
        )
        tl.store(grad_interblock_lse + token_offsets, -dlogit, mask=token_mask)
        tl.atomic_add(grad_query_out + d_offsets, grad_query_tile, mask=d_mask, sem="relaxed")


    @triton.jit
    def _lrid_phase2_forward_kernel(
        partial_value,
        partial_key,
        query,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        n_tokens,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        KEY_NORM: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        LOGIT_SCALE: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        r_offsets = tl.arange(0, BLOCK_R)
        d_mask = d_offsets < D
        r_mask = r_offsets < R

        k = tl.load(partial_key + token * R + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
        q = tl.load(query + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
        if KEY_NORM:
            key_inv_rms = tl.rsqrt(tl.sum(k * k, axis=0) / R + NORM_EPS)
            k = k * key_inv_rms
        logit = tl.sum(k * q, axis=0) * LOGIT_SCALE

        v = tl.load(partial_value + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        prev = tl.load(interblock_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        prev_lse = tl.load(interblock_lse + token).to(tl.float32)
        prob = tl.sigmoid(logit - prev_lse)
        merged = prev + prob * (v - prev)
        if OUTPUT_NORM:
            out_inv_rms = tl.rsqrt(tl.sum(merged * merged, axis=0) / D + NORM_EPS)
            merged = merged * out_inv_rms
            tl.store(output_inv_rms + token, out_inv_rms)
        tl.store(output + token * D + d_offsets, merged, mask=d_mask)


    @triton.jit
    def _lrid_phase2_backward_kernel(
        partial_value,
        partial_key,
        query,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        grad_output,
        grad_partial_value,
        grad_partial_key,
        grad_query_out,
        grad_interblock_output,
        grad_interblock_lse,
        n_tokens,
        D: tl.constexpr,
        R: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
        KEY_NORM: tl.constexpr,
        OUTPUT_NORM: tl.constexpr,
        LOGIT_SCALE: tl.constexpr,
        NORM_EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_offsets = tl.arange(0, BLOCK_D)
        r_offsets = tl.arange(0, BLOCK_R)
        d_mask = d_offsets < D
        r_mask = r_offsets < R

        v = tl.load(partial_value + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        prev = tl.load(interblock_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        if OUTPUT_NORM:
            y = tl.load(output + token * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
            out_inv_rms = tl.load(output_inv_rms + token).to(tl.float32)
            grad = out_inv_rms * (grad - y * tl.sum(grad * y, axis=0) / D)

        raw_key = tl.load(partial_key + token * R + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
        q = tl.load(query + r_offsets, mask=r_mask, other=0.0).to(tl.float32)
        key = raw_key
        key_inv_rms = tl.full((), 1.0, tl.float32)
        if KEY_NORM:
            key_inv_rms = tl.rsqrt(tl.sum(raw_key * raw_key, axis=0) / R + NORM_EPS)
            key = raw_key * key_inv_rms
        logit = tl.sum(key * q, axis=0) * LOGIT_SCALE
        prev_lse = tl.load(interblock_lse + token).to(tl.float32)

        prob = tl.sigmoid(logit - prev_lse)
        prev_prob = 1.0 - prob
        merge_grad = tl.sum(grad * (v - prev), axis=0)
        dlogit = prob * prev_prob * merge_grad
        scaled_dlogit = dlogit * LOGIT_SCALE

        grad_key = scaled_dlogit * q
        if KEY_NORM:
            grad_key = key_inv_rms * (grad_key - key * tl.sum(grad_key * key, axis=0) / R)
        grad_query_vec = scaled_dlogit * key

        tl.store(grad_partial_value + token * D + d_offsets, prob * grad, mask=d_mask)
        tl.store(grad_partial_key + token * R + r_offsets, grad_key, mask=r_mask)
        tl.atomic_add(grad_query_out + r_offsets, grad_query_vec, mask=r_mask, sem="relaxed")
        tl.store(grad_interblock_output + token * D + d_offsets, prev_prob * grad, mask=d_mask)
        tl.store(grad_interblock_lse + token, -dlogit)


    @triton.jit
    def _attnres_reduce_query_grad_kernel(
        grad_partial,
        grad_query,
        n_tokens,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D_REDUCE: tl.constexpr,
    ):
        token_block = tl.program_id(0)
        query_idx = tl.program_id(1)
        d_block = tl.program_id(2)
        token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offsets = d_block * BLOCK_D_REDUCE + tl.arange(0, BLOCK_D_REDUCE)

        grad = tl.load(
            grad_partial + (query_idx * n_tokens + token_offsets[:, None]) * D + d_offsets[None, :],
            mask=(token_offsets[:, None] < n_tokens) & (d_offsets[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        reduced = tl.sum(grad, axis=0)
        tl.atomic_add(
            grad_query + query_idx * D + d_offsets,
            reduced,
            mask=d_offsets < D,
            sem="relaxed",
        )


def _can_use_triton(*tensors: Tensor) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    if not tensors:
        return False
    allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    return all(
        isinstance(t, Tensor)
        and t.is_cuda
        and t.is_contiguous()
        and t.dtype in allowed_dtypes
        for t in tensors
    )


def _can_use_triton_attnres_list(values, query: Tensor) -> bool:
    if not _TRITON_AVAILABLE or isinstance(values, Tensor):
        return False
    if len(values) < 2 or len(values) > 16:
        return False
    return _can_use_triton(*values, query)


def _attnres_read_torch_stacked(values: Tensor, query: Tensor, key_norm: bool, source_counts=None) -> Tensor:
    keys = _norm(values) if key_norm else values
    query = query.to(keys.dtype)
    logits = torch.einsum("d,sbtd->sbt", query, keys)
    log_count_bias = _source_log_count_bias(source_counts, values.size(0), logits.device)
    if log_count_bias is not None:
        logits = logits + log_count_bias.view(-1, 1, 1)
    weights = F.softmax(logits.float(), dim=0).to(values.dtype)
    return torch.einsum("sbt,sbtd->btd", weights, values)


def _attnres_read_torch_list(values, query: Tensor, key_norm: bool, source_counts=None) -> Tensor:
    query = query.to(values[0].dtype)
    log_count_bias = _source_log_count_bias(source_counts, len(values), values[0].device)
    logits = []
    for idx, value in enumerate(values):
        key = _norm(value) if key_norm else value
        logit = torch.sum(key * query, dim=-1)
        if log_count_bias is not None:
            logit = logit + log_count_bias[idx]
        logits.append(logit)
    weights = F.softmax(torch.stack(logits, dim=0).float(), dim=0).to(values[0].dtype)

    output = torch.zeros_like(values[0])
    for idx, value in enumerate(values):
        output = output + weights[idx].unsqueeze(-1) * value
    return output


def attention_residual_read_torch(values, query: Tensor, key_norm: bool, source_counts=None) -> Tensor:
    values = _as_stacked(values)
    return _attnres_read_torch_stacked(values, query, key_norm, source_counts)


def _attnres_read_triton(values: Tensor, query: Tensor, key_norm: bool, normalize_output: bool = False) -> Tensor:
    S, B, T, D = values.shape
    n_tokens = B * T
    if D > 4096:
        raise RuntimeError("Fused AttnRes only supports n_embd <= 4096")

    values = values.contiguous()
    query = query.contiguous()
    output = torch.empty((B, T, D), device=values.device, dtype=values.dtype)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    num_warps = 8 if block_d >= 2048 else 4

    _attnres_read_kernel[(n_tokens,)](
        values,
        query,
        output,
        n_tokens,
        S,
        D,
        s_block,
        block_d,
        key_norm,
        normalize_output,
        torch.finfo(values.dtype).eps,
        num_warps=num_warps,
    )
    return output


def _attnres_read_list_triton(
    values,
    query: Tensor,
    key_norm: bool,
    normalize_output: bool = False,
    return_inv_rms: bool = False,
):
    S = len(values)
    B, T, D = values[0].shape
    n_tokens = B * T
    if D > 4096:
        raise RuntimeError("Fused AttnRes only supports n_embd <= 4096")

    query = query.contiguous()
    output = torch.empty((B, T, D), device=values[0].device, dtype=values[0].dtype)
    output_inv_rms = (
        torch.empty((n_tokens,), device=values[0].device, dtype=torch.float32)
        if normalize_output
        else torch.empty((1,), device=values[0].device, dtype=torch.float32)
    )
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    num_warps = 8 if block_d >= 2048 else 4
    padded_values = list(values) + [values[0]] * (16 - S)
    _attnres_list_read_kernel[(n_tokens,)](
        *padded_values,
        query,
        output,
        output_inv_rms,
        n_tokens,
        S,
        D,
        s_block,
        block_d,
        key_norm,
        normalize_output,
        torch.finfo(values[0].dtype).eps,
        num_warps=num_warps,
    )
    if return_inv_rms:
        return output, output_inv_rms
    return output


def _attnres_read_backward_triton(
    values: Tensor,
    query: Tensor,
    grad_output: Tensor,
    key_norm: bool,
):
    S, B, T, D = values.shape
    n_tokens = B * T
    values = values.contiguous()
    query = query.contiguous()
    grad_output = grad_output.contiguous()
    grad_values = torch.empty_like(values)
    grad_query = torch.zeros(query.shape, device=query.device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    _attnres_stacked_backward_kernel[(n_tokens,)](
        values,
        query,
        grad_output,
        grad_values,
        grad_query,
        n_tokens,
        S,
        D,
        s_block,
        block_d,
        bool(key_norm),
        torch.finfo(values.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return grad_values, grad_query.to(query.dtype)


def _attnres_read_list_backward_triton(
    values,
    query: Tensor,
    grad_output: Tensor,
    key_norm: bool,
    normalize_output: bool = False,
    normed_output: Tensor = None,
    output_inv_rms: Tensor = None,
):
    S = len(values)
    B, T, D = values[0].shape
    n_tokens = B * T
    grad_output = grad_output.contiguous()
    if normalize_output:
        normed_output = normed_output.contiguous()
    else:
        normed_output = grad_output
        output_inv_rms = torch.empty((1,), device=values[0].device, dtype=torch.float32)
    grad_values = [torch.empty_like(value) for value in values]
    grad_query_accum = torch.zeros(query.shape, device=query.device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    num_warps = 8 if block_d >= 2048 else 4
    padded_values = list(values) + [values[0]] * (16 - S)
    padded_grad_values = grad_values + [grad_values[0]] * (16 - S)
    _attnres_list_backward_kernel[(n_tokens,)](
        *padded_values,
        query.contiguous(),
        grad_output,
        normed_output,
        output_inv_rms,
        *padded_grad_values,
        grad_query_accum,
        n_tokens,
        S,
        D,
        s_block,
        block_d,
        key_norm,
        normalize_output,
        torch.finfo(values[0].dtype).eps,
        num_warps=num_warps,
    )
    return grad_values, grad_query_accum.to(dtype=query.dtype)


class _TritonAttentionResidualRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: Tensor, query: Tensor, key_norm: bool, normalize_output: bool):
        ctx.save_for_backward(values, query)
        ctx.key_norm = bool(key_norm)
        ctx.normalize_output = bool(normalize_output)
        return _attnres_read_triton(values, query, bool(key_norm), bool(normalize_output))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        values, query = ctx.saved_tensors
        needs_values, needs_query = ctx.needs_input_grad[:2]
        if not ctx.normalize_output:
            grad_values, grad_query = _attnres_read_backward_triton(
                values,
                query,
                grad_output,
                ctx.key_norm,
            )
            return grad_values if needs_values else None, grad_query if needs_query else None, None, None
        with torch.enable_grad():
            values_ = values.detach().requires_grad_(True)
            query_ = query.detach().requires_grad_(True)
            output = _attnres_read_torch_stacked(values_, query_, ctx.key_norm)
            if ctx.normalize_output:
                output = _norm(output)
        grad_values, grad_query = torch.autograd.grad(
            output,
            (values_, query_),
            grad_output,
            allow_unused=True,
        )
        return grad_values if needs_values else None, grad_query if needs_query else None, None, None


class _TritonAttentionResidualListRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key_norm: bool, normalize_output: bool, *values):
        if normalize_output:
            output, output_inv_rms = _attnres_read_list_triton(
                values,
                query,
                bool(key_norm),
                normalize_output=True,
                return_inv_rms=True,
            )
            ctx.save_for_backward(query, output, output_inv_rms, *values)
        else:
            output = _attnres_read_list_triton(values, query, bool(key_norm))
            ctx.save_for_backward(query, *values)
        ctx.key_norm = bool(key_norm)
        ctx.normalize_output = bool(normalize_output)
        ctx.num_prefix_tensors = 3 if normalize_output else 1
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        saved = ctx.saved_tensors
        query = saved[0]
        if ctx.normalize_output:
            normed_output = saved[1]
            output_inv_rms = saved[2]
            values = saved[3:]
        else:
            normed_output = None
            output_inv_rms = None
            values = saved[1:]
        grad_values, grad_query = _attnres_read_list_backward_triton(
            values,
            query,
            grad_output,
            ctx.key_norm,
            ctx.normalize_output,
            normed_output,
            output_inv_rms,
        )
        return (grad_query, None, None, *grad_values)


def attention_residual_read(
    values,
    query: Tensor,
    key_norm: bool,
    force_triton: bool = False,
    normalize_output: bool = False,
    source_counts=None,
) -> Tensor:
    if not isinstance(values, Tensor):
        if len(values) == 1:
            output = values[0]
            return _norm(output) if normalize_output else output
        query = query.to(values[0].dtype)
        if torch.is_grad_enabled():
            if source_counts is None and _use_triton_training_kernel() and _can_use_triton_attnres_list(values, query):
                return _TritonAttentionResidualListRead.apply(
                    query,
                    bool(key_norm),
                    bool(normalize_output),
                    *values,
                )
            else:
                if source_counts is not None and force_triton:
                    raise RuntimeError("Triton fused AttnRes path does not support source_counts")
                output = _attnres_read_torch_list(values, query, key_norm, source_counts)
            return _norm(output) if normalize_output else output
        if source_counts is None and _can_use_triton_attnres_list(values, query):
            return _attnres_read_list_triton(values, query, bool(key_norm), bool(normalize_output))
        if force_triton:
            raise RuntimeError("Triton fused AttnRes path is not available for these tensors")
        output = _attnres_read_torch_list(values, query, key_norm, source_counts)
        return _norm(output) if normalize_output else output

    values = _as_stacked(values)
    if values.size(0) == 1:
        output = values[0]
        return _norm(output) if normalize_output else output
    query = query.to(values.dtype)
    if source_counts is None and _can_use_triton(values, query):
        return _TritonAttentionResidualRead.apply(values, query, bool(key_norm), bool(normalize_output))
    if force_triton:
        raise RuntimeError("Triton fused AttnRes path is not available for these tensors")
    output = _attnres_read_torch_stacked(values, query, key_norm, source_counts)
    return _norm(output) if normalize_output else output


def attention_residual_average_read(
    values,
    normalize_output: bool = False,
    force_triton: bool = False,
) -> Tensor:
    values = _as_stacked(values) if isinstance(values, Tensor) else values
    if isinstance(values, Tensor):
        if values.size(0) == 1:
            output = values[0]
            return _norm(output) if normalize_output else output
        output = values.mean(dim=0)
        return _norm(output) if normalize_output else output

    if len(values) == 1:
        output = values[0]
        return _norm(output) if normalize_output else output
    if len(values) > 16:
        if force_triton:
            raise RuntimeError("Triton average AttnRes path supports at most 16 sources")
        output = torch.stack(values, dim=0).mean(dim=0)
        return _norm(output) if normalize_output else output

    first = values[0]
    if not _can_use_triton(*values):
        if force_triton:
            raise RuntimeError("Triton average AttnRes path is not available for these tensors")
        output = torch.stack(values, dim=0).mean(dim=0)
        return _norm(output) if normalize_output else output

    B, T, D = first.shape
    n_tokens = B * T
    if D > 4096:
        raise RuntimeError("Fused average AttnRes only supports n_embd <= 4096")
    output = torch.empty((B, T, D), device=first.device, dtype=first.dtype)
    s_block = _next_power_of_2(len(values))
    padded_values = list(values) + [values[0]] * (16 - len(values))
    block_d = _next_power_of_2(D)
    _attnres_list_average_kernel[(n_tokens,)](
        *padded_values,
        output,
        n_tokens,
        len(values),
        D,
        s_block,
        block_d,
        normalize_output,
        torch.finfo(first.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return output


def _attnres_phase1_read_torch(values: Tensor, queries: Tensor, key_norm: bool):
    keys = _norm(values) if key_norm else values
    queries = queries.to(keys.dtype)
    logits = torch.einsum("qd,sbtd->qsbt", queries, keys)
    lse = torch.logsumexp(logits.float(), dim=1)
    weights = F.softmax(logits.float(), dim=1).to(values.dtype)
    output = torch.einsum("qsbt,sbtd->qbtd", weights, values)
    return output, lse


def _attnres_phase2_merge_torch(
    partial_value: Tensor,
    query: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    key_norm: bool,
    logit_bias: float = 0.0,
) -> Tensor:
    key = _norm(partial_value) if key_norm else partial_value
    logit = torch.sum(key * query.to(key.dtype), dim=-1).float()
    if logit_bias:
        logit = logit + float(logit_bias)
    prob = torch.sigmoid(logit - interblock_lse.float()).to(partial_value.dtype)
    return torch.addcmul(interblock_output, partial_value - interblock_output, prob.unsqueeze(-1))


def _attnres_phase1_read_triton(values: Tensor, queries: Tensor, key_norm: bool):
    S, B, T, D = values.shape
    Q = queries.size(0)
    n_tokens = B * T
    if D > 4096 or Q < 1 or Q > 16 or S < 1 or S > 16:
        raise RuntimeError("Two-phase AttnRes dimensions exceed the Triton kernel limits")

    values = values.contiguous()
    queries = queries.contiguous()
    output = torch.empty((Q, B, T, D), device=values.device, dtype=values.dtype)
    lse = torch.empty((Q, B, T), device=values.device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    _attnres_phase1_forward_kernel[(n_tokens,)](
        values,
        queries,
        output,
        lse,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        bool(key_norm),
        torch.finfo(values.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return output, lse


def _attnres_reduce_query_grad(grad_partial: Tensor, n_queries: int, n_tokens: int, D: int):
    grad_query = torch.zeros((n_queries, D), device=grad_partial.device, dtype=torch.float32)
    block_m = 256
    block_d = 64 if D <= 1024 else 128
    _attnres_reduce_query_grad_kernel[
        (triton.cdiv(n_tokens, block_m), n_queries, triton.cdiv(D, block_d))
    ](
        grad_partial,
        grad_query,
        n_tokens,
        D,
        block_m,
        block_d,
        num_warps=4,
    )
    return grad_query


def _attnres_phase1_backward_triton(
    values: Tensor,
    queries: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    grad_lse: Tensor,
    key_norm: bool,
):
    S, B, T, D = values.shape
    Q = queries.size(0)
    n_tokens = B * T
    if grad_output is None:
        grad_output = torch.zeros((Q, B, T, D), device=values.device, dtype=values.dtype)
    else:
        grad_output = grad_output.contiguous()
    has_grad_lse = grad_lse is not None
    if has_grad_lse:
        grad_lse = grad_lse.contiguous()
    else:
        grad_lse = torch.empty((1,), device=values.device, dtype=torch.float32)

    grad_values = torch.empty((S, B, T, D), device=values.device, dtype=torch.float32)
    grad_queries_partial = torch.empty((Q, n_tokens, D), device=values.device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    _attnres_phase1_backward_kernel[(n_tokens,)](
        values,
        queries,
        lse,
        grad_output,
        grad_lse,
        grad_values,
        grad_queries_partial,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        bool(key_norm),
        has_grad_lse,
        torch.finfo(values.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    grad_queries = _attnres_reduce_query_grad(grad_queries_partial, Q, n_tokens, D)
    return grad_values.to(values.dtype), grad_queries.to(queries.dtype)


def _attnres_phase1_logits_read_triton(values: Tensor, logits: Tensor, normalize_output: bool):
    S, B, T, D = values.shape
    Q = logits.size(0)
    n_tokens = B * T
    if D > 4096 or Q < 1 or Q > 16 or S < 1 or S > 16:
        raise RuntimeError("Cached-logit AttnRes dimensions exceed the Triton kernel limits")
    values = values.contiguous()
    logits = logits.contiguous()
    output = torch.empty((Q, B, T, D), device=values.device, dtype=values.dtype)
    lse = torch.empty((Q, B, T), device=values.device, dtype=torch.float32)
    saved_weights = torch.empty((Q, S, B, T), device=values.device, dtype=values.dtype)
    output_inv_rms = (
        torch.empty((Q, B, T), device=values.device, dtype=torch.float32)
        if normalize_output
        else torch.empty((1,), device=values.device, dtype=torch.float32)
    )
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    _attnres_phase1_logits_forward_kernel[(n_tokens,)](
        values,
        logits,
        output,
        lse,
        output_inv_rms,
        saved_weights,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        bool(normalize_output),
        torch.finfo(values.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return output, lse, output_inv_rms, saved_weights


def _attnres_phase1_logits_read_tiled_triton(values: Tensor, logits: Tensor):
    S, B, T, D = values.shape
    Q = logits.size(0)
    n_tokens = B * T
    values = values.contiguous()
    logits = logits.contiguous()
    output = torch.empty((Q, B, T, D), device=values.device, dtype=values.dtype)
    lse = torch.empty((Q, B, T), device=values.device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = 256 if D >= 256 else _next_power_of_2(D)
    grid = (n_tokens, Q, triton.cdiv(D, block_d))
    _attnres_phase1_logits_forward_tiled_kernel[grid](
        values,
        logits,
        output,
        lse,
        n_tokens,
        S,
        D,
        s_block,
        block_d,
        num_warps=4,
    )
    return output, lse


def _attnres_phase1_logits_read_qsplit_triton(values: Tensor, logits: Tensor):
    S, B, T, D = values.shape
    Q = logits.size(0)
    n_tokens = B * T
    values = values.contiguous()
    logits = logits.contiguous()
    output = torch.empty((Q, B, T, D), device=values.device, dtype=values.dtype)
    lse = torch.empty((Q, B, T), device=values.device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    _attnres_phase1_logits_forward_qsplit_kernel[(n_tokens, Q)](
        values,
        logits,
        output,
        lse,
        n_tokens,
        S,
        D,
        s_block,
        block_d,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return output, lse


def _attnres_phase1_logits_backward_triton(
    values: Tensor,
    logits: Tensor,
    lse: Tensor,
    output: Tensor,
    output_inv_rms: Tensor,
    grad_output: Tensor,
    grad_lse: Tensor,
    normalize_output: bool,
):
    S, B, T, D = values.shape
    Q = logits.size(0)
    n_tokens = B * T
    if grad_output is None:
        grad_output = torch.zeros((Q, B, T, D), device=values.device, dtype=values.dtype)
    else:
        grad_output = grad_output.contiguous()
    has_grad_lse = grad_lse is not None
    if has_grad_lse:
        grad_lse = grad_lse.contiguous()
    else:
        grad_lse = torch.empty((1,), device=values.device, dtype=torch.float32)
    values = values.contiguous()
    logits = logits.contiguous()
    grad_values = torch.empty_like(values)
    grad_logits = torch.empty_like(logits)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    _attnres_phase1_logits_backward_kernel[(n_tokens,)](
        values,
        logits,
        lse,
        output,
        output_inv_rms,
        grad_output,
        grad_lse,
        grad_values,
        grad_logits,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        has_grad_lse,
        bool(normalize_output),
        num_warps=8 if block_d >= 2048 else 4,
    )
    return grad_values, grad_logits


def _attnres_phase1_logits_backward_weights_triton(
    values: Tensor,
    saved_weights: Tensor,
    output: Tensor,
    output_inv_rms: Tensor,
    grad_output: Tensor,
    grad_lse: Tensor,
    normalize_output: bool,
):
    S, B, T, D = values.shape
    Q = saved_weights.size(0)
    n_tokens = B * T
    if grad_output is None:
        grad_output = torch.zeros((Q, B, T, D), device=values.device, dtype=values.dtype)
    else:
        grad_output = grad_output.contiguous()
    has_grad_lse = grad_lse is not None
    if has_grad_lse:
        grad_lse = grad_lse.contiguous()
    else:
        grad_lse = torch.empty((1,), device=values.device, dtype=torch.float32)
    values = values.contiguous()
    saved_weights = saved_weights.contiguous()
    grad_values = torch.empty_like(values)
    grad_logits = torch.empty((Q, S, B, T), device=saved_weights.device, dtype=values.dtype)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    _attnres_phase1_logits_backward_weights_kernel[(n_tokens,)](
        values,
        saved_weights,
        output,
        output_inv_rms,
        grad_output,
        grad_lse,
        grad_values,
        grad_logits,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        has_grad_lse,
        bool(normalize_output),
        num_warps=8 if block_d >= 2048 else 4,
    )
    return grad_values, grad_logits


def _attnres_phase1_logits_read_list_triton(values, logits: Tensor, normalize_output: bool):
    S = len(values)
    B, T, D = values[0].shape
    Q = logits.size(0)
    n_tokens = B * T
    if D > 4096 or Q < 1 or Q > 16 or S < 1 or S > 16:
        raise RuntimeError("Cached-logit AttnRes dimensions exceed the Triton list-kernel limits")

    values = tuple(value.contiguous() for value in values)
    logits = logits.contiguous()
    output = torch.empty((Q, B, T, D), device=values[0].device, dtype=values[0].dtype)
    lse = torch.empty((Q, B, T), device=values[0].device, dtype=torch.float32)
    saved_weights = torch.empty((Q, S, B, T), device=values[0].device, dtype=values[0].dtype)
    output_inv_rms = (
        torch.empty((Q, B, T), device=values[0].device, dtype=torch.float32)
        if normalize_output
        else torch.empty((1,), device=values[0].device, dtype=torch.float32)
    )
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    padded_values = list(values) + [values[0]] * (16 - S)
    _attnres_phase1_logits_list_forward_kernel[(n_tokens,)](
        *padded_values,
        logits,
        output,
        lse,
        output_inv_rms,
        saved_weights,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        bool(normalize_output),
        torch.finfo(values[0].dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return output, lse, output_inv_rms, saved_weights, values


def _attnres_phase1_logits_backward_list_weights_triton(
    values,
    saved_weights: Tensor,
    output: Tensor,
    output_inv_rms: Tensor,
    grad_output: Tensor,
    grad_lse: Tensor,
    normalize_output: bool,
):
    S = len(values)
    B, T, D = values[0].shape
    Q = saved_weights.size(0)
    n_tokens = B * T
    if grad_output is None:
        grad_output = torch.zeros((Q, B, T, D), device=values[0].device, dtype=values[0].dtype)
    else:
        grad_output = grad_output.contiguous()
    has_grad_lse = grad_lse is not None
    if has_grad_lse:
        grad_lse = grad_lse.contiguous()
    else:
        grad_lse = torch.empty((1,), device=values[0].device, dtype=torch.float32)

    values = tuple(value.contiguous() for value in values)
    saved_weights = saved_weights.contiguous()
    grad_values = [torch.empty_like(value) for value in values]
    grad_logits = torch.empty((Q, S, B, T), device=saved_weights.device, dtype=values[0].dtype)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    padded_values = list(values) + [values[0]] * (16 - S)
    padded_grad_values = grad_values + [grad_values[0]] * (16 - S)
    _attnres_phase1_logits_list_backward_weights_kernel[(n_tokens,)](
        *padded_values,
        saved_weights,
        output,
        output_inv_rms,
        grad_output,
        grad_lse,
        *padded_grad_values,
        grad_logits,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        has_grad_lse,
        bool(normalize_output),
        num_warps=8 if block_d >= 2048 else 4,
    )
    return grad_values, grad_logits


def _attnres_phase1_logits_backward_qsplit_triton(
    values: Tensor,
    logits: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    grad_lse: Tensor,
):
    S, B, T, D = values.shape
    Q = logits.size(0)
    n_tokens = B * T
    if grad_output is None:
        grad_output = torch.zeros((Q, B, T, D), device=values.device, dtype=values.dtype)
    else:
        grad_output = grad_output.contiguous()
    has_grad_lse = grad_lse is not None
    if has_grad_lse:
        grad_lse = grad_lse.contiguous()
    else:
        grad_lse = torch.empty((1,), device=values.device, dtype=torch.float32)
    values = values.contiguous()
    logits = logits.contiguous()
    grad_values = torch.zeros((S, B, T, D), device=values.device, dtype=torch.float32)
    grad_logits = torch.empty((Q, S, B, T), device=logits.device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    _attnres_phase1_logits_backward_qsplit_kernel[(n_tokens, Q)](
        values,
        logits,
        lse,
        grad_output,
        grad_lse,
        grad_values,
        grad_logits,
        n_tokens,
        S,
        D,
        s_block,
        block_d,
        has_grad_lse,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return grad_values.to(values.dtype), grad_logits.to(logits.dtype)


def _attnres_phase1_logits_backward_tiled_triton(
    values: Tensor,
    logits: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    grad_lse: Tensor,
):
    S, B, T, D = values.shape
    Q = logits.size(0)
    n_tokens = B * T
    if grad_output is None:
        grad_output = torch.zeros((Q, B, T, D), device=values.device, dtype=values.dtype)
    else:
        grad_output = grad_output.contiguous()
    has_grad_lse = grad_lse is not None
    if has_grad_lse:
        grad_lse = grad_lse.contiguous()
    else:
        grad_lse = torch.empty((1,), device=values.device, dtype=torch.float32)
    values = values.contiguous()
    logits = logits.contiguous()
    s_block = _next_power_of_2(S)
    block_d = 256 if D >= 256 else _next_power_of_2(D)
    n_dblocks = triton.cdiv(D, block_d)
    partial_dweights = torch.empty((Q, S, n_tokens, n_dblocks), device=values.device, dtype=torch.float32)
    grad_values = torch.empty((S, B, T, D), device=values.device, dtype=torch.float32)
    grad_logits = torch.empty((Q, S, B, T), device=logits.device, dtype=torch.float32)

    _attnres_phase1_logits_dweights_kernel[
        (n_tokens, Q, n_dblocks)
    ](
        values,
        grad_output,
        partial_dweights,
        n_tokens,
        S,
        D,
        s_block,
        block_d,
        n_dblocks,
        num_warps=4,
    )
    _attnres_phase1_logits_dlogits_kernel[(n_tokens, Q)](
        logits,
        lse,
        grad_lse,
        partial_dweights,
        grad_logits,
        n_tokens,
        S,
        s_block,
        n_dblocks,
        has_grad_lse,
        num_warps=1,
    )
    _attnres_phase1_logits_grad_values_kernel[
        (n_tokens, n_dblocks)
    ](
        values,
        logits,
        lse,
        grad_output,
        grad_logits,
        grad_values,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        num_warps=4,
    )
    return grad_values.to(values.dtype), grad_logits.to(logits.dtype)


def _attnres_phase1_list_read_triton(values, queries: Tensor, key_norm: bool):
    S = len(values)
    B, T, D = values[0].shape
    Q = queries.size(0)
    n_tokens = B * T
    if D > 4096 or Q < 1 or Q > 16 or S < 1 or S > 16:
        raise RuntimeError("Two-phase AttnRes dimensions exceed the Triton kernel limits")

    queries = queries.contiguous()
    output = torch.empty((Q, B, T, D), device=values[0].device, dtype=values[0].dtype)
    lse = torch.empty((Q, B, T), device=values[0].device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    padded_values = list(values) + [values[0]] * (16 - S)
    _attnres_phase1_list_forward_kernel[(n_tokens,)](
        *padded_values,
        queries,
        output,
        lse,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        bool(key_norm),
        torch.finfo(values[0].dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return output, lse


def _attnres_phase1_list_backward_triton(
    values,
    queries: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    grad_lse: Tensor,
    key_norm: bool,
):
    S = len(values)
    B, T, D = values[0].shape
    Q = queries.size(0)
    n_tokens = B * T
    if grad_output is None:
        grad_output = torch.zeros((Q, B, T, D), device=values[0].device, dtype=values[0].dtype)
    else:
        grad_output = grad_output.contiguous()
    has_grad_lse = grad_lse is not None
    if has_grad_lse:
        grad_lse = grad_lse.contiguous()
    else:
        grad_lse = torch.empty((1,), device=values[0].device, dtype=torch.float32)

    grad_values = [torch.empty_like(value) for value in values]
    grad_queries_partial = torch.empty((Q, n_tokens, D), device=values[0].device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(D)
    padded_values = list(values) + [values[0]] * (16 - S)
    padded_grad_values = grad_values + [grad_values[0]] * (16 - S)
    _attnres_phase1_list_backward_kernel[(n_tokens,)](
        *padded_values,
        queries,
        lse,
        grad_output,
        grad_lse,
        *padded_grad_values,
        grad_queries_partial,
        n_tokens,
        S,
        Q,
        D,
        s_block,
        block_d,
        bool(key_norm),
        has_grad_lse,
        torch.finfo(values[0].dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    grad_queries = _attnres_reduce_query_grad(grad_queries_partial, Q, n_tokens, D)
    return grad_values, grad_queries.to(queries.dtype)


def _attnres_phase2_merge_triton(
    partial_value: Tensor,
    query: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    key_norm: bool,
    normalize_output: bool,
):
    B, T, D = partial_value.shape
    n_tokens = B * T
    partial_value = partial_value.contiguous()
    query = query.contiguous()
    interblock_output = interblock_output.contiguous()
    interblock_lse = interblock_lse.contiguous()
    output = torch.empty_like(partial_value)
    output_inv_rms = (
        torch.empty((B, T), device=partial_value.device, dtype=torch.float32)
        if normalize_output
        else torch.empty((1,), device=partial_value.device, dtype=torch.float32)
    )
    block_d = _next_power_of_2(D)
    _attnres_phase2_forward_kernel[(n_tokens,)](
        partial_value,
        query,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        n_tokens,
        D,
        block_d,
        bool(key_norm),
        bool(normalize_output),
        torch.finfo(partial_value.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return output, output_inv_rms


def _attnres_phase2_backward_triton(
    partial_value: Tensor,
    query: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    output: Tensor,
    output_inv_rms: Tensor,
    grad_output: Tensor,
    key_norm: bool,
    normalize_output: bool,
):
    B, T, D = partial_value.shape
    n_tokens = B * T
    grad_output = grad_output.contiguous()
    if os.environ.get("ATTNRES_PHASE2_FUSED_QUERY", "1") == "1":
        grad_partial = torch.empty((B, T, D), device=partial_value.device, dtype=torch.float32)
        grad_query = torch.zeros((D,), device=query.device, dtype=torch.float32)
        grad_interblock_output = torch.empty((B, T, D), device=partial_value.device, dtype=torch.float32)
        grad_interblock_lse = torch.empty((B, T), device=partial_value.device, dtype=torch.float32)
        block_m = 16
        block_d = _next_power_of_2(D)
        _attnres_phase2_backward_fused_query_kernel[(triton.cdiv(n_tokens, block_m),)](
            partial_value,
            query,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
            grad_output,
            grad_partial,
            grad_query,
            grad_interblock_output,
            grad_interblock_lse,
            n_tokens,
            D,
            block_m,
            block_d,
            bool(key_norm),
            bool(normalize_output),
            torch.finfo(partial_value.dtype).eps,
            num_warps=8 if block_d >= 1024 else 4,
        )
        return (
            grad_partial.to(partial_value.dtype),
            grad_query.to(query.dtype),
            grad_interblock_output.to(interblock_output.dtype),
            grad_interblock_lse,
        )
    grad_partial = torch.empty((B, T, D), device=partial_value.device, dtype=torch.float32)
    grad_query_partial = torch.empty((n_tokens, D), device=partial_value.device, dtype=torch.float32)
    grad_interblock_output = torch.empty((B, T, D), device=partial_value.device, dtype=torch.float32)
    grad_interblock_lse = torch.empty((B, T), device=partial_value.device, dtype=torch.float32)
    block_d = _next_power_of_2(D)
    _attnres_phase2_backward_kernel[(n_tokens,)](
        partial_value,
        query,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        grad_output,
        grad_partial,
        grad_query_partial,
        grad_interblock_output,
        grad_interblock_lse,
        n_tokens,
        D,
        block_d,
        bool(key_norm),
        bool(normalize_output),
        torch.finfo(partial_value.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    grad_query = _attnres_reduce_query_grad(grad_query_partial.view(1, n_tokens, D), 1, n_tokens, D)[0]
    return (
        grad_partial.to(partial_value.dtype),
        grad_query.to(query.dtype),
        grad_interblock_output.to(interblock_output.dtype),
        grad_interblock_lse,
    )


def _attnres_phase2_logits_merge_triton(
    partial_value: Tensor,
    partial_logit: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    normalize_output: bool,
):
    B, T, D = partial_value.shape
    n_tokens = B * T
    partial_value = partial_value.contiguous()
    partial_logit = partial_logit.contiguous()
    interblock_output = interblock_output.contiguous()
    interblock_lse = interblock_lse.contiguous()
    output = torch.empty_like(partial_value)
    output_inv_rms = (
        torch.empty((B, T), device=partial_value.device, dtype=torch.float32)
        if normalize_output
        else torch.empty((1,), device=partial_value.device, dtype=torch.float32)
    )
    block_d = _next_power_of_2(D)
    _attnres_phase2_logits_forward_kernel[(n_tokens,)](
        partial_value,
        partial_logit,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        n_tokens,
        D,
        block_d,
        bool(normalize_output),
        torch.finfo(partial_value.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return output, output_inv_rms


def _attnres_phase2_logits_backward_triton(
    partial_value: Tensor,
    partial_logit: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    output: Tensor,
    output_inv_rms: Tensor,
    grad_output: Tensor,
    normalize_output: bool,
):
    B, T, D = partial_value.shape
    n_tokens = B * T
    grad_output = grad_output.contiguous()
    grad_partial = torch.empty((B, T, D), device=partial_value.device, dtype=torch.float32)
    grad_logit = torch.empty((B, T), device=partial_value.device, dtype=torch.float32)
    grad_interblock_output = torch.empty((B, T, D), device=partial_value.device, dtype=torch.float32)
    grad_interblock_lse = torch.empty((B, T), device=partial_value.device, dtype=torch.float32)
    block_d = _next_power_of_2(D)
    _attnres_phase2_logits_backward_kernel[(n_tokens,)](
        partial_value,
        partial_logit,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        grad_output,
        grad_partial,
        grad_logit,
        grad_interblock_output,
        grad_interblock_lse,
        n_tokens,
        D,
        block_d,
        bool(normalize_output),
        num_warps=8 if block_d >= 2048 else 4,
    )
    return (
        grad_partial.to(partial_value.dtype),
        grad_logit.to(partial_logit.dtype),
        grad_interblock_output.to(interblock_output.dtype),
        grad_interblock_lse,
    )


def _lrid_phase2_merge_triton(
    partial_value: Tensor,
    partial_key: Tensor,
    query: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    logit_scale: float,
    key_norm: bool,
    normalize_output: bool,
):
    B, T, D = partial_value.shape
    R = partial_key.size(-1)
    n_tokens = B * T
    partial_value = partial_value.contiguous()
    partial_key = partial_key.contiguous()
    query = query.contiguous()
    interblock_output = interblock_output.contiguous()
    interblock_lse = interblock_lse.contiguous()
    output = torch.empty_like(partial_value)
    output_inv_rms = (
        torch.empty((B, T), device=partial_value.device, dtype=torch.float32)
        if normalize_output
        else torch.empty((1,), device=partial_value.device, dtype=torch.float32)
    )
    block_d = _next_power_of_2(D)
    block_r = _next_power_of_2(R)
    _lrid_phase2_forward_kernel[(n_tokens,)](
        partial_value,
        partial_key,
        query,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        n_tokens,
        D,
        R,
        block_d,
        block_r,
        bool(key_norm),
        bool(normalize_output),
        float(logit_scale),
        torch.finfo(partial_value.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return output, output_inv_rms


def _lrid_phase2_backward_triton(
    partial_value: Tensor,
    partial_key: Tensor,
    query: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    output: Tensor,
    output_inv_rms: Tensor,
    grad_output: Tensor,
    logit_scale: float,
    key_norm: bool,
    normalize_output: bool,
):
    B, T, D = partial_value.shape
    R = partial_key.size(-1)
    n_tokens = B * T
    grad_output = grad_output.contiguous()
    grad_partial = torch.empty((B, T, D), device=partial_value.device, dtype=torch.float32)
    grad_key = torch.empty((B, T, R), device=partial_value.device, dtype=torch.float32)
    grad_query = torch.zeros((R,), device=partial_value.device, dtype=torch.float32)
    grad_interblock_output = torch.empty((B, T, D), device=partial_value.device, dtype=torch.float32)
    grad_interblock_lse = torch.empty((B, T), device=partial_value.device, dtype=torch.float32)
    block_d = _next_power_of_2(D)
    block_r = _next_power_of_2(R)
    _lrid_phase2_backward_kernel[(n_tokens,)](
        partial_value,
        partial_key,
        query,
        interblock_output,
        interblock_lse,
        output,
        output_inv_rms,
        grad_output,
        grad_partial,
        grad_key,
        grad_query,
        grad_interblock_output,
        grad_interblock_lse,
        n_tokens,
        D,
        R,
        block_d,
        block_r,
        bool(key_norm),
        bool(normalize_output),
        float(logit_scale),
        torch.finfo(partial_value.dtype).eps,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return (
        grad_partial.to(partial_value.dtype),
        grad_key.to(partial_key.dtype),
        grad_query.to(query.dtype),
        grad_interblock_output.to(interblock_output.dtype),
        grad_interblock_lse,
    )


class _TritonAttentionResidualPhase1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: Tensor, queries: Tensor, key_norm: bool):
        output, lse = _attnres_phase1_read_triton(values, queries, bool(key_norm))
        ctx.save_for_backward(values, queries, lse)
        ctx.key_norm = bool(key_norm)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_lse: Tensor):
        values, queries, lse = ctx.saved_tensors
        grad_values, grad_queries = _attnres_phase1_backward_triton(
            values,
            queries,
            lse,
            grad_output,
            grad_lse,
            ctx.key_norm,
        )
        return grad_values, grad_queries, None


class _TritonAttentionResidualPhase1Logits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: Tensor, logits: Tensor, normalize_output: bool):
        normalize_output = bool(normalize_output)
        use_qsplit = (not normalize_output) and os.environ.get("ATTNRES_PHASE1_QSPLIT", "0") == "1"
        use_tiled = (not normalize_output) and os.environ.get("ATTNRES_PHASE1_TILED", "0") == "1"
        if use_qsplit:
            output, lse = _attnres_phase1_logits_read_qsplit_triton(values, logits)
        elif use_tiled:
            output, lse = _attnres_phase1_logits_read_tiled_triton(values, logits)
        else:
            output, lse, output_inv_rms, saved_weights = _attnres_phase1_logits_read_triton(
                values,
                logits,
                normalize_output,
            )
        if use_qsplit or use_tiled:
            output_inv_rms = torch.empty((1,), device=values.device, dtype=torch.float32)
            saved_weights = torch.empty((1,), device=values.device, dtype=torch.float32)
        ctx.save_for_backward(values, logits, lse, output, output_inv_rms, saved_weights)
        ctx.use_qsplit = use_qsplit
        ctx.use_tiled = use_tiled
        ctx.normalize_output = normalize_output
        return output, lse

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_lse: Tensor):
        values, logits, lse, output, output_inv_rms, saved_weights = ctx.saved_tensors
        if ctx.use_qsplit:
            grad_values, grad_logits = _attnres_phase1_logits_backward_qsplit_triton(
                values,
                logits,
                lse,
                grad_output,
                grad_lse,
            )
        elif ctx.use_tiled:
            grad_values, grad_logits = _attnres_phase1_logits_backward_tiled_triton(
                values,
                logits,
                lse,
                grad_output,
                grad_lse,
            )
        else:
            grad_values, grad_logits = _attnres_phase1_logits_backward_weights_triton(
                values,
                saved_weights,
                output,
                output_inv_rms,
                grad_output,
                grad_lse,
                ctx.normalize_output,
            )
            grad_logits = grad_logits.to(logits.dtype)
        return grad_values, grad_logits, None


class _TritonAttentionResidualPhase1LogitsList(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, normalize_output: bool, *values):
        normalize_output = bool(normalize_output)
        output, lse, output_inv_rms, saved_weights, kernel_values = _attnres_phase1_logits_read_list_triton(
            values,
            logits,
            normalize_output,
        )
        ctx.save_for_backward(logits, output, output_inv_rms, saved_weights, *kernel_values)
        ctx.normalize_output = normalize_output
        return output, lse

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_lse: Tensor):
        saved = ctx.saved_tensors
        logits = saved[0]
        output = saved[1]
        output_inv_rms = saved[2]
        saved_weights = saved[3]
        values = saved[4:]
        grad_values, grad_logits = _attnres_phase1_logits_backward_list_weights_triton(
            values,
            saved_weights,
            output,
            output_inv_rms,
            grad_output,
            grad_lse,
            ctx.normalize_output,
        )
        return (grad_logits.to(logits.dtype), None, *grad_values)


class _TritonAttentionResidualPhase1List(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries: Tensor, key_norm: bool, *values):
        output, lse = _attnres_phase1_list_read_triton(values, queries, bool(key_norm))
        ctx.save_for_backward(queries, lse, *values)
        ctx.key_norm = bool(key_norm)
        ctx.num_values = len(values)
        return output, lse

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_lse: Tensor):
        saved = ctx.saved_tensors
        queries = saved[0]
        lse = saved[1]
        values = saved[2:]
        grad_values, grad_queries = _attnres_phase1_list_backward_triton(
            values,
            queries,
            lse,
            grad_output,
            grad_lse,
            ctx.key_norm,
        )
        return (grad_queries, None, *grad_values)


class _TritonAttentionResidualPhase2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        partial_value: Tensor,
        query: Tensor,
        interblock_output: Tensor,
        interblock_lse: Tensor,
        key_norm: bool,
        normalize_output: bool,
    ):
        normalize_output = bool(normalize_output)
        output, output_inv_rms = _attnres_phase2_merge_triton(
            partial_value,
            query,
            interblock_output,
            interblock_lse,
            bool(key_norm),
            normalize_output,
        )
        ctx.save_for_backward(
            partial_value,
            query,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
        )
        ctx.key_norm = bool(key_norm)
        ctx.normalize_output = normalize_output
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            partial_value,
            query,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
        ) = ctx.saved_tensors
        grad_partial, grad_query, grad_interblock_output, grad_interblock_lse = _attnres_phase2_backward_triton(
            partial_value,
            query,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
            grad_output,
            ctx.key_norm,
            ctx.normalize_output,
        )
        return grad_partial, grad_query, grad_interblock_output, grad_interblock_lse, None, None


class _TritonAttentionResidualPhase2Logits(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        partial_value: Tensor,
        partial_logit: Tensor,
        interblock_output: Tensor,
        interblock_lse: Tensor,
        normalize_output: bool,
    ):
        normalize_output = bool(normalize_output)
        output, output_inv_rms = _attnres_phase2_logits_merge_triton(
            partial_value,
            partial_logit,
            interblock_output,
            interblock_lse,
            normalize_output,
        )
        ctx.save_for_backward(
            partial_value,
            partial_logit,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
        )
        ctx.normalize_output = normalize_output
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            partial_value,
            partial_logit,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
        ) = ctx.saved_tensors
        grad_partial, grad_logit, grad_interblock_output, grad_interblock_lse = _attnres_phase2_logits_backward_triton(
            partial_value,
            partial_logit,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
            grad_output,
            ctx.normalize_output,
        )
        return grad_partial, grad_logit, grad_interblock_output, grad_interblock_lse, None


class _TritonLRIDAttentionResidualPhase2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        partial_value: Tensor,
        partial_key: Tensor,
        query: Tensor,
        interblock_output: Tensor,
        interblock_lse: Tensor,
        logit_scale: float,
        key_norm: bool,
        normalize_output: bool,
    ):
        normalize_output = bool(normalize_output)
        output, output_inv_rms = _lrid_phase2_merge_triton(
            partial_value,
            partial_key,
            query,
            interblock_output,
            interblock_lse,
            float(logit_scale),
            bool(key_norm),
            normalize_output,
        )
        ctx.save_for_backward(
            partial_value,
            partial_key,
            query,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
        )
        ctx.logit_scale = float(logit_scale)
        ctx.key_norm = bool(key_norm)
        ctx.normalize_output = normalize_output
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (
            partial_value,
            partial_key,
            query,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
        ) = ctx.saved_tensors
        (
            grad_partial,
            grad_key,
            grad_query,
            grad_interblock_output,
            grad_interblock_lse,
        ) = _lrid_phase2_backward_triton(
            partial_value,
            partial_key,
            query,
            interblock_output,
            interblock_lse,
            output,
            output_inv_rms,
            grad_output,
            ctx.logit_scale,
            ctx.key_norm,
            ctx.normalize_output,
        )
        return (
            grad_partial,
            grad_key,
            grad_query,
            grad_interblock_output,
            grad_interblock_lse,
            None,
            None,
            None,
        )


def attention_residual_phase1(
    values,
    queries: Tensor,
    key_norm: bool,
    force_triton: bool = False,
):
    if not isinstance(values, Tensor):
        queries = queries.to(values[0].dtype)
        if (
            os.environ.get("ATTNRES_BLOCK_PHASE1_LIST", "0") == "1"
            and
            _TRITON_AVAILABLE
            and len(values) <= 16
            and queries.size(0) <= 16
            and values[0].size(-1) <= 4096
            and _can_use_triton(*values, queries)
        ):
            return _TritonAttentionResidualPhase1List.apply(
                queries,
                bool(key_norm),
                *values,
            )
        if force_triton:
            raise RuntimeError("Triton two-phase AttnRes phase1 is not available for these tensors")
        return _attnres_phase1_read_torch(_as_stacked(values), queries, key_norm)

    values = _as_stacked(values)
    if values.size(0) == 0:
        raise RuntimeError("Two-phase AttnRes phase1 requires at least one source")
    queries = queries.to(values.dtype)
    if (
        _TRITON_AVAILABLE
        and values.is_cuda
        and queries.is_cuda
        and values.is_contiguous()
        and queries.is_contiguous()
        and values.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and queries.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and values.size(0) <= 16
        and queries.size(0) <= 16
        and values.size(-1) <= 4096
    ):
        return _TritonAttentionResidualPhase1.apply(values, queries, bool(key_norm))
    if force_triton:
        raise RuntimeError("Triton two-phase AttnRes phase1 is not available for these tensors")
    return _attnres_phase1_read_torch(values, queries, key_norm)


def attention_residual_phase1_from_logits(
    values,
    logits: Tensor,
    force_triton: bool = False,
    normalize_output: bool = False,
):
    if not isinstance(values, Tensor):
        if len(values) == 0:
            raise RuntimeError("Cached-logit AttnRes phase1 requires at least one source")
        first = values[0]
        list_shapes_match = all(value.shape == first.shape for value in values)
        list_device_dtype_match = all(value.device == first.device and value.dtype == first.dtype for value in values)
        if (
            os.environ.get("ATTNRES_PHASE1_LOGITS_LIST", "0") == "1"
            and
            _TRITON_AVAILABLE
            and list_shapes_match
            and list_device_dtype_match
            and first.is_cuda
            and logits.is_cuda
            and first.dtype in {torch.float16, torch.bfloat16, torch.float32}
            and logits.dtype in {torch.float16, torch.bfloat16, torch.float32}
            and len(values) <= 16
            and logits.size(0) <= 16
            and logits.size(1) == len(values)
            and first.size(-1) <= 4096
        ):
            return _TritonAttentionResidualPhase1LogitsList.apply(logits, bool(normalize_output), *values)
        if force_triton and os.environ.get("ATTNRES_PHASE1_LOGITS_LIST", "0") == "1":
            raise RuntimeError("Triton cached-logit AttnRes phase1 list kernel is not available for these tensors")
    values = _as_stacked(values)
    if values.size(0) == 0:
        raise RuntimeError("Cached-logit AttnRes phase1 requires at least one source")
    if (
        _TRITON_AVAILABLE
        and values.is_cuda
        and logits.is_cuda
        and values.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and logits.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and values.size(0) <= 16
        and logits.size(0) <= 16
        and values.size(-1) <= 4096
    ):
        return _TritonAttentionResidualPhase1Logits.apply(values, logits, bool(normalize_output))
    if force_triton:
        raise RuntimeError("Triton cached-logit AttnRes phase1 is not available for these tensors")
    weights = F.softmax(logits.float(), dim=1).to(values.dtype)
    output = torch.einsum("qsbt,sbtd->qbtd", weights, values)
    if normalize_output:
        output = _norm(output)
    lse = torch.logsumexp(logits.float(), dim=1)
    return output, lse


def attention_residual_phase2(
    partial_value: Tensor,
    query: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    key_norm: bool,
    force_triton: bool = False,
    normalize_output: bool = False,
    logit_bias: float = 0.0,
):
    query = query.to(partial_value.dtype)
    logit_bias = float(logit_bias)
    if logit_bias:
        key = _norm(partial_value) if key_norm else partial_value
        partial_logit = torch.sum(key * query.to(key.dtype), dim=-1).float() + logit_bias
        return attention_residual_phase2_from_logit(
            partial_value,
            partial_logit,
            interblock_output,
            interblock_lse,
            force_triton=force_triton,
            normalize_output=normalize_output,
        )
    if (
        _TRITON_AVAILABLE
        and partial_value.is_cuda
        and query.is_cuda
        and interblock_output.is_cuda
        and interblock_lse.is_cuda
        and partial_value.is_contiguous()
        and query.is_contiguous()
        and interblock_output.is_contiguous()
        and interblock_lse.is_contiguous()
        and partial_value.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and partial_value.size(-1) <= 4096
    ):
        return _TritonAttentionResidualPhase2.apply(
            partial_value,
            query,
            interblock_output,
            interblock_lse,
            bool(key_norm),
            bool(normalize_output),
        )
    if force_triton:
        raise RuntimeError("Triton two-phase AttnRes phase2 is not available for these tensors")
    output = _attnres_phase2_merge_torch(
        partial_value,
        query,
        interblock_output,
        interblock_lse,
        key_norm,
        logit_bias,
    )
    return _norm(output) if normalize_output else output


def attention_residual_phase2_torch(
    partial_value: Tensor,
    query: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    key_norm: bool,
    normalize_output: bool = False,
    logit_bias: float = 0.0,
):
    query = query.to(partial_value.dtype)
    output = _attnres_phase2_merge_torch(
        partial_value,
        query,
        interblock_output,
        interblock_lse,
        key_norm,
        logit_bias,
    )
    return _norm(output) if normalize_output else output


def attention_residual_phase2_from_logit(
    partial_value: Tensor,
    partial_logit: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    force_triton: bool = False,
    normalize_output: bool = False,
):
    if (
        _TRITON_AVAILABLE
        and partial_value.is_cuda
        and partial_logit.is_cuda
        and interblock_output.is_cuda
        and interblock_lse.is_cuda
        and partial_value.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and partial_logit.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and partial_value.size(-1) <= 4096
    ):
        return _TritonAttentionResidualPhase2Logits.apply(
            partial_value,
            partial_logit,
            interblock_output,
            interblock_lse,
            bool(normalize_output),
        )
    if force_triton:
        raise RuntimeError("Triton cached-logit AttnRes phase2 is not available for these tensors")
    prob = torch.sigmoid(partial_logit.float() - interblock_lse.float()).to(partial_value.dtype)
    output = torch.addcmul(interblock_output, partial_value - interblock_output, prob.unsqueeze(-1))
    return _norm(output) if normalize_output else output


def lrid_attention_residual_phase2(
    partial_value: Tensor,
    partial_key: Tensor,
    query: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    logit_scale: float,
    key_norm: bool,
    force_triton: bool = False,
    normalize_output: bool = False,
    logit_bias: float = 0.0,
):
    query = query.reshape(-1).to(partial_key.dtype)
    partial_key = partial_key.reshape(partial_key.size(0), partial_key.size(1), -1)
    logit_bias = float(logit_bias)
    if logit_bias:
        key = _norm(partial_key.float()).to(partial_value.dtype) if key_norm else partial_key
        logit = torch.sum(key * query.to(key.dtype).view(1, 1, -1), dim=-1) * float(logit_scale)
        logit = logit.float() + logit_bias
        return attention_residual_phase2_from_logit(
            partial_value,
            logit,
            interblock_output,
            interblock_lse,
            force_triton=force_triton,
            normalize_output=normalize_output,
        )
    if (
        _TRITON_AVAILABLE
        and partial_value.is_cuda
        and partial_key.is_cuda
        and query.is_cuda
        and interblock_output.is_cuda
        and interblock_lse.is_cuda
        and partial_value.is_contiguous()
        and partial_key.is_contiguous()
        and query.is_contiguous()
        and interblock_output.is_contiguous()
        and interblock_lse.is_contiguous()
        and partial_value.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and partial_key.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and partial_value.size(-1) <= 4096
        and partial_key.size(-1) <= 1024
    ):
        return _TritonLRIDAttentionResidualPhase2.apply(
            partial_value,
            partial_key,
            query,
            interblock_output,
            interblock_lse,
            float(logit_scale),
            bool(key_norm),
            bool(normalize_output),
        )
    if force_triton:
        raise RuntimeError("Triton LRID cached-logit phase2 is not available for these tensors")
    key = _norm(partial_key.float()).to(partial_value.dtype) if key_norm else partial_key
    logit = torch.sum(key * query.to(key.dtype).view(1, 1, -1), dim=-1) * float(logit_scale)
    return attention_residual_phase2_from_logit(
        partial_value,
        logit,
        interblock_output,
        interblock_lse,
        normalize_output=normalize_output,
    )


def lrid_attention_residual_phase2_torch(
    partial_value: Tensor,
    partial_key: Tensor,
    query: Tensor,
    interblock_output: Tensor,
    interblock_lse: Tensor,
    logit_scale: float,
    key_norm: bool,
    normalize_output: bool = False,
    logit_bias: float = 0.0,
):
    query = query.reshape(-1).to(partial_key.dtype)
    partial_key = partial_key.reshape(partial_key.size(0), partial_key.size(1), -1)
    key = _norm(partial_key.float()).to(partial_value.dtype) if key_norm else partial_key
    logit = torch.sum(key * query.to(key.dtype).view(1, 1, -1), dim=-1) * float(logit_scale)
    if logit_bias:
        logit = logit.float() + float(logit_bias)
    prob = torch.sigmoid(logit.float() - interblock_lse.float()).to(partial_value.dtype)
    output = torch.lerp(interblock_output, partial_value, prob.unsqueeze(-1))
    return _norm(output) if normalize_output else output


def _lrid_read_torch_stacked(
    values: Tensor,
    keys: Tensor,
    query: Tensor,
    num_heads: int,
    logit_scale: float,
    key_norm: bool,
    source_counts=None,
) -> Tensor:
    num_heads = int(num_heads)
    key_head_dim = keys.size(-1) // num_heads
    value_head_dim = values.size(-1) // num_heads

    keys = keys.reshape(*keys.shape[:-1], num_heads, key_head_dim)
    values = values.reshape(*values.shape[:-1], num_heads, value_head_dim)
    if key_norm:
        keys = _norm(keys.float()).to(values.dtype)

    query = query.to(keys.dtype)
    if query.dim() == 4:
        logits = torch.einsum("sbthr,bthr->sbth", keys, query) * logit_scale
    else:
        logits = torch.einsum("sbthr,hr->sbth", keys, query) * logit_scale
    log_count_bias = _source_log_count_bias(source_counts, values.size(0), logits.device)
    if log_count_bias is not None:
        logits = logits + log_count_bias.view(-1, 1, 1, 1)
    weights = F.softmax(logits.float(), dim=0).to(values.dtype)
    output = torch.einsum("sbth,sbthd->bthd", weights, values)
    return output.reshape(output.size(0), output.size(1), num_heads * value_head_dim)


def _lrid_read_torch_list(
    values,
    keys,
    query: Tensor,
    num_heads: int,
    logit_scale: float,
    key_norm: bool,
    source_counts=None,
) -> Tensor:
    num_heads = int(num_heads)
    B, T, D = values[0].shape
    key_head_dim = keys[0].size(-1) // num_heads
    value_head_dim = D // num_heads

    query = query.to(keys[0].dtype)
    log_count_bias = _source_log_count_bias(source_counts, len(values), values[0].device)
    value_views = []
    logits = []
    for idx, (value, key) in enumerate(zip(values, keys)):
        key = key.reshape(B, T, num_heads, key_head_dim)
        if key_norm:
            key = _norm(key.float()).to(values[0].dtype)
        value = value.reshape(B, T, num_heads, value_head_dim)
        value_views.append(value)
        if query.dim() == 4:
            logit = torch.sum(key * query, dim=-1) * logit_scale
        else:
            logit = torch.sum(key * query.view(1, 1, num_heads, key_head_dim), dim=-1) * logit_scale
        if log_count_bias is not None:
            logit = logit + log_count_bias[idx]
        logits.append(logit)

    weights = F.softmax(torch.stack(logits, dim=0).float(), dim=0).to(values[0].dtype)
    output = torch.zeros_like(value_views[0])
    for idx, value in enumerate(value_views):
        output = output + weights[idx].unsqueeze(-1) * value
    return output.reshape(B, T, D)


def lrid_attention_residual_read_torch(
    values,
    keys,
    query: Tensor,
    num_heads: int,
    logit_scale: float,
    key_norm: bool,
    source_counts=None,
) -> Tensor:
    values = _as_stacked(values)
    keys = _as_stacked(keys)
    return _lrid_read_torch_stacked(values, keys, query, num_heads, logit_scale, key_norm, source_counts)



def _can_use_triton_lrid_list(values, keys, query: Tensor) -> bool:
    if not _TRITON_AVAILABLE or isinstance(values, Tensor) or isinstance(keys, Tensor):
        return False
    if len(values) != len(keys) or len(values) < 2 or len(values) > 16:
        return False
    allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    tensors = (*values, *keys)
    return (
        isinstance(query, Tensor)
        and query.is_cuda
        and query.is_contiguous()
        and query.dtype in allowed_dtypes
        and all(
            isinstance(t, Tensor)
            and t.is_cuda
            and t.dim() == 3
            and t.stride(-1) == 1
            and t.stride(0) == t.size(1) * t.stride(1)
            and t.dtype in allowed_dtypes
            for t in tensors
        )
    )


def _row_strides(tensors):
    return [int(t.stride(1)) for t in tensors]


def _lrid_read_list_backward_triton(
    values,
    keys,
    query: Tensor,
    grad_output: Tensor,
    num_heads: int,
    logit_scale: float,
    key_norm: bool,
    normalize_output: bool = False,
    normed_output: Tensor = None,
    output_inv_rms: Tensor = None,
    alpha_saved: Tensor = None,
    key_inv_rms_saved: Tensor = None,
):
    S = len(values)
    B, T, D = values[0].shape
    R = keys[0].size(-1)
    H = int(num_heads)
    n_tokens = B * T
    value_head_dim = D // H
    key_head_dim = R // H

    if value_head_dim > 4096 or key_head_dim > 256:
        raise RuntimeError("Fused LRID AttnRes dimensions exceed the Triton kernel limits")

    query = query.contiguous()
    grad_output = grad_output.contiguous()
    if normalize_output:
        normed_output = normed_output.reshape(n_tokens, H, value_head_dim).contiguous()
    else:
        normed_output = grad_output.reshape(n_tokens, H, value_head_dim)
        output_inv_rms = torch.empty((1,), device=values[0].device, dtype=torch.float32)
    use_aux = alpha_saved is not None and key_inv_rms_saved is not None
    if use_aux:
        alpha_saved = alpha_saved.contiguous()
        key_inv_rms_saved = key_inv_rms_saved.contiguous()
    else:
        alpha_saved = torch.empty((1,), device=values[0].device, dtype=torch.float32)
        key_inv_rms_saved = torch.empty((1,), device=values[0].device, dtype=torch.float32)
    grad_values = [torch.empty_like(value, memory_format=torch.contiguous_format) for value in values]
    grad_keys = [torch.empty_like(key, memory_format=torch.contiguous_format) for key in keys]
    grad_query_accum = torch.zeros(query.shape, device=query.device, dtype=torch.float32)
    s_block = _next_power_of_2(S)
    block_d = _next_power_of_2(value_head_dim)
    block_k = _next_power_of_2(key_head_dim)
    num_warps = 8 if block_d >= 2048 else 4
    grid = (n_tokens, H)

    padded_values = list(values) + [values[0]] * (16 - S)
    padded_keys = list(keys) + [keys[0]] * (16 - S)
    padded_value_strides = _row_strides(padded_values)
    padded_key_strides = _row_strides(padded_keys)
    padded_grad_values = grad_values + [grad_values[0]] * (16 - S)
    padded_grad_keys = grad_keys + [grad_keys[0]] * (16 - S)
    _lrid_attnres_list_backward_kernel[grid](
        *padded_values,
        *padded_keys,
        *padded_value_strides,
        *padded_key_strides,
        query,
        grad_output,
        normed_output,
        output_inv_rms,
        alpha_saved,
        key_inv_rms_saved,
        *padded_grad_values,
        *padded_grad_keys,
        grad_query_accum,
        n_tokens,
        S,
        float(logit_scale),
        H,
        value_head_dim,
        key_head_dim,
        s_block,
        block_d,
        block_k,
        key_norm,
        normalize_output,
        use_aux,
        torch.finfo(values[0].dtype).eps,
        num_warps=num_warps,
    )
    return grad_values, grad_keys, grad_query_accum.to(dtype=query.dtype)


class _TritonLRIDListAttentionResidualRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, num_heads: int, logit_scale: float, key_norm: bool, normalize_output: bool, *sources):
        if len(sources) % 2 != 0:
            raise RuntimeError("LRID list read expects value sources followed by key sources")
        split = len(sources) // 2
        values = sources[:split]
        keys = sources[split:]
        if normalize_output:
            output, output_inv_rms, alpha_saved, key_inv_rms_saved = _lrid_read_list_triton(
                values,
                keys,
                query,
                int(num_heads),
                float(logit_scale),
                bool(key_norm),
                normalize_output=True,
                return_aux=True,
            )
            ctx.save_for_backward(query, output, output_inv_rms, alpha_saved, key_inv_rms_saved, *values, *keys)
        else:
            output, alpha_saved, key_inv_rms_saved = _lrid_read_list_triton(
                values,
                keys,
                query,
                int(num_heads),
                float(logit_scale),
                bool(key_norm),
                False,
                return_aux=True,
            )
            ctx.save_for_backward(query, alpha_saved, key_inv_rms_saved, *values, *keys)
        ctx.num_heads = int(num_heads)
        ctx.logit_scale = float(logit_scale)
        ctx.key_norm = bool(key_norm)
        ctx.normalize_output = bool(normalize_output)
        ctx.num_sources = split
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        saved = ctx.saved_tensors
        query = saved[0]
        if ctx.normalize_output:
            normed_output = saved[1]
            output_inv_rms = saved[2]
            alpha_saved = saved[3]
            key_inv_rms_saved = saved[4]
            values = saved[5:5 + ctx.num_sources]
            keys = saved[5 + ctx.num_sources:]
        else:
            normed_output = None
            output_inv_rms = None
            alpha_saved = saved[1]
            key_inv_rms_saved = saved[2]
            values = saved[3:3 + ctx.num_sources]
            keys = saved[3 + ctx.num_sources:]
        grad_values, grad_keys, grad_query = _lrid_read_list_backward_triton(
            values,
            keys,
            query,
            grad_output,
            ctx.num_heads,
            ctx.logit_scale,
            ctx.key_norm,
            ctx.normalize_output,
            normed_output,
            output_inv_rms,
            alpha_saved,
            key_inv_rms_saved,
        )
        return (grad_query, None, None, None, None, *grad_values, *grad_keys)


def _lrid_read_list_triton(
    values,
    keys,
    query: Tensor,
    num_heads: int,
    logit_scale: float,
    key_norm: bool,
    normalize_output: bool = False,
    return_inv_rms: bool = False,
    return_aux: bool = False,
):
    S = len(values)
    B, T, D = values[0].shape
    R = keys[0].size(-1)
    H = int(num_heads)
    n_tokens = B * T
    value_head_dim = D // H
    key_head_dim = R // H
    query_dynamic = query.dim() == 4

    if value_head_dim > 4096 or key_head_dim > 256:
        raise RuntimeError("Fused LRID AttnRes dimensions exceed the Triton kernel limits")

    query = query.contiguous()
    if query_dynamic:
        query = query.reshape(n_tokens, H, key_head_dim).contiguous()

    output = torch.empty((n_tokens, H, value_head_dim), device=values[0].device, dtype=values[0].dtype)
    output_inv_rms = (
        torch.empty((n_tokens, H), device=values[0].device, dtype=torch.float32)
        if normalize_output
        else torch.empty((1,), device=values[0].device, dtype=torch.float32)
    )
    alpha_saved = (
        torch.empty((n_tokens, H, S), device=values[0].device, dtype=torch.float32)
        if return_aux
        else torch.empty((1,), device=values[0].device, dtype=torch.float32)
    )
    key_inv_rms_saved = (
        torch.empty((n_tokens, H, S), device=values[0].device, dtype=torch.float32)
        if return_aux
        else torch.empty((1,), device=values[0].device, dtype=torch.float32)
    )
    s_block = _next_power_of_2(S)
    block_k = _next_power_of_2(key_head_dim)
    if value_head_dim <= 128:
        block_m = 16
        block_d = _next_power_of_2(value_head_dim)
    else:
        block_m = 1
        block_d = _next_power_of_2(value_head_dim)
    num_warps = 8 if block_d >= 2048 else 4
    grid = (
        triton.cdiv(n_tokens, block_m),
        H,
        triton.cdiv(value_head_dim, block_d),
    )

    padded_values = list(values) + [values[0]] * (16 - S)
    padded_keys = list(keys) + [keys[0]] * (16 - S)
    padded_value_strides = _row_strides(padded_values)
    padded_key_strides = _row_strides(padded_keys)
    _lrid_attnres_list_read_kernel[grid](
        *padded_values,
        *padded_keys,
        *padded_value_strides,
        *padded_key_strides,
        query,
        output,
        output_inv_rms,
        alpha_saved,
        key_inv_rms_saved,
        n_tokens,
        S,
        float(logit_scale),
        H,
        value_head_dim,
        key_head_dim,
        s_block,
        block_m,
        block_d,
        block_k,
        key_norm,
        query_dynamic,
        normalize_output,
        return_aux,
        torch.finfo(values[0].dtype).eps,
        num_warps=num_warps,
    )
    output = output.reshape(B, T, D)
    if return_aux:
        if return_inv_rms or normalize_output:
            return output, output_inv_rms, alpha_saved, key_inv_rms_saved
        return output, alpha_saved, key_inv_rms_saved
    if return_inv_rms:
        return output, output_inv_rms
    return output


if _TRITON_AVAILABLE and hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton"):
    _triton_op = torch.library.triton_op
    _wrap_triton = torch.library.wrap_triton

    @_triton_op("attnres::lrid_read_list", mutates_args={})
    def _lrid_read_list_library_op(
        values: list[Tensor],
        keys: list[Tensor],
        query: Tensor,
        num_heads: int,
        logit_scale: float,
        key_norm: bool,
        normalize_output: bool,
        save_aux: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        S = len(values)
        B, T, D = values[0].shape
        R = keys[0].size(-1)
        H = int(num_heads)
        n_tokens = B * T
        value_head_dim = D // H
        key_head_dim = R // H
        query_dynamic = query.dim() == 4

        query = query.contiguous()
        if query_dynamic:
            query = query.reshape(n_tokens, H, key_head_dim).contiguous()

        output = torch.empty((n_tokens, H, value_head_dim), device=values[0].device, dtype=values[0].dtype)
        output_inv_rms = (
            torch.empty((n_tokens, H), device=values[0].device, dtype=torch.float32)
            if normalize_output
            else torch.empty((1,), device=values[0].device, dtype=torch.float32)
        )
        alpha_saved = (
            torch.empty((n_tokens, H, S), device=values[0].device, dtype=torch.float32)
            if save_aux
            else torch.empty((1,), device=values[0].device, dtype=torch.float32)
        )
        key_inv_rms_saved = (
            torch.empty((n_tokens, H, S), device=values[0].device, dtype=torch.float32)
            if save_aux
            else torch.empty((1,), device=values[0].device, dtype=torch.float32)
        )
        s_block = _next_power_of_2(S)
        block_k = _next_power_of_2(key_head_dim)
        if value_head_dim <= 128:
            block_m = 16
            block_d = _next_power_of_2(value_head_dim)
        else:
            block_m = 1
            block_d = _next_power_of_2(value_head_dim)
        num_warps = 8 if block_d >= 2048 else 4
        grid = (
            triton.cdiv(n_tokens, block_m),
            H,
            triton.cdiv(value_head_dim, block_d),
        )

        padded_values = list(values) + [values[0]] * (16 - S)
        padded_keys = list(keys) + [keys[0]] * (16 - S)
        padded_value_strides = _row_strides(padded_values)
        padded_key_strides = _row_strides(padded_keys)
        _wrap_triton(_lrid_attnres_list_read_kernel)[grid](
            *padded_values,
            *padded_keys,
            *padded_value_strides,
            *padded_key_strides,
            query,
            output,
            output_inv_rms,
            alpha_saved,
            key_inv_rms_saved,
            n_tokens,
            S,
            float(logit_scale),
            H,
            value_head_dim,
            key_head_dim,
            s_block,
            block_m,
            block_d,
            block_k,
            key_norm,
            query_dynamic,
            normalize_output,
            save_aux,
            torch.finfo(values[0].dtype).eps,
            num_warps=num_warps,
        )
        return output.reshape(B, T, D), output_inv_rms, alpha_saved, key_inv_rms_saved


    @_triton_op("attnres::lrid_read_list_backward", mutates_args={})
    def _lrid_read_list_backward_library_op(
        values: list[Tensor],
        keys: list[Tensor],
        query: Tensor,
        grad_output: Tensor,
        normed_output: Tensor,
        output_inv_rms: Tensor,
        alpha_saved: Tensor,
        key_inv_rms_saved: Tensor,
        num_heads: int,
        logit_scale: float,
        key_norm: bool,
        normalize_output: bool,
        use_aux: bool,
    ) -> tuple[list[Tensor], list[Tensor], Tensor]:
        S = len(values)
        B, T, D = values[0].shape
        R = keys[0].size(-1)
        H = int(num_heads)
        n_tokens = B * T
        value_head_dim = D // H
        key_head_dim = R // H

        query = query.contiguous()
        grad_output = grad_output.contiguous()
        if normalize_output:
            normed_output = normed_output.reshape(n_tokens, H, value_head_dim).contiguous()
        else:
            normed_output = grad_output.reshape(n_tokens, H, value_head_dim)
            output_inv_rms = torch.empty((1,), device=values[0].device, dtype=torch.float32)
        alpha_saved = alpha_saved.contiguous()
        key_inv_rms_saved = key_inv_rms_saved.contiguous()
        grad_values = [torch.empty_like(value, memory_format=torch.contiguous_format) for value in values]
        grad_keys = [torch.empty_like(key, memory_format=torch.contiguous_format) for key in keys]
        grad_query_accum = torch.zeros(query.shape, device=query.device, dtype=torch.float32)
        s_block = _next_power_of_2(S)
        block_d = _next_power_of_2(value_head_dim)
        block_k = _next_power_of_2(key_head_dim)
        num_warps = 8 if block_d >= 2048 else 4
        grid = (n_tokens, H)

        padded_values = list(values) + [values[0]] * (16 - S)
        padded_keys = list(keys) + [keys[0]] * (16 - S)
        padded_value_strides = _row_strides(padded_values)
        padded_key_strides = _row_strides(padded_keys)
        padded_grad_values = grad_values + [grad_values[0]] * (16 - S)
        padded_grad_keys = grad_keys + [grad_keys[0]] * (16 - S)
        _wrap_triton(_lrid_attnres_list_backward_kernel)[grid](
            *padded_values,
            *padded_keys,
            *padded_value_strides,
            *padded_key_strides,
            query,
            grad_output,
            normed_output,
            output_inv_rms,
            alpha_saved,
            key_inv_rms_saved,
            *padded_grad_values,
            *padded_grad_keys,
            grad_query_accum,
            n_tokens,
            S,
            float(logit_scale),
            H,
            value_head_dim,
            key_head_dim,
            s_block,
            block_d,
            block_k,
            key_norm,
            normalize_output,
            use_aux,
            torch.finfo(values[0].dtype).eps,
            num_warps=num_warps,
        )
        return grad_values, grad_keys, grad_query_accum.to(dtype=query.dtype)


    def _lrid_read_list_library_setup_context(ctx, inputs, output):
        values, keys, query, num_heads, logit_scale, key_norm, normalize_output, save_aux = inputs
        normed_output, output_inv_rms, alpha_saved, key_inv_rms_saved = output
        ctx.save_for_backward(query, normed_output, output_inv_rms, alpha_saved, key_inv_rms_saved, *values, *keys)
        ctx.num_sources = len(values)
        ctx.num_heads = int(num_heads)
        ctx.logit_scale = float(logit_scale)
        ctx.key_norm = bool(key_norm)
        ctx.normalize_output = bool(normalize_output)
        ctx.use_aux = bool(save_aux)


    def _lrid_read_list_library_backward(ctx, grad_output, grad_output_inv_rms, grad_alpha_saved, grad_key_inv_rms_saved):
        saved = ctx.saved_tensors
        query = saved[0]
        normed_output = saved[1]
        output_inv_rms = saved[2]
        alpha_saved = saved[3]
        key_inv_rms_saved = saved[4]
        values = list(saved[5:5 + ctx.num_sources])
        keys = list(saved[5 + ctx.num_sources:])
        grad_values, grad_keys, grad_query = _lrid_read_list_backward_library_op(
            values,
            keys,
            query,
            grad_output,
            normed_output,
            output_inv_rms,
            alpha_saved,
            key_inv_rms_saved,
            ctx.num_heads,
            ctx.logit_scale,
            ctx.key_norm,
            ctx.normalize_output,
            ctx.use_aux,
        )
        return grad_values, grad_keys, grad_query, None, None, None, None, None


    _lrid_read_list_library_op.register_autograd(
        _lrid_read_list_library_backward,
        setup_context=_lrid_read_list_library_setup_context,
    )
else:
    _lrid_read_list_library_op = None


def lrid_attention_residual_read(
    values,
    keys,
    query: Tensor,
    num_heads: int,
    logit_scale: float,
    key_norm: bool,
    force_triton: bool = False,
    normalize_output: bool = False,
    source_counts=None,
) -> Tensor:
    if isinstance(values, Tensor):
        first_value = values[0]
        key_dtype = keys.dtype
    else:
        first_value = values[0]
        key_dtype = keys[0].dtype
    query = query.to(key_dtype)

    use_fused_output_norm = normalize_output and int(num_heads) == 1
    if source_counts is None and _can_use_triton_lrid_list(values, keys, query):
        if torch.is_grad_enabled():
            training_kernel_mode = _training_kernel_mode()
            if training_kernel_mode in {"auto", "triton_op"} and _lrid_read_list_library_op is not None and query.dim() != 4:
                output, _, _, _ = _lrid_read_list_library_op(
                    list(values),
                    list(keys),
                    query,
                    int(num_heads),
                    float(logit_scale),
                    bool(key_norm),
                    bool(use_fused_output_norm),
                    os.environ.get("ATTNRES_SAVE_AUX", "0") == "1",
                )
                return output if use_fused_output_norm or not normalize_output else _norm(output)
            if training_kernel_mode == "triton" and query.dim() != 4:
                return _TritonLRIDListAttentionResidualRead.apply(
                    query,
                    int(num_heads),
                    float(logit_scale),
                    bool(key_norm),
                    bool(use_fused_output_norm),
                    *values,
                    *keys,
                )
            else:
                output = _lrid_read_torch_list(
                    values,
                    keys,
                    query,
                    num_heads,
                    logit_scale,
                    key_norm,
                )
            return _norm(output) if normalize_output else output
        if not torch.is_grad_enabled():
            output = _lrid_read_list_triton(
                values,
                keys,
                query,
                num_heads,
                logit_scale,
                key_norm,
                use_fused_output_norm,
            )
            return output if use_fused_output_norm or not normalize_output else _norm(output)

    if not isinstance(values, Tensor) and not isinstance(keys, Tensor):
        if len(values) == 1:
            output = values[0]
            return _norm(output) if normalize_output else output
        if force_triton:
            raise RuntimeError("Triton fused LRID AttnRes path is not available for these tensors")
        output = _lrid_read_torch_list(
            values,
            keys,
            query,
            num_heads,
            logit_scale,
            key_norm,
            source_counts,
        )
        return _norm(output) if normalize_output else output

    values = _as_stacked(values)
    keys = _as_stacked(keys)
    if values.size(0) == 1:
        return _norm(first_value) if normalize_output else first_value
    if force_triton:
        raise RuntimeError("Triton fused LRID AttnRes path is not available for these tensors")
    output = _lrid_read_torch_stacked(values, keys, query, num_heads, logit_scale, key_norm, source_counts)
    return _norm(output) if normalize_output else output
