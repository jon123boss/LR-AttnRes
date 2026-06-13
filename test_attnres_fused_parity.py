#!/usr/bin/env python3
"""
Standalone fused AttnRes parity and backward-compatibility checks.

Run on a CUDA/Triton machine to exercise the fused kernels:
    python test_attnres_fused_parity.py

Useful narrower runs:
    python test_attnres_fused_parity.py --mode direct
    python test_attnres_fused_parity.py --mode model
    python test_attnres_fused_parity.py --mode compat

The script also runs on CPU, but direct kernel checks fall back to the PyTorch
path when CUDA/Triton is unavailable. Use the printed fused_available line to
confirm whether the actual Triton kernels were exercised.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Iterable

import torch

from attnres_ops import (
    attention_residual_read,
    attention_residual_read_torch,
    is_fused_attnres_available,
    lrid_attention_residual_read,
    lrid_attention_residual_read_torch,
)
from model import ModelConfig, OBPM


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def parse_args():
    parser = argparse.ArgumentParser(description="Check fused AttnRes parity and compatibility.")
    parser.add_argument("--mode", choices=("all", "direct", "model", "compat"), default="all")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--force-triton", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def tolerances(dtype: torch.dtype, atol: float | None, rtol: float | None):
    if atol is not None and rtol is not None:
        return atol, rtol
    if dtype == torch.float32:
        return 3e-4 if atol is None else atol, 3e-4 if rtol is None else rtol
    return 1e-1 if atol is None else atol, 1e-1 if rtol is None else rtol


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    diff = (actual_f - expected_f).abs()
    max_err = float(diff.max().item()) if diff.numel() else 0.0
    mean_err = float(diff.mean().item()) if diff.numel() else 0.0
    if not torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} mismatch: max_err={max_err:.6g}, mean_err={mean_err:.6g}, "
            f"atol={atol}, rtol={rtol}"
        )
    print(f"PASS {name}: max_err={max_err:.6g}, mean_err={mean_err:.6g}")


def clone_leaf_list(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.detach().clone().requires_grad_(True) for tensor in tensors]


def direct_base_check(device: torch.device, dtype: torch.dtype, force_triton: bool, atol: float, rtol: float) -> None:
    torch.manual_seed(10)
    values_ref = [torch.randn(2, 5, 64, device=device, dtype=dtype, requires_grad=True) for _ in range(5)]
    query_ref = torch.randn(64, device=device, dtype=dtype, requires_grad=True)
    values_fused = clone_leaf_list(values_ref)
    query_fused = query_ref.detach().clone().requires_grad_(True)

    expected = attention_residual_read_torch(values_ref, query_ref, key_norm=True)
    actual = attention_residual_read(values_fused, query_fused, True, force_triton=force_triton)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    if device.type == "cuda":
        torch.cuda.synchronize()

    assert_close("base forward", actual, expected, atol, rtol)
    for idx, (actual_value, expected_value) in enumerate(zip(values_fused, values_ref)):
        assert_close(f"base grad value[{idx}]", actual_value.grad, expected_value.grad, atol, rtol)
    assert_close("base grad query", query_fused.grad, query_ref.grad, atol, rtol)


def direct_lrid_check(device: torch.device, dtype: torch.dtype, force_triton: bool, atol: float, rtol: float) -> None:
    torch.manual_seed(20)
    num_heads = 2
    rank = 32
    n_embd = 128
    values_ref = [torch.randn(2, 6, n_embd, device=device, dtype=dtype, requires_grad=True) for _ in range(6)]
    keys_ref = [torch.randn(2, 6, rank, device=device, dtype=dtype, requires_grad=True) for _ in range(6)]
    query_ref = torch.randn(num_heads, rank // num_heads, device=device, dtype=dtype, requires_grad=True)
    values_fused = clone_leaf_list(values_ref)
    keys_fused = clone_leaf_list(keys_ref)
    query_fused = query_ref.detach().clone().requires_grad_(True)

    expected = lrid_attention_residual_read_torch(values_ref, keys_ref, query_ref, num_heads, 0.25, True)
    actual = lrid_attention_residual_read(
        values_fused,
        keys_fused,
        query_fused,
        num_heads,
        0.25,
        True,
        force_triton=force_triton,
    )
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    if device.type == "cuda":
        torch.cuda.synchronize()

    assert_close("lrid forward", actual, expected, atol, rtol)
    for idx, (actual_value, expected_value) in enumerate(zip(values_fused, values_ref)):
        assert_close(f"lrid grad value[{idx}]", actual_value.grad, expected_value.grad, atol, rtol)
    for idx, (actual_key, expected_key) in enumerate(zip(keys_fused, keys_ref)):
        assert_close(f"lrid grad key[{idx}]", actual_key.grad, expected_key.grad, atol, rtol)
    assert_close("lrid grad query", query_fused.grad, query_ref.grad, atol, rtol)


def direct_lrid_output_tail_check(
    device: torch.device,
    dtype: torch.dtype,
    force_triton: bool,
    atol: float,
    rtol: float,
) -> None:
    torch.manual_seed(25)
    num_heads = 2
    rank = 32
    n_embd = 128
    values_ref = [torch.randn(2, 6, n_embd, device=device, dtype=dtype, requires_grad=True) for _ in range(6)]
    values_fused = clone_leaf_list(values_ref)
    query_ref = torch.randn(num_heads, rank // num_heads, device=device, dtype=dtype, requires_grad=True)
    query_fused = query_ref.detach().clone().requires_grad_(True)
    keys_ref = [value[..., -rank:].contiguous() for value in values_ref]
    keys_fused = [value[..., -rank:].contiguous() for value in values_fused]

    expected = lrid_attention_residual_read_torch(values_ref, keys_ref, query_ref, num_heads, 0.25, True)
    actual = lrid_attention_residual_read(
        values_fused,
        keys_fused,
        query_fused,
        num_heads,
        0.25,
        True,
        force_triton=force_triton,
    )
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    if device.type == "cuda":
        torch.cuda.synchronize()

    assert_close("lrid output-tail forward", actual, expected, atol, rtol)
    for idx, (actual_value, expected_value) in enumerate(zip(values_fused, values_ref)):
        assert_close(f"lrid output-tail grad value[{idx}]", actual_value.grad, expected_value.grad, atol, rtol)
    assert_close("lrid output-tail grad query", query_fused.grad, query_ref.grad, atol, rtol)


def common_model_kwargs(
    use_lrid: bool,
    attnres_type: str,
    attnres_block_average_mode: str = "count",
    attnres_block_learned_scale: bool = False,
    attnres_block_learned_scale_init: str = "count",
    attnres_block_value_norm: bool = False,
    attnres_block_alpha: str = "legacy",
    attnres_block_beta: str = "legacy",
    attnres_block_alpha_learned: bool = False,
    attnres_block_beta_learned: bool = False,
    attnres_block_alpha_scope: str = "shared",
    attnres_block_beta_scope: str = "shared",
    attnres_block_split_sublayers: bool = False,
) -> dict:
    attnres_block_count_prior = (
        attnres_type == "block"
        and not attnres_block_learned_scale
        and not attnres_block_value_norm
    )
    return dict(
        n_layer=3,
        n_head=4,
        n_embd=64,
        mlp_hidden_dim=128,
        vocab_size=128,
        block_size=12,
        flash_attention=False,
        norm_pos="before",
        use_attnres=True,
        attnres_type=attnres_type,
        attnres_num_blocks=2,
        attnres_block_average=True,
        attnres_block_average_mode=attnres_block_average_mode,
        attnres_block_count_prior=attnres_block_count_prior,
        attnres_block_alpha=attnres_block_alpha,
        attnres_block_beta=attnres_block_beta,
        attnres_block_alpha_learned=attnres_block_alpha_learned,
        attnres_block_beta_learned=attnres_block_beta_learned,
        attnres_block_alpha_scope=attnres_block_alpha_scope,
        attnres_block_beta_scope=attnres_block_beta_scope,
        attnres_block_split_sublayers=attnres_block_split_sublayers,
        attnres_block_learned_scale=attnres_block_learned_scale,
        attnres_block_learned_scale_init=attnres_block_learned_scale_init,
        attnres_block_value_norm=attnres_block_value_norm,
        attnres_key_norm=True,
        attn_res_query_init="normal",
        use_lrid=use_lrid,
        lrid_rank=32,
        lrid_num_heads=1,
        lrid_use_logit_scale=True,
    )


def model_parity_check(
    device: torch.device,
    dtype: torch.dtype,
    use_lrid: bool,
    attnres_type: str,
    lrid_key_from_output_tail: bool,
    attnres_block_average_mode: str,
    attnres_block_learned_scale: bool,
    attnres_block_learned_scale_init: str,
    attnres_block_value_norm: bool,
    atol: float,
    rtol: float,
    attnres_block_alpha: str = "legacy",
    attnres_block_beta: str = "legacy",
    attnres_block_alpha_learned: bool = False,
    attnres_block_beta_learned: bool = False,
    attnres_block_alpha_scope: str = "shared",
    attnres_block_beta_scope: str = "shared",
    attnres_block_split_sublayers: bool = False,
) -> None:
    torch.manual_seed(
        30
        + int(use_lrid)
        + (0 if attnres_type == "block" else 10)
        + (100 if lrid_key_from_output_tail else 0)
        + (1000 if attnres_block_average_mode == "sqrt" else 0)
        + (2000 if attnres_block_learned_scale else 0)
        + (3000 if attnres_block_learned_scale_init == "sqrt" else 0)
        + (4000 if attnres_block_learned_scale_init == "one" else 0)
        + (5000 if attnres_block_value_norm else 0)
        + (6000 if attnres_block_alpha != "legacy" else 0)
        + (7000 if attnres_block_beta != "legacy" else 0)
        + (8000 if attnres_block_alpha_learned else 0)
        + (9000 if attnres_block_beta_learned else 0)
        + (10000 if attnres_block_alpha_scope == "per_residual" else 0)
        + (11000 if attnres_block_alpha_scope == "per_block" else 0)
        + (12000 if attnres_block_split_sublayers else 0)
    )
    kwargs = common_model_kwargs(
        use_lrid,
        attnres_type,
        attnres_block_average_mode,
        attnres_block_learned_scale,
        attnres_block_learned_scale_init,
        attnres_block_value_norm,
        attnres_block_alpha,
        attnres_block_beta,
        attnres_block_alpha_learned,
        attnres_block_beta_learned,
        attnres_block_alpha_scope,
        attnres_block_beta_scope,
        attnres_block_split_sublayers,
    )
    if use_lrid:
        kwargs["lrid_key_from_output_tail"] = lrid_key_from_output_tail
    ref = OBPM(ModelConfig(**kwargs, use_fused_attnres=False)).to(device=device, dtype=dtype).train()
    fused = OBPM(ModelConfig(**kwargs, use_fused_attnres=True)).to(device=device, dtype=dtype).train()
    fused.load_state_dict(ref.state_dict(), strict=True)
    idx = torch.randint(0, kwargs["vocab_size"], (2, kwargs["block_size"]), device=device)

    expected = ref(idx, return_hidden=True)
    actual = fused(idx, return_hidden=True)
    ref_loss = expected.float().square().mean()
    fused_loss = actual.float().square().mean()
    ref_loss.backward()
    fused_loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    prefix = (
        f"model use_lrid={use_lrid} tail_key={lrid_key_from_output_tail} "
        f"type={attnres_type} block_avg_mode={attnres_block_average_mode} "
        f"learned_scale={attnres_block_learned_scale} learned_init={attnres_block_learned_scale_init} "
        f"value_norm={attnres_block_value_norm} alpha={attnres_block_alpha} beta={attnres_block_beta} "
        f"alpha_learned={attnres_block_alpha_learned} beta_learned={attnres_block_beta_learned} "
        f"alpha_scope={attnres_block_alpha_scope} beta_scope={attnres_block_beta_scope} "
        f"split_sublayers={attnres_block_split_sublayers}"
    )
    assert_close(f"{prefix} forward", actual, expected, atol, rtol)
    assert_close(f"{prefix} embedding grad", fused.transformer.wte.weight.grad, ref.transformer.wte.weight.grad, atol, rtol)


def compatibility_check(device: torch.device) -> None:
    old_base_args = dict(
        n_layer=2,
        n_head=2,
        n_embd=32,
        mlp_hidden_dim=64,
        vocab_size=96,
        block_size=8,
        flash_attention=False,
        norm_pos="before",
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_average=True,
        attnres_key_norm=True,
        use_lrid=False,
    )
    old_lrid_args = dict(old_base_args, use_lrid=True, lrid_rank=16, lrid_num_heads=1)

    for name, old_args in [("base", old_base_args), ("lrid", old_lrid_args)]:
        cfg = ModelConfig(**old_args)
        assert cfg.use_fused_attnres is False
        assert cfg.lrid_key_from_output_tail is False
        assert cfg.attnres_block_average_mode == "count"
        assert cfg.attnres_block_count_prior is True
        assert cfg.attnres_block_alpha == "legacy"
        assert cfg.attnres_block_beta == "legacy"
        assert cfg.attnres_block_alpha_learned is False
        assert cfg.attnres_block_beta_learned is False
        assert cfg.attnres_block_alpha_scope == "shared"
        assert cfg.attnres_block_beta_scope == "shared"
        assert cfg.attnres_block_split_sublayers is False
        assert cfg.attnres_block_learned_scale is False
        assert cfg.attnres_block_learned_scale_init == "count"
        assert cfg.attnres_block_value_norm is False
        if cfg.use_lrid:
            assert cfg.lrid_projection_rank == cfg.lrid_rank

        model = OBPM(cfg).to(device)
        state = model.state_dict()
        reloaded = OBPM(ModelConfig(**old_args)).to(device)
        reloaded.load_state_dict(state, strict=True)
        idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size), device=device)
        with torch.no_grad():
            expected = model(idx, return_hidden=True)
            actual = reloaded(idx, return_hidden=True)
        assert_close(f"compat old {name} strict load", actual, expected, 0.0, 0.0)

        old_checkpoint_model_args = dict(old_args)
        cfg_from_old_checkpoint = ModelConfig(**old_checkpoint_model_args)
        assert "use_fused_attnres" not in old_checkpoint_model_args
        assert "lrid_key_from_output_tail" not in old_checkpoint_model_args
        assert "attnres_block_average_mode" not in old_checkpoint_model_args
        assert "attnres_block_count_prior" not in old_checkpoint_model_args
        assert "attnres_block_alpha" not in old_checkpoint_model_args
        assert "attnres_block_beta" not in old_checkpoint_model_args
        assert "attnres_block_alpha_learned" not in old_checkpoint_model_args
        assert "attnres_block_beta_learned" not in old_checkpoint_model_args
        assert "attnres_block_alpha_scope" not in old_checkpoint_model_args
        assert "attnres_block_beta_scope" not in old_checkpoint_model_args
        assert "attnres_block_split_sublayers" not in old_checkpoint_model_args
        assert "attnres_block_learned_scale" not in old_checkpoint_model_args
        assert "attnres_block_learned_scale_init" not in old_checkpoint_model_args
        assert "attnres_block_value_norm" not in old_checkpoint_model_args
        assert cfg_from_old_checkpoint.use_fused_attnres is False
        assert cfg_from_old_checkpoint.lrid_key_from_output_tail is False
        assert cfg_from_old_checkpoint.attnres_block_average_mode == "count"
        assert cfg_from_old_checkpoint.attnres_block_count_prior is True
        assert cfg_from_old_checkpoint.attnres_block_alpha == "legacy"
        assert cfg_from_old_checkpoint.attnres_block_beta == "legacy"
        assert cfg_from_old_checkpoint.attnres_block_alpha_learned is False
        assert cfg_from_old_checkpoint.attnres_block_beta_learned is False
        assert cfg_from_old_checkpoint.attnres_block_alpha_scope == "shared"
        assert cfg_from_old_checkpoint.attnres_block_beta_scope == "shared"
        assert cfg_from_old_checkpoint.attnres_block_split_sublayers is False
        assert cfg_from_old_checkpoint.attnres_block_learned_scale is False
        assert cfg_from_old_checkpoint.attnres_block_learned_scale_init == "count"
        assert cfg_from_old_checkpoint.attnres_block_value_norm is False

        roundtrip_args = asdict(cfg_from_old_checkpoint)
        cfg_from_new_checkpoint = ModelConfig(**roundtrip_args)
        assert cfg_from_new_checkpoint.use_fused_attnres is False
        assert cfg_from_new_checkpoint.lrid_key_from_output_tail is False
        assert cfg_from_new_checkpoint.attnres_block_average_mode == "count"
        assert cfg_from_new_checkpoint.attnres_block_count_prior is True
        assert cfg_from_new_checkpoint.attnres_block_alpha == "legacy"
        assert cfg_from_new_checkpoint.attnres_block_beta == "legacy"
        assert cfg_from_new_checkpoint.attnres_block_alpha_learned is False
        assert cfg_from_new_checkpoint.attnres_block_beta_learned is False
        assert cfg_from_new_checkpoint.attnres_block_alpha_scope == "shared"
        assert cfg_from_new_checkpoint.attnres_block_beta_scope == "shared"
        assert cfg_from_new_checkpoint.attnres_block_split_sublayers is False
        assert cfg_from_new_checkpoint.attnres_block_learned_scale is False
        assert cfg_from_new_checkpoint.attnres_block_learned_scale_init == "count"
        assert cfg_from_new_checkpoint.attnres_block_value_norm is False
        print(f"PASS compat old {name} model_args defaults")

    tail_cfg = ModelConfig(**old_lrid_args, lrid_key_from_output_tail=True)
    tail_model = OBPM(tail_cfg).to(device)
    attn_proj = tail_model.transformer.layers[0].attn.c_proj
    mlp_proj = tail_model.transformer.layers[0].mlp.fc2
    assert attn_proj.proj.out_features == tail_cfg.n_embd
    assert mlp_proj.proj.out_features == tail_cfg.n_embd
    idx = torch.randint(0, tail_cfg.vocab_size, (1, tail_cfg.block_size), device=device)
    with torch.no_grad():
        tail_model(idx, return_hidden=True)
    print("PASS compat lrid_key_from_output_tail has no learned key-projection rows")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested, but this CUDA device does not support bf16")

    fused_available = is_fused_attnres_available() and device.type == "cuda"
    force_triton = bool(args.force_triton and fused_available)
    atol, rtol = tolerances(dtype, args.atol, args.rtol)
    print(
        f"device={device} dtype={dtype} fused_available={fused_available} "
        f"force_triton={force_triton} atol={atol} rtol={rtol}"
    )

    if args.mode in {"all", "direct"}:
        direct_base_check(device, dtype, force_triton, atol, rtol)
        direct_lrid_check(device, dtype, force_triton, atol, rtol)
        direct_lrid_output_tail_check(device, dtype, force_triton, atol, rtol)

    if args.mode in {"all", "model"}:
        for use_lrid in [False, True]:
            for attnres_type in ["block", "full"]:
                tail_key_options = [False, True] if use_lrid else [False]
                for lrid_key_from_output_tail in tail_key_options:
                    for attnres_block_average_mode in ["count", "sqrt"]:
                        model_parity_check(
                            device,
                            dtype,
                            use_lrid,
                            attnres_type,
                            lrid_key_from_output_tail,
                            attnres_block_average_mode,
                            False,
                            "count",
                            False,
                            atol,
                            rtol,
                        )
                    if attnres_type == "block":
                        alpha_beta_cases = [
                            ("0.75", "0.5", False, False, "shared", "shared"),
                            ("0.75", "0.5", True, True, "shared", "shared"),
                            ("0.1,0.2,0.3,0.4,0.5,0.6", "0.6,0.5,0.4,0.3,0.2,0.1", True, True, "per_residual", "per_residual"),
                            ("0.5,1.0", "0.25,0.75", True, True, "per_block", "per_block"),
                        ]
                        for (
                            attnres_block_alpha,
                            attnres_block_beta,
                            attnres_block_alpha_learned,
                            attnres_block_beta_learned,
                            attnres_block_alpha_scope,
                            attnres_block_beta_scope,
                        ) in alpha_beta_cases:
                            model_parity_check(
                                device,
                                dtype,
                                use_lrid,
                                attnres_type,
                                lrid_key_from_output_tail,
                                "count",
                                False,
                                "count",
                                False,
                                atol,
                                rtol,
                                attnres_block_alpha=attnres_block_alpha,
                                attnres_block_beta=attnres_block_beta,
                                attnres_block_alpha_learned=attnres_block_alpha_learned,
                                attnres_block_beta_learned=attnres_block_beta_learned,
                                attnres_block_alpha_scope=attnres_block_alpha_scope,
                                attnres_block_beta_scope=attnres_block_beta_scope,
                            )
                        for attnres_block_learned_scale_init in ["count", "sqrt", "one"]:
                            model_parity_check(
                                device,
                                dtype,
                                use_lrid,
                                attnres_type,
                                lrid_key_from_output_tail,
                                "count",
                                True,
                                attnres_block_learned_scale_init,
                                False,
                                atol,
                                rtol,
                            )
                        model_parity_check(
                            device,
                            dtype,
                            use_lrid,
                            attnres_type,
                            lrid_key_from_output_tail,
                            "count",
                            False,
                            "count",
                            True,
                            atol,
                            rtol,
                        )
                        model_parity_check(
                            device,
                            dtype,
                            use_lrid,
                            attnres_type,
                            lrid_key_from_output_tail,
                            "count",
                            False,
                            "count",
                            False,
                            atol,
                            rtol,
                            attnres_block_split_sublayers=True,
                        )

    if args.mode in {"all", "compat"}:
        compatibility_check(device)

    print("All requested checks passed.")


if __name__ == "__main__":
    main()
