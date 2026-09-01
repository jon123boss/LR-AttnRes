import math

import pytest
import torch

import fast_attnres as fast_module
from fast_attnres import (
    FastAttnResPackageError,
    FastAttnResRuntimeError,
    assess_fast_attnres_inputs,
    fast_attnres_read,
)
from model import ModelConfig, OBPM


def _oracle(values, query, *, eps, scale):
    values_f32 = torch.stack(tuple(values), dim=0).float()
    query_f32 = query.float()
    tail = values_f32[..., -query.numel():]
    key = tail * torch.rsqrt(tail.square().mean(dim=-1, keepdim=True) + eps)
    logits = (key * query_f32).sum(dim=-1) * scale
    weights = torch.softmax(logits, dim=0)
    return (weights.unsqueeze(-1) * values_f32).sum(dim=0).to(values[0].dtype)


def test_fast_read_matches_independent_oracle_and_value_query_gradients():
    torch.manual_seed(101)
    batch, tokens, width, rank = 2, 3, 9, 4
    # Strided aliases exercise the public source-list adapter without turning
    # the test into a contiguous-only implementation check.
    bases_ref = [torch.randn(batch, tokens, width * 2, requires_grad=True) for _ in range(2)]
    bases_fast = [base.detach().clone().requires_grad_(True) for base in bases_ref]
    values_ref = [bases_ref[0][..., ::2], bases_ref[1][..., ::2], bases_ref[0][..., ::2]]
    values_fast = [bases_fast[0][..., ::2], bases_fast[1][..., ::2], bases_fast[0][..., ::2]]
    query_ref = torch.randn(rank, requires_grad=True)
    query_fast = query_ref.detach().clone().requires_grad_(True)
    eps = torch.finfo(torch.float32).eps
    scale = 0.37

    expected = _oracle(values_ref, query_ref, eps=eps, scale=scale)
    actual, decision = fast_attnres_read(
        values_fast,
        query_fast,
        eps=eps,
        scale=scale,
    )

    assert decision.eligible
    assert actual is not None
    assert not all(value.is_contiguous() for value in values_fast)
    assert values_fast[0].data_ptr() == values_fast[2].data_ptr()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    assert torch.allclose(query_fast.grad, query_ref.grad, atol=1e-6, rtol=1e-6)
    for base_fast, base_ref in zip(bases_fast, bases_ref):
        assert torch.allclose(base_fast.grad, base_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"key_norm": False}, "key_normalization_disabled"),
        ({"source_counts": [1, 2]}, "source_counts_or_logit_biases"),
        ({"source_logit_biases": [0.0, 0.0]}, "source_counts_or_logit_biases"),
    ],
)
def test_fast_input_semantic_fallbacks_are_structured(kwargs, reason):
    values = [torch.randn(2, 3, 8) for _ in range(2)]
    decision = assess_fast_attnres_inputs(values, torch.randn(8), **kwargs)
    assert not decision.eligible
    assert decision.path == "legacy"
    assert decision.reason == reason
    assert decision.as_dict()["reason"] == reason


def test_fast_input_dtype_shape_and_single_source_fallbacks():
    assert assess_fast_attnres_inputs(
        [torch.randn(2, 3, 8, dtype=torch.float16) for _ in range(2)],
        torch.randn(8),
    ).reason == "unsupported_dtype"
    assert assess_fast_attnres_inputs(
        [torch.randn(2, 3, 8) for _ in range(2)],
        torch.randn(1, 8),
    ).reason == "unsupported_query_shape"
    assert assess_fast_attnres_inputs(
        [torch.randn(2, 3, 8)],
        torch.randn(8),
    ).reason == "single_source_noop"
    too_many = [torch.randn(1, 1, 1) for _ in range(130)]
    assert assess_fast_attnres_inputs(too_many, torch.randn(1)).reason == "unsupported_source_count"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"use_lrid": True, "lrid_rank": 4}, "projected_or_non_tail_lrid_key"),
        (
            {"use_lrid": True, "lrid_rank": 4, "lrid_key_from_output_tail": True, "lrid_num_heads": 2},
            "multi_head_lrid",
        ),
        (
            {
                "use_lrid": True,
                "lrid_rank": 4,
                "lrid_key_from_output_tail": True,
                "lrid_input_dependent_query": True,
            },
            "dynamic_lrid_query",
        ),
    ],
)
def test_fast_model_semantic_fallback_reasons(kwargs, reason):
    config = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=16,
        block_size=4,
        use_attnres=True,
        attnres_backend="fast",
        attnres_type="full",
        attnres_key_norm=True,
        flash_attention=False,
        **kwargs,
    )
    model = OBPM(config).eval()
    with torch.no_grad():
        model(torch.randint(config.vocab_size, (1, config.block_size)), return_hidden=True)
    report = model.fast_attnres_route_report()
    assert report["fast_reads"] == 0
    assert report["fallback_reasons"][reason] == 2


def test_fast_full_and_tail_lr_model_routes_match_legacy():
    common = dict(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=16,
        block_size=4,
        use_attnres=True,
        attnres_type="full",
        attnres_key_norm=True,
        attn_res_query_norm=True,
        flash_attention=False,
        norm_pos="before",
    )
    for lr_kwargs in ({}, {"use_lrid": True, "lrid_rank": 4, "lrid_key_from_output_tail": True}):
        torch.manual_seed(103)
        legacy = OBPM(ModelConfig(**common, **lr_kwargs, attnres_backend="legacy")).train()
        fast = OBPM(ModelConfig(**common, **lr_kwargs, attnres_backend="fast")).train()
        fast.load_state_dict(legacy.state_dict())
        idx = torch.randint(common["vocab_size"], (2, common["block_size"]))
        expected = legacy(idx, return_hidden=True)
        actual = fast(idx, return_hidden=True)
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
        assert fast.fast_attnres_route_report()["fast_reads"] == 4


def test_fast_block_routes_only_reads_without_count_prior_bias():
    config = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=16,
        block_size=4,
        use_attnres=True,
        attnres_backend="fast",
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_count_prior=True,
        attnres_key_norm=True,
        flash_attention=False,
    )
    model = OBPM(config).eval()
    with torch.no_grad():
        model(torch.randint(config.vocab_size, (1, config.block_size)), return_hidden=True)
    report = model.fast_attnres_route_report()
    assert report["fast_reads"] == 0
    assert report["legacy_reads"] == 4
    assert report["fallback_reasons"] == {"source_count_prior": 4}


def test_fast_block_tail_uses_exact_key_payload_for_transformed_summary():
    common = dict(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=16,
        block_size=4,
        use_attnres=True,
        use_lrid=True,
        lrid_rank=4,
        lrid_key_from_output_tail=True,
        attnres_backend="fast",
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_count_prior=False,
        norm_pos="before",
        flash_attention=False,
    )
    averaged = OBPM(ModelConfig(**common, attnres_block_average=True))
    report = averaged.fast_attnres_route_report()
    assert report["fast_reads"] == 4
    assert averaged._fast_lrid_requires_key_payload()

    compatible = OBPM(ModelConfig(**common, attnres_block_average=False))
    assert compatible.fast_attnres_route_report()["fast_reads"] == 4
    assert not compatible._fast_lrid_requires_key_payload()


def test_model_config_defaults_to_legacy_backend():
    config = ModelConfig()
    assert config.attnres_backend == "legacy"
    with pytest.raises(ValueError, match="attnres_backend"):
        ModelConfig(attnres_backend="other")


def test_fast_package_and_kernel_failures_are_not_silent(monkeypatch):
    values = [torch.randn(1, 2, 4) for _ in range(2)]
    query = torch.randn(4)

    def missing():
        raise FastAttnResPackageError("missing")

    monkeypatch.setattr(fast_module, "load_fast_attnres", missing)
    with pytest.raises(FastAttnResPackageError):
        fast_attnres_read(values, query, eps=math.ulp(1.0), scale=1.0)

    def kernel(_values, _query, *, eps, scale):
        raise RuntimeError("kernel launch failed")

    monkeypatch.setattr(fast_module, "load_fast_attnres", lambda: kernel)
    with pytest.raises(FastAttnResRuntimeError):
        fast_attnres_read(values, query, eps=math.ulp(1.0), scale=1.0)


@pytest.mark.parametrize(
    ("attnres_type", "norm_pos", "value_norm", "learned_scale"),
    [
        ("full", "before", False, False),
        ("full", "after", False, False),
        ("full", "both", False, False),
        ("block", "before", False, False),
        ("block", "before", True, False),
        ("block", "after", True, False),
        ("block", "before", False, True),
    ],
)
def test_fast_output_tail_matches_all_parameter_gradients_and_adamw_update(
    attnres_type, norm_pos, value_norm, learned_scale
):
    common = dict(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=19,
        block_size=4,
        use_attnres=True,
        use_lrid=True,
        lrid_rank=4,
        lrid_key_from_output_tail=True,
        attnres_type=attnres_type,
        attnres_num_blocks=2,
        attnres_block_average=True,
        attnres_block_count_prior=False,
        attnres_block_value_norm=value_norm,
        attnres_block_learned_scale=learned_scale,
        attnres_block_learned_scale_init="one",
        attnres_training_cache_phase1=False,
        norm_pos=norm_pos,
        flash_attention=False,
    )
    torch.manual_seed(20260901)
    legacy = OBPM(ModelConfig(**common, attnres_backend="legacy")).train()
    fast = OBPM(ModelConfig(**common, attnres_backend="fast")).train()
    fast.load_state_dict(legacy.state_dict())
    if learned_scale:
        with torch.no_grad():
            legacy.transformer.attnres_block_scales.fill_(-0.5)
            fast.transformer.attnres_block_scales.fill_(-0.5)
    legacy_optimizer = torch.optim.AdamW(legacy.parameters(), lr=3e-4)
    fast_optimizer = torch.optim.AdamW(fast.parameters(), lr=3e-4)
    tokens = torch.randint(common["vocab_size"], (2, common["block_size"]))
    upstream = torch.randn(2, common["block_size"], common["n_embd"])

    expected = legacy(tokens, return_hidden=True)
    actual = fast(tokens, return_hidden=True)
    assert fast.fast_attnres_route_report()["active_reads"] == 4
    assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-4)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()

    legacy_parameters = dict(legacy.named_parameters())
    fast_parameters = dict(fast.named_parameters())
    assert legacy_parameters.keys() == fast_parameters.keys()
    for name in legacy_parameters:
        expected_grad = legacy_parameters[name].grad
        actual_grad = fast_parameters[name].grad
        assert (expected_grad is None) == (actual_grad is None), name
        if expected_grad is not None:
            assert torch.allclose(actual_grad, expected_grad, rtol=1e-3, atol=1e-4), name

    legacy_optimizer.step()
    fast_optimizer.step()
    for name in legacy_parameters:
        assert torch.allclose(
            fast_parameters[name], legacy_parameters[name], rtol=1e-3, atol=1e-4
        ), name


def test_fast_model_uses_the_legacy_normalization_epsilons():
    assert OBPM._fast_attnres_eps(torch.bfloat16) == torch.finfo(torch.float32).eps
    assert OBPM._fast_attnres_eps(torch.float32) == torch.finfo(torch.float32).eps
    assert OBPM._fast_attnres_eps(torch.bfloat16, lrid=True) == torch.finfo(torch.float32).eps
