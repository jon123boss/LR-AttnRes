import math

import pytest
import torch

import model as model_module
from attnres_ops import (
    _lrid_list_dims_within_triton_limits,
    attention_residual_phase1_from_logits,
    attention_residual_phase2,
    attention_residual_phase2_torch,
    attention_residual_read,
    attention_residual_read_torch,
    is_fused_attnres_available,
    lrid_attention_residual_read,
    lrid_attention_residual_read_torch,
)
from model import ModelConfig, OBPM


@pytest.mark.parametrize("mode,denominator", [("count", 4.0), ("sqrt", 2.0)])
def test_attnres_block_average_mode_scales_block_source(mode, denominator):
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_block_average=True,
        attnres_block_average_mode=mode,
        attnres_block_count_prior=True,
        attnres_key_norm=False,
        use_lrid=True,
        lrid_rank=4,
        lrid_num_heads=1,
    )
    model = OBPM(cfg)
    value = torch.full((1, 2, cfg.n_embd), 8.0)
    key = torch.full((1, 2, cfg.lrid_rank), 4.0)

    assert torch.allclose(model._attnres_block_summary(value, 4), value / denominator)
    lrid_value, lrid_key = model._lrid_block_source(value, key, count=4)
    assert torch.allclose(lrid_value, value / denominator)
    assert torch.allclose(lrid_key, key / denominator)


@pytest.mark.parametrize(
    "init,expected",
    [
        ("count", [1.0, 0.5, 1.0, 0.5]),
        ("sqrt", [1.0, 2.0 ** -0.5, 1.0, 2.0 ** -0.5]),
        ("one", [1.0, 1.0, 1.0, 1.0]),
        ("1/c", [1.0, 0.5, 1.0, 0.5]),
        ("1/sqrtc", [1.0, 2.0 ** -0.5, 1.0, 2.0 ** -0.5]),
        ("1", [1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_attnres_learned_block_scale_init_and_usage(init, expected):
    cfg = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_average=True,
        attnres_block_average_mode="count",
        attnres_block_count_prior=False,
        attnres_block_learned_scale=True,
        attnres_block_learned_scale_init=init,
        attnres_key_norm=False,
        use_lrid=True,
        lrid_rank=4,
        lrid_num_heads=1,
    )
    model = OBPM(cfg)
    expected = torch.tensor(expected)
    assert torch.allclose(model.transformer.attnres_block_scales.detach(), expected)

    value = torch.full((1, 2, cfg.n_embd), 8.0)
    key = torch.full((1, 2, cfg.lrid_rank), 4.0)
    scale = expected[1]
    assert torch.allclose(model._attnres_block_summary(value, 2, summary_idx=2), value * scale)
    lrid_value, lrid_key = model._lrid_block_source(value, key, count=2, summary_idx=2)
    assert torch.allclose(lrid_value, value * scale)
    assert torch.allclose(lrid_key, key * scale)


def test_attnres_block_value_norm_overrides_scaling():
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_block_average=True,
        attnres_block_average_mode="count",
        attnres_block_count_prior=False,
        attnres_block_value_norm=True,
        attnres_key_norm=False,
        use_lrid=True,
        lrid_rank=4,
        lrid_num_heads=1,
    )
    model = OBPM(cfg)
    value = torch.arange(1, 17, dtype=torch.float32).reshape(1, 2, cfg.n_embd)
    key = torch.full((1, 2, cfg.lrid_rank), 4.0)

    block_source = model._attnres_block_summary(value, 4, summary_idx=1)
    rms = block_source.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(block_source, value / 4)

    lrid_value, lrid_key = model._lrid_block_source(value, key, count=4, summary_idx=1)
    lrid_rms = lrid_value.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(lrid_rms, torch.ones_like(lrid_rms), atol=1e-6, rtol=1e-6)
    assert torch.equal(lrid_key, key)


@pytest.mark.parametrize(
    "scope,expected_shape",
    [("shared", (1,)), ("per_residual", (4,)), ("per_block", (2,))],
)
def test_attnres_block_alpha_beta_learned_parameter_shapes(scope, expected_shape):
    cfg = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_alpha_learned=True,
        attnres_block_beta_learned=True,
        attnres_block_alpha_scope=scope,
        attnres_block_beta_scope=scope,
    )
    model = OBPM(cfg)

    assert tuple(model.transformer.attnres_block_alphas.shape) == expected_shape
    assert tuple(model.transformer.attnres_block_betas.shape) == expected_shape


def test_attnres_block_per_block_scope_uses_configured_num_blocks():
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=4,
        attnres_block_alpha="0.25,0.5,0.75,1.0",
        attnres_block_beta="0.0,0.1,0.2,0.3",
        attnres_block_alpha_learned=True,
        attnres_block_beta_learned=True,
        attnres_block_alpha_scope="per_block",
        attnres_block_beta_scope="per_block",
    )
    model = OBPM(cfg)

    assert tuple(model.transformer.attnres_block_alphas.shape) == (4,)
    assert tuple(model.transformer.attnres_block_betas.shape) == (4,)
    assert math.isclose(float(model.transformer.attnres_block_alphas[0]), 0.25)
    assert math.isclose(float(model.transformer.attnres_block_betas[0]), 0.0)


def test_attnres_block_power_logging_values_use_live_learned_parameters():
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_alpha="0.25,0.5",
        attnres_block_beta="0.75,1.0",
        attnres_block_alpha_learned=True,
        attnres_block_beta_learned=True,
        attnres_block_alpha_scope="per_block",
        attnres_block_beta_scope="per_block",
    )
    model = OBPM(cfg)
    with torch.no_grad():
        model.transformer.attnres_block_alphas.add_(1.0)
        model.transformer.attnres_block_betas.sub_(0.25)

    assert torch.allclose(
        model.attnres_block_power_values_for_logging("alpha"),
        torch.tensor([1.25, 1.5]),
    )
    assert torch.allclose(
        model.attnres_block_power_values_for_logging("beta"),
        torch.tensor([0.5, 0.75]),
    )


def test_learned_alpha_beta_stay_fp32_after_mixed_precision():
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_block_alpha_learned=True,
        attnres_block_beta_learned=True,
    )
    model = OBPM(cfg).to_mixed_precision(dtype=torch.bfloat16)

    assert model.transformer.wte.weight.dtype == torch.bfloat16
    assert model.transformer.attnres_block_alphas.dtype == torch.float32
    assert model.transformer.attnres_block_betas.dtype == torch.float32


def test_attnres_block_split_sublayers_forward_and_fused_read_match():
    torch.manual_seed(41)
    common = dict(
        n_layer=3,
        n_head=2,
        n_embd=16,
        mlp_hidden_dim=32,
        vocab_size=64,
        block_size=8,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=1,
        attnres_block_average=True,
        attnres_block_count_prior=True,
        attnres_block_split_sublayers=True,
        attnres_key_norm=True,
        flash_attention=False,
        norm_pos="before",
    )
    ref = OBPM(ModelConfig(**common, use_fused_attnres=False)).eval()
    fused = OBPM(ModelConfig(**common, use_fused_attnres=True)).eval()
    fused.load_state_dict(ref.state_dict())
    idx = torch.randint(0, common["vocab_size"], (2, common["block_size"]))

    with torch.no_grad():
        expected = ref(idx, return_hidden=True)
        actual = fused(idx, return_hidden=True)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_attnres_block_split_sublayers_keeps_separate_source_counts(monkeypatch):
    cfg = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        use_fused_attnres=False,
        attnres_type="block",
        attnres_num_blocks=1,
        attnres_block_average=True,
        attnres_block_count_prior=True,
        attnres_block_split_sublayers=True,
        norm_pos="before",
    )
    model = OBPM(cfg).eval()
    captured = []
    original = OBPM._apply_attnres

    def wrapped(self, residual_idx, sources, *args, **kwargs):
        biases = kwargs.get("source_logit_biases")
        captured.append(
            (
                residual_idx,
                len(sources),
                None if biases is None else [float(bias) for bias in biases],
            )
        )
        return original(self, residual_idx, sources, *args, **kwargs)

    monkeypatch.setattr(OBPM, "_apply_attnres", wrapped)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    with torch.no_grad():
        model(idx, return_hidden=True)

    assert [(idx, n_sources) for idx, n_sources, _ in captured] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 3),
        (4, 3),
    ]
    assert captured[3][2] == pytest.approx([0.0, math.log(2.0), 0.0])
    assert captured[4][2] == pytest.approx([0.0, math.log(2.0), math.log(2.0)])


@pytest.mark.parametrize("input_dependent_query", [False, True])
def test_attnres_block_split_sublayers_lrid_forward_and_fused_read_match(input_dependent_query):
    torch.manual_seed(43)
    common = dict(
        n_layer=3,
        n_head=2,
        n_embd=16,
        mlp_hidden_dim=32,
        vocab_size=64,
        block_size=8,
        use_attnres=True,
        use_lrid=True,
        lrid_rank=4,
        lrid_input_dependent_query=input_dependent_query,
        lrid_key_from_output_tail=True,
        attnres_type="block",
        attnres_num_blocks=1,
        attnres_block_average=True,
        attnres_block_count_prior=True,
        attnres_block_split_sublayers=True,
        attnres_key_norm=True,
        flash_attention=False,
        norm_pos="before",
    )
    ref = OBPM(ModelConfig(**common, use_fused_attnres=False)).eval()
    fused = OBPM(ModelConfig(**common, use_fused_attnres=True)).eval()
    fused.load_state_dict(ref.state_dict())
    idx = torch.randint(0, common["vocab_size"], (2, common["block_size"]))

    with torch.no_grad():
        expected = ref(idx, return_hidden=True)
        actual = fused(idx, return_hidden=True)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_attnres_block_alpha_fixed_scalar_and_per_block_values():
    cfg = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_alpha="1.0,0.5",
        attnres_block_alpha_scope="per_block",
        attnres_block_count_prior=False,
        attnres_key_norm=False,
    )
    model = OBPM(cfg)
    value = torch.full((1, 2, cfg.n_embd), 8.0)

    assert torch.allclose(model._attnres_block_summary(value, 4, summary_idx=1), value / 4.0)
    assert torch.allclose(model._attnres_block_summary(value, 4, summary_idx=3), value / 2.0)


def test_attnres_block_beta_fixed_per_residual_and_per_block_values():
    residual_cfg = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_beta="0.0,0.5,1.0,2.0",
        attnres_block_beta_scope="per_residual",
    )
    residual_model = OBPM(residual_cfg)
    assert math.isclose(
        residual_model._attnres_block_count_logit_bias(4, read_idx=3, source_summary_idx=1),
        math.log(4.0),
    )

    block_cfg = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_beta="0.25,0.75",
        attnres_block_beta_scope="per_block",
    )
    block_model = OBPM(block_cfg)
    assert math.isclose(
        block_model._attnres_block_count_logit_bias(4, read_idx=1, source_summary_idx=3),
        0.75 * math.log(4.0),
    )


def test_attnres_block_power_list_length_validation():
    with pytest.raises(ValueError, match="attnres_block_alpha list length must be 4"):
        ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=8,
            mlp_hidden_dim=16,
            vocab_size=32,
            block_size=4,
            use_attnres=True,
            attnres_type="block",
            attnres_block_alpha="0.25,0.5",
            attnres_block_alpha_scope="per_residual",
        )


def test_source_logit_biases_read_keeps_bias_gradients():
    torch.manual_seed(23)
    values = [torch.randn(2, 3, 5, requires_grad=True) for _ in range(3)]
    query = torch.randn(5, requires_grad=True)
    source_logit_biases = torch.tensor([0.0, 0.7, -0.2], requires_grad=True)

    output = attention_residual_read(
        values,
        query,
        key_norm=True,
        source_logit_biases=source_logit_biases,
    )
    output.float().square().mean().backward()

    assert source_logit_biases.grad is not None
    assert source_logit_biases.grad.abs().sum() > 0


def test_learned_alpha_beta_get_gradients_in_cached_block_path():
    torch.manual_seed(29)
    cfg = ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        use_fused_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_alpha="0.75",
        attnres_block_beta="0.5",
        attnres_block_alpha_learned=True,
        attnres_block_beta_learned=True,
        attn_res_query_init="normal",
        attnres_training_cache_phase1=True,
        attnres_key_norm=False,
    )
    model = OBPM(cfg).train()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))

    output = model(idx, return_hidden=True)
    output.float().square().mean().backward()

    assert model.transformer.attnres_block_alphas.grad is not None
    assert model.transformer.attnres_block_alphas.grad.abs().sum() > 0
    assert model.transformer.attnres_block_betas.grad is not None
    assert model.transformer.attnres_block_betas.grad.abs().sum() > 0


def test_attnres_source_count_prior_matches_zero_query_uniform_average():
    embedding = torch.full((1, 2, 4), 1.0)
    block_mean = torch.full((1, 2, 4), 7.0)
    query = torch.zeros(4)

    actual = attention_residual_read(
        [embedding, block_mean],
        query,
        key_norm=False,
        source_counts=[1, 3],
    )
    expected = (embedding + 3 * block_mean) / 4

    assert torch.allclose(actual, expected)


def test_lrid_source_count_prior_matches_zero_query_uniform_average():
    embedding = torch.full((1, 2, 8), 2.0)
    block_mean = torch.full((1, 2, 8), 10.0)
    keys = [torch.zeros(1, 2, 4), torch.zeros(1, 2, 4)]
    query = torch.zeros(1, 4)

    actual = lrid_attention_residual_read(
        [embedding, block_mean],
        keys,
        query,
        num_heads=1,
        logit_scale=1.0,
        key_norm=False,
        source_counts=[1, 3],
    )
    expected = (embedding + 3 * block_mean) / 4

    assert torch.allclose(actual, expected)


def test_lrid_triton_dimension_limit_rejects_r512_single_head():
    values = [torch.empty(1, 1, 1024), torch.empty(1, 1, 1024)]
    keys = [torch.empty(1, 1, 512), torch.empty(1, 1, 512)]

    assert not _lrid_list_dims_within_triton_limits(values, keys, num_heads=1)
    assert _lrid_list_dims_within_triton_limits(values, keys, num_heads=2)


def test_attnres_source_count_prior_read_matches_torch_gradients():
    torch.manual_seed(17)
    counts = [1, 3, 2]
    values_ref = [torch.randn(2, 3, 5, requires_grad=True) for _ in counts]
    query_ref = torch.randn(5, requires_grad=True)
    values_actual = [value.detach().clone().requires_grad_(True) for value in values_ref]
    query_actual = query_ref.detach().clone().requires_grad_(True)

    expected = attention_residual_read_torch(values_ref, query_ref, key_norm=True, source_counts=counts)
    actual = attention_residual_read(values_actual, query_actual, key_norm=True, source_counts=counts)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    for actual_value, expected_value in zip(values_actual, values_ref):
        assert torch.allclose(actual_value.grad, expected_value.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(query_actual.grad, query_ref.grad, atol=1e-6, rtol=1e-6)


def test_lrid_source_count_prior_read_matches_torch_gradients():
    torch.manual_seed(19)
    counts = [1, 3, 2]
    num_heads = 2
    values_ref = [torch.randn(2, 3, 8, requires_grad=True) for _ in counts]
    keys_ref = [torch.randn(2, 3, 6, requires_grad=True) for _ in counts]
    query_ref = torch.randn(num_heads, 3, requires_grad=True)
    values_actual = [value.detach().clone().requires_grad_(True) for value in values_ref]
    keys_actual = [key.detach().clone().requires_grad_(True) for key in keys_ref]
    query_actual = query_ref.detach().clone().requires_grad_(True)

    expected = lrid_attention_residual_read_torch(
        values_ref,
        keys_ref,
        query_ref,
        num_heads=num_heads,
        logit_scale=0.5,
        key_norm=True,
        source_counts=counts,
    )
    actual = lrid_attention_residual_read(
        values_actual,
        keys_actual,
        query_actual,
        num_heads=num_heads,
        logit_scale=0.5,
        key_norm=True,
        source_counts=counts,
    )
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    for actual_value, expected_value in zip(values_actual, values_ref):
        assert torch.allclose(actual_value.grad, expected_value.grad, atol=1e-6, rtol=1e-6)
    for actual_key, expected_key in zip(keys_actual, keys_ref):
        assert torch.allclose(actual_key.grad, expected_key.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(query_actual.grad, query_ref.grad, atol=1e-6, rtol=1e-6)


def test_attnres_cached_phase2_applies_partial_count_prior():
    torch.manual_seed(11)
    values = [torch.randn(2, 3, 6) for _ in range(2)]
    partial = torch.randn(2, 3, 6)
    query = torch.randn(6)
    counts = [1, 4]
    partial_count = 2

    logits = []
    for value, count in zip(values, counts):
        logits.append(torch.sum(value * query, dim=-1) + torch.log(torch.tensor(float(count))))
    logits = torch.stack(logits, dim=0).unsqueeze(0)
    phase_output, phase_lse = attention_residual_phase1_from_logits(values, logits)
    actual = attention_residual_phase2_torch(
        partial,
        query,
        phase_output[0],
        phase_lse[0],
        key_norm=False,
        logit_bias=torch.log(torch.tensor(float(partial_count))).item(),
    )
    expected = attention_residual_read_torch(
        values + [partial],
        query,
        key_norm=False,
        source_counts=counts + [partial_count],
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"attnres_block_learned_scale": True},
        {"attnres_block_value_norm": True},
    ],
)
def test_count_prior_requires_alpha_formula_block_sources(overrides):
    kwargs = dict(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_block_count_prior=True,
    )
    kwargs.update(overrides)

    with pytest.raises(ValueError, match="attnres_block_count_prior requires alpha-formula block summaries"):
        ModelConfig(**kwargs)


def test_count_prior_allows_sqrt_block_average():
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        attnres_type="block",
        attnres_block_average=True,
        attnres_block_average_mode="sqrt",
        attnres_block_count_prior=True,
    )

    assert cfg.attnres_block_average_mode == "sqrt"
    assert cfg.attnres_block_count_prior


@pytest.mark.parametrize("use_lrid", [False, True])
def test_count_prior_uses_cached_fused_block_path(use_lrid):
    cfg = ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        use_fused_attnres=True,
        attnres_type="block",
        attnres_num_blocks=1,
        attnres_block_average=True,
        attnres_block_average_mode="count",
        attnres_block_count_prior=True,
        attnres_training_cache_phase1=True,
        use_lrid=use_lrid,
        lrid_rank=4,
        lrid_num_heads=1,
    )
    model = OBPM(cfg).train()

    if use_lrid:
        assert not model._use_block_attnres_fused_training_path(None, False)
        assert model._use_block_lrid_attnres_fused_training_path(None, False)
    else:
        assert model._use_block_attnres_fused_training_path(None, False)
        assert not model._use_block_lrid_attnres_fused_training_path(None, False)


@pytest.mark.parametrize("use_lrid", [False, True])
def test_cached_fused_block_path_respects_count_prior_toggle(use_lrid, monkeypatch):
    cfg_kwargs = dict(
        n_layer=2,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=4,
        use_attnres=True,
        use_fused_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_average=True,
        attnres_block_average_mode="count",
        attn_res_query_init="normal",
        attnres_training_cache_phase1=True,
        use_lrid=use_lrid,
        lrid_rank=4,
        lrid_num_heads=1,
    )
    torch.manual_seed(31)
    no_prior = OBPM(ModelConfig(**cfg_kwargs, attnres_block_count_prior=False)).train()
    with_prior = OBPM(ModelConfig(**cfg_kwargs, attnres_block_count_prior=True)).train()
    with_prior.load_state_dict(no_prior.state_dict(), strict=True)
    idx = torch.randint(0, cfg_kwargs["vocab_size"], (2, cfg_kwargs["block_size"]))

    original_phase1 = model_module.attention_residual_phase1_from_logits
    captured = []

    def wrapped_phase1(values, logits, *args, **kwargs):
        captured.append(logits.detach().cpu())
        return original_phase1(values, logits, *args, **kwargs)

    monkeypatch.setattr(model_module, "attention_residual_phase1_from_logits", wrapped_phase1)
    with torch.no_grad():
        no_prior(idx, return_hidden=True)
    no_prior_logits = captured[0]
    captured.clear()

    with torch.no_grad():
        with_prior(idx, return_hidden=True)
    prior_logits = captured[0]

    assert no_prior_logits.size(1) == 2
    diff = prior_logits - no_prior_logits
    assert torch.allclose(diff[:, 0], torch.zeros_like(diff[:, 0]), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        diff[:, 1],
        torch.full_like(diff[:, 1], torch.log(torch.tensor(2.0)).item()),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("use_lrid", [False, True])
def test_fused_count_prior_matches_unfused_training_gradients(use_lrid, device_name):
    device = torch.device("cpu") if device_name == "cpu" else _cuda_device()
    atol = 2e-6 if device.type == "cpu" else 3e-4
    rtol = 2e-6 if device.type == "cpu" else 3e-4
    cfg_kwargs = dict(
        n_layer=2,
        n_head=2,
        n_embd=16,
        mlp_hidden_dim=32,
        vocab_size=64,
        block_size=6,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_average=True,
        attnres_block_average_mode="count",
        attnres_block_count_prior=True,
        attn_res_query_init="normal",
        use_lrid=use_lrid,
        lrid_rank=8,
        lrid_num_heads=1,
    )
    torch.manual_seed(21)
    ref = OBPM(ModelConfig(**cfg_kwargs, use_fused_attnres=False)).to(device).train()
    fused = OBPM(ModelConfig(**cfg_kwargs, use_fused_attnres=True)).to(device).train()
    fused.load_state_dict(ref.state_dict(), strict=True)
    if use_lrid:
        assert fused._use_block_lrid_attnres_fused_training_path(None, False)
    else:
        assert fused._use_block_attnres_fused_training_path(None, False)

    idx = torch.randint(0, cfg_kwargs["vocab_size"], (2, cfg_kwargs["block_size"]), device=device)
    ref_output = ref(idx, return_hidden=True)
    fused_output = fused(idx, return_hidden=True)
    ref_loss = ref_output.float().square().mean()
    fused_loss = fused_output.float().square().mean()
    ref_loss.backward()
    fused_loss.backward()

    assert torch.allclose(fused_output, ref_output, atol=atol, rtol=rtol)
    for (ref_name, ref_param), (fused_name, fused_param) in zip(ref.named_parameters(), fused.named_parameters()):
        assert ref_name == fused_name
        if ref_param.grad is None and fused_param.grad is None:
            continue
        assert ref_param.grad is not None
        assert fused_param.grad is not None
        assert torch.allclose(fused_param.grad, ref_param.grad, atol=atol, rtol=rtol), ref_name


@pytest.mark.parametrize("use_lrid", [False, True])
@pytest.mark.parametrize("count_prior", [False, True])
def test_block_forward_passes_expected_source_logit_biases(use_lrid, count_prior):
    cfg = ModelConfig(
        n_layer=3,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=32,
        block_size=5,
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_average=True,
        attnres_block_count_prior=count_prior,
        attnres_key_norm=False,
        use_lrid=use_lrid,
        lrid_rank=4,
        lrid_num_heads=1,
    )
    model = OBPM(cfg).eval()
    records = []
    method_name = "_apply_lrid_attnres" if use_lrid else "_apply_attnres"
    original = getattr(model, method_name)

    def wrapped(residual_idx, sources, *args, **kwargs):
        source_logit_biases = kwargs.get("source_logit_biases")
        if source_logit_biases is None:
            records.append((residual_idx, None))
        else:
            records.append(
                (
                    residual_idx,
                    [
                        float(bias.detach().cpu())
                        if torch.is_tensor(bias)
                        else float(bias)
                        for bias in source_logit_biases
                    ],
                )
            )
        return original(residual_idx, sources, *args, **kwargs)

    setattr(model, method_name, wrapped)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    with torch.no_grad():
        model(idx, return_hidden=True)

    expected_with_prior = [
        (0, None),
        (1, None),
        (2, [0.0, math.log(2.0)]),
        (3, [0.0, math.log(3.0)]),
        (4, [0.0, math.log(3.0), 0.0]),
        (5, [0.0, math.log(3.0), math.log(2.0)]),
        (6, [0.0, math.log(3.0), math.log(3.0)]),
    ]
    expected = (
        expected_with_prior
        if count_prior
        else [(residual_idx, None) for residual_idx, _ in expected_with_prior]
    )

    assert len(records) == len(expected)
    for actual, expected_item in zip(records, expected):
        assert actual[0] == expected_item[0]
        if expected_item[1] is None:
            assert actual[1] is None
        else:
            assert actual[1] == pytest.approx(expected_item[1])


def _cuda_device():
    if not torch.cuda.is_available() or not is_fused_attnres_available():
        pytest.skip("CUDA/Triton fused AttnRes path is not available")
    return torch.device("cuda")


def _assert_close(actual, expected, dtype):
    diff = (actual.float() - expected.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    if dtype == torch.float32:
        assert max_err < 2e-4
        assert mean_err < 2e-5
    else:
        assert max_err < 8e-2
        assert mean_err < 8e-3


@pytest.mark.parametrize("key_norm", [False, True])
def test_fused_base_attnres_matches_torch(key_norm):
    device = _cuda_device()
    torch.manual_seed(1)
    dtype = torch.float32
    sources = [torch.randn(2, 5, 64, device=device, dtype=dtype) for _ in range(5)]
    query = torch.randn(64, device=device, dtype=dtype)

    expected = attention_residual_read_torch(sources, query, key_norm)
    actual = attention_residual_read(sources, query, key_norm, force_triton=True)
    torch.cuda.synchronize()
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("key_norm", [False, True])
def test_fused_lrid_static_query_matches_torch(num_heads, key_norm):
    device = _cuda_device()
    torch.manual_seed(2 + num_heads + int(key_norm))
    dtype = torch.float32
    rank = 32
    n_embd = 128
    sources = [torch.randn(2, 7, n_embd, device=device, dtype=dtype) for _ in range(6)]
    keys = [torch.randn(2, 7, rank, device=device, dtype=dtype) for _ in range(6)]
    query = torch.randn(num_heads, rank // num_heads, device=device, dtype=dtype)

    expected = lrid_attention_residual_read_torch(sources, keys, query, num_heads, 0.25, key_norm)
    actual = lrid_attention_residual_read(sources, keys, query, num_heads, 0.25, key_norm, force_triton=True)
    torch.cuda.synchronize()
    _assert_close(actual, expected, dtype)


def test_fused_lrid_r512_single_head_eval_falls_back_to_torch():
    device = _cuda_device()
    torch.manual_seed(26)
    dtype = torch.float32
    num_heads = 1
    rank = 512
    n_embd = 1024
    sources = [torch.randn(1, 3, n_embd, device=device, dtype=dtype) for _ in range(3)]
    keys = [torch.randn(1, 3, rank, device=device, dtype=dtype) for _ in range(3)]
    query = torch.randn(num_heads, rank // num_heads, device=device, dtype=dtype)

    with torch.no_grad():
        expected = lrid_attention_residual_read_torch(sources, keys, query, num_heads, 1.0, True)
        actual = lrid_attention_residual_read(sources, keys, query, num_heads, 1.0, True)
    torch.cuda.synchronize()
    _assert_close(actual, expected, dtype)


def test_fused_lrid_output_tail_keys_match_torch():
    device = _cuda_device()
    torch.manual_seed(6)
    dtype = torch.float32
    num_heads = 2
    rank = 32
    n_embd = 128
    ref_sources = [
        torch.randn(2, 7, n_embd, device=device, dtype=dtype, requires_grad=True)
        for _ in range(6)
    ]
    actual_sources = [
        source.detach().clone().requires_grad_(True)
        for source in ref_sources
    ]
    ref_keys = [source[..., -rank:].contiguous() for source in ref_sources]
    actual_keys = [source[..., -rank:].contiguous() for source in actual_sources]
    ref_query = torch.randn(num_heads, rank // num_heads, device=device, dtype=dtype, requires_grad=True)
    actual_query = ref_query.detach().clone().requires_grad_(True)

    expected = lrid_attention_residual_read_torch(ref_sources, ref_keys, ref_query, num_heads, 0.25, True)
    actual = lrid_attention_residual_read(
        actual_sources,
        actual_keys,
        actual_query,
        num_heads,
        0.25,
        True,
        force_triton=True,
    )
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.cuda.synchronize()

    _assert_close(actual, expected, dtype)
    for actual_source, ref_source in zip(actual_sources, ref_sources):
        _assert_close(actual_source.grad, ref_source.grad, dtype)
    _assert_close(actual_query.grad, ref_query.grad, dtype)


def test_fused_lrid_dynamic_query_matches_torch():
    device = _cuda_device()
    torch.manual_seed(7)
    dtype = torch.float32
    num_heads = 2
    rank = 32
    n_embd = 128
    sources = [torch.randn(2, 6, n_embd, device=device, dtype=dtype) for _ in range(4)]
    keys = [torch.randn(2, 6, rank, device=device, dtype=dtype) for _ in range(4)]
    query = torch.randn(2, 6, num_heads, rank // num_heads, device=device, dtype=dtype)

    expected = lrid_attention_residual_read_torch(sources, keys, query, num_heads, 0.5, True)
    actual = lrid_attention_residual_read(sources, keys, query, num_heads, 0.5, True, force_triton=True)
    torch.cuda.synchronize()
    _assert_close(actual, expected, dtype)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="bf16 CUDA is not available")
def test_fused_lrid_bfloat16_matches_torch():
    _cuda_device()
    torch.manual_seed(8)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = 1
    rank = 64
    n_embd = 256
    sources = [torch.randn(1, 8, n_embd, device=device, dtype=dtype) for _ in range(5)]
    keys = [torch.randn(1, 8, rank, device=device, dtype=dtype) for _ in range(5)]
    query = torch.randn(num_heads, rank // num_heads, device=device, dtype=dtype)

    expected = lrid_attention_residual_read_torch(sources, keys, query, num_heads, 0.125, True)
    actual = lrid_attention_residual_read(sources, keys, query, num_heads, 0.125, True, force_triton=True)
    torch.cuda.synchronize()
    _assert_close(actual, expected, dtype)


def test_fused_lrid_backward_smoke():
    device = _cuda_device()
    torch.manual_seed(9)
    values = torch.randn(4, 2, 3, 64, device=device, requires_grad=True)
    keys = torch.randn(4, 2, 3, 32, device=device, requires_grad=True)
    query = torch.randn(2, 16, device=device, requires_grad=True)

    out = lrid_attention_residual_read(
        list(values.unbind(0)),
        list(keys.unbind(0)),
        query,
        2,
        0.25,
        True,
        force_triton=True,
    )
    out.square().mean().backward()
    assert values.grad is not None
    assert keys.grad is not None
    assert query.grad is not None


@pytest.mark.parametrize("normalize_output", [False, True])
def test_phase1_from_logits_backward_matches_torch(normalize_output):
    device = _cuda_device()
    torch.manual_seed(11 + int(normalize_output))
    S, Q, B, T, D = 4, 3, 2, 5, 32
    values = torch.randn(S, B, T, D, device=device, requires_grad=True)
    logits = torch.randn(Q, S, B, T, device=device, requires_grad=True)
    ref_values = values.detach().clone().requires_grad_(True)
    ref_logits = logits.detach().clone().requires_grad_(True)

    actual, actual_lse = attention_residual_phase1_from_logits(
        values,
        logits,
        force_triton=True,
        normalize_output=normalize_output,
    )
    weights = torch.softmax(ref_logits.float(), dim=1).to(ref_values.dtype)
    expected = torch.einsum("qsbt,sbtd->qbtd", weights, ref_values)
    if normalize_output:
        expected = torch.nn.functional.rms_norm(expected, (D,))
    expected_lse = torch.logsumexp(ref_logits.float(), dim=1)

    upstream = torch.randn_like(actual)
    lse_upstream = torch.randn_like(actual_lse)
    actual.backward(upstream, retain_graph=True)
    actual_lse.backward(lse_upstream)
    expected.backward(upstream, retain_graph=True)
    expected_lse.backward(lse_upstream)
    torch.cuda.synchronize()

    _assert_close(actual, expected, torch.float32)
    _assert_close(actual_lse, expected_lse, torch.float32)
    _assert_close(values.grad, ref_values.grad, torch.float32)
    _assert_close(logits.grad, ref_logits.grad, torch.float32)


@pytest.mark.parametrize("normalize_output", [False, True])
def test_phase1_from_logits_list_backward_matches_torch(normalize_output, monkeypatch):
    device = _cuda_device()
    monkeypatch.setenv("ATTNRES_PHASE1_LOGITS_LIST", "1")
    torch.manual_seed(31 + int(normalize_output))
    S, Q, B, T, D = 5, 3, 2, 4, 32
    values = [torch.randn(B, T, D, device=device, requires_grad=True) for _ in range(S)]
    logits = torch.randn(Q, S, B, T, device=device, requires_grad=True)
    ref_values = torch.stack([value.detach().clone() for value in values], dim=0).requires_grad_(True)
    ref_logits = logits.detach().clone().requires_grad_(True)

    actual, actual_lse = attention_residual_phase1_from_logits(
        values,
        logits,
        force_triton=True,
        normalize_output=normalize_output,
    )
    weights = torch.softmax(ref_logits.float(), dim=1).to(ref_values.dtype)
    expected = torch.einsum("qsbt,sbtd->qbtd", weights, ref_values)
    if normalize_output:
        expected = torch.nn.functional.rms_norm(expected, (D,))
    expected_lse = torch.logsumexp(ref_logits.float(), dim=1)

    upstream = torch.randn_like(actual)
    lse_upstream = torch.randn_like(actual_lse)
    actual.backward(upstream, retain_graph=True)
    actual_lse.backward(lse_upstream)
    expected.backward(upstream, retain_graph=True)
    expected_lse.backward(lse_upstream)
    torch.cuda.synchronize()

    _assert_close(actual, expected, torch.float32)
    _assert_close(actual_lse, expected_lse, torch.float32)
    for source_idx, value in enumerate(values):
        _assert_close(value.grad, ref_values.grad[source_idx], torch.float32)
    _assert_close(logits.grad, ref_logits.grad, torch.float32)


@pytest.mark.parametrize("key_norm", [False, True])
@pytest.mark.parametrize("normalize_output", [False, True])
def test_phase2_backward_matches_torch(key_norm, normalize_output):
    device = _cuda_device()
    torch.manual_seed(13 + int(key_norm) + 2 * int(normalize_output))
    B, T, D = 2, 5, 32
    partial = torch.randn(B, T, D, device=device, requires_grad=True)
    query = torch.randn(D, device=device, requires_grad=True)
    inter = torch.randn(B, T, D, device=device, requires_grad=True)
    lse = torch.randn(B, T, device=device, requires_grad=True)
    ref_partial = partial.detach().clone().requires_grad_(True)
    ref_query = query.detach().clone().requires_grad_(True)
    ref_inter = inter.detach().clone().requires_grad_(True)
    ref_lse = lse.detach().clone().requires_grad_(True)

    actual = attention_residual_phase2(
        partial,
        query,
        inter,
        lse,
        key_norm,
        force_triton=True,
        normalize_output=normalize_output,
    )
    key = torch.nn.functional.rms_norm(ref_partial, (D,)) if key_norm else ref_partial
    logit = torch.sum(key * ref_query.view(1, 1, D), dim=-1)
    prob = torch.sigmoid(logit.float() - ref_lse.float()).to(ref_partial.dtype)
    expected = ref_inter + prob.unsqueeze(-1) * (ref_partial - ref_inter)
    if normalize_output:
        expected = torch.nn.functional.rms_norm(expected, (D,))

    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.cuda.synchronize()

    _assert_close(actual, expected, torch.float32)
    _assert_close(partial.grad, ref_partial.grad, torch.float32)
    _assert_close(query.grad, ref_query.grad, torch.float32)
    _assert_close(inter.grad, ref_inter.grad, torch.float32)
    _assert_close(lse.grad, ref_lse.grad, torch.float32)


@pytest.mark.parametrize("use_lrid", [False, True])
@pytest.mark.parametrize("attnres_type", ["block", "full"])
@pytest.mark.parametrize("lrid_key_from_output_tail", [False, True])
def test_model_fused_attnres_matches_pytorch_path(use_lrid, attnres_type, lrid_key_from_output_tail):
    device = _cuda_device()
    lrid_key_from_output_tail = bool(use_lrid and lrid_key_from_output_tail)
    torch.manual_seed(10 + int(use_lrid) + (100 if lrid_key_from_output_tail else 0))
    common = dict(
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
        attnres_key_norm=True,
        use_lrid=use_lrid,
        lrid_rank=32,
        lrid_num_heads=2,
        lrid_key_from_output_tail=lrid_key_from_output_tail,
        lrid_use_logit_scale=True,
    )
    ref = OBPM(ModelConfig(**common, use_fused_attnres=False)).to(device).eval()
    fused = OBPM(ModelConfig(**common, use_fused_attnres=True)).to(device).eval()
    fused.load_state_dict(ref.state_dict())
    idx = torch.randint(0, common["vocab_size"], (2, common["block_size"]), device=device)

    with torch.no_grad():
        expected = ref(idx, return_hidden=True)
        actual = fused(idx, return_hidden=True)
    torch.cuda.synchronize()
    _assert_close(actual, expected, torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="bf16 CUDA is not available")
@pytest.mark.parametrize("use_lrid", [False, True])
@pytest.mark.parametrize("lrid_key_from_output_tail", [False, True])
def test_cached_block_training_path_matches_pytorch_bfloat16(use_lrid, lrid_key_from_output_tail):
    device = _cuda_device()
    lrid_key_from_output_tail = bool(use_lrid and lrid_key_from_output_tail)
    torch.manual_seed(14 + int(use_lrid) + (100 if lrid_key_from_output_tail else 0))
    common = dict(
        n_layer=4,
        n_head=4,
        n_embd=128,
        mlp_hidden_dim=256,
        vocab_size=256,
        block_size=32,
        flash_attention=False,
        norm_pos="before",
        use_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_block_average=True,
        attnres_key_norm=True,
        attn_res_query_init="normal",
        use_lrid=use_lrid,
        lrid_rank=32,
        lrid_num_heads=1,
        lrid_key_from_output_tail=lrid_key_from_output_tail,
        lrid_use_logit_scale=True,
        attnres_training_cache_phase1=True,
        attnres_training_torch_phase2=True,
    )
    ref = OBPM(ModelConfig(**common, use_fused_attnres=False)).to(device).train().to(dtype=torch.bfloat16)
    fused = OBPM(ModelConfig(**common, use_fused_attnres=True)).to(device).train().to(dtype=torch.bfloat16)
    fused.load_state_dict(ref.state_dict())
    idx = torch.randint(0, common["vocab_size"], (2, common["block_size"]), device=device)

    expected = ref(idx, return_hidden=True)
    actual = fused(idx, return_hidden=True)
    torch.cuda.synchronize()
    _assert_close(actual, expected, torch.bfloat16)

    ref_loss = expected.float().square().mean()
    fused_loss = actual.float().square().mean()
    ref_loss.backward()
    fused_loss.backward()
    torch.cuda.synchronize()

    _assert_close(
        fused.transformer.wte.weight.grad,
        ref.transformer.wte.weight.grad,
        torch.bfloat16,
    )


def test_lrid_projection_rank_padding_is_ignored():
    device = _cuda_device()
    torch.manual_seed(12)
    common = dict(
        n_layer=3,
        n_head=4,
        n_embd=64,
        mlp_hidden_dim=128,
        vocab_size=128,
        block_size=12,
        flash_attention=False,
        norm_pos="before",
        use_attnres=True,
        use_fused_attnres=True,
        attnres_type="block",
        attnres_num_blocks=2,
        attnres_key_norm=True,
        use_lrid=True,
        lrid_rank=32,
        lrid_projection_rank=64,
        lrid_num_heads=2,
        lrid_use_logit_scale=True,
    )
    ref = OBPM(ModelConfig(**common)).to(device).eval()
    padded = OBPM(ModelConfig(**common)).to(device).eval()
    padded.load_state_dict(ref.state_dict())

    with torch.no_grad():
        for module in padded.modules():
            if not hasattr(module, "projection_rank") or module.projection_rank == module.rank:
                continue
            start = module.key_offset + module.rank
            end = module.key_offset + module.projection_rank
            module.proj.weight[start:end].normal_(mean=0.0, std=10.0)

    idx = torch.randint(0, common["vocab_size"], (2, common["block_size"]), device=device)
    with torch.no_grad():
        expected = ref(idx, return_hidden=True)
        actual = padded(idx, return_hidden=True)
    torch.cuda.synchronize()
    _assert_close(actual, expected, torch.float32)


def test_lrid_fused_projection_value_is_contiguous():
    device = _cuda_device()
    config = ModelConfig(
        n_layer=1,
        n_head=4,
        n_embd=64,
        mlp_hidden_dim=128,
        use_lrid=True,
        lrid_rank=16,
        lrid_projection_rank=32,
    )
    model = OBPM(config).to(device).eval()
    x = torch.randn(2, 8, config.n_embd, device=device)

    with torch.no_grad():
        value, key = model.transformer.layers[0].attn.c_proj(x)

    assert value.is_contiguous()
    assert key is not None


def test_lrid_output_tail_key_uses_value_tail():
    device = _cuda_device()
    config = ModelConfig(
        n_layer=1,
        n_head=4,
        n_embd=64,
        mlp_hidden_dim=128,
        use_lrid=True,
        lrid_rank=16,
        lrid_key_from_output_tail=True,
    )
    model = OBPM(config).to(device).eval()
    x = torch.randn(2, 8, config.n_embd, device=device)

    with torch.no_grad():
        value, key = model.transformer.layers[0].attn.c_proj(x)

    assert model.transformer.layers[0].attn.c_proj.proj.out_features == config.n_embd
    assert value.is_contiguous()
    assert key.is_contiguous()
    _assert_close(key, value[..., -config.lrid_rank:], torch.float32)
