import pytest
import torch

from attnres_ops import (
    attention_residual_phase1_from_logits,
    attention_residual_phase2,
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
