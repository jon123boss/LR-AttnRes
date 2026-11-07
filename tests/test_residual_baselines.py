"""Baseline integration: shared backbone, training, checkpoints and decoding."""

from dataclasses import asdict
import io
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch.nn import functional as F

from checkpoint_config import model_config_from_checkpoint
from model import ModelConfig, MultiHeadAttention, OBPM
from optimizer import OptimizerConfig, configure_optimizers
import run_eval
from utils import get_model, load_model_checkpoint, validate_exact_resume_data_config


def config(kind="standard", **kwargs):
    return ModelConfig(
        n_layer=2, n_head=2, n_embd=16, mlp_hidden_dim=24,
        vocab_size=23, block_size=8, norm_pos="before",
        residual_type=kind, **kwargs,
    )


@pytest.mark.parametrize("kind", ["gated", "ihc", "mhc"])
def test_shared_initialization_training_and_optimizer_coverage(kind):
    torch.manual_seed(71)
    reference = OBPM(config())
    torch.manual_seed(71)
    model = OBPM(config(kind))
    parameters = dict(model.named_parameters())
    for name, value in reference.named_parameters():
        torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)

    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    logits = model(tokens)
    hidden = model(tokens, return_hidden=True)
    torch.testing.assert_close(logits, F.linear(hidden, model.get_lm_head_weight()))
    F.cross_entropy(logits.flatten(0, 1), tokens.roll(-1, 1).flatten()).backward()
    for name, value in parameters.items():
        assert value.grad is not None, name
        assert torch.isfinite(value.grad).all(), name

    optimizers = configure_optimizers(model, OptimizerConfig(verbose=False))
    groups = [[p for group in opt.param_groups for p in group["params"]] for opt in optimizers]
    ids = [id(p) for group in groups for p in group]
    assert len(ids) == len(set(ids)) == len(parameters)
    assert set(ids) == {id(p) for p in model.parameters()}
    adam_ids = {id(p) for p in groups[1]}
    assert all(id(p) in adam_ids for name, p in parameters.items() if ".residual_" in name)
    for optimizer in optimizers:
        optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize("kind", ["gated", "ihc", "mhc"])
def test_checkpoint_and_mixed_precision_preserve_controller_values(kind, tmp_path):
    model = OBPM(config(kind))
    # Values not exactly representable in BF16 expose destructive round trips.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".residual_" in name:
                p.add_(0.00012345)
    path = tmp_path / "checkpoint.pt"
    torch.save({"step": 1, "model_args": asdict(model.config), "model": model.state_dict()}, path)
    _, restored, restored_config = load_model_checkpoint(path, torch.device("cpu"), verbose=False)
    assert restored_config.residual_type == kind
    tokens = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
    preserved = {
        name: p.detach().clone() for name, p in restored.named_parameters()
        if ".residual_" in name and (kind != "gated" or name.endswith("norm_weight"))
    }
    restored.to_mixed_precision()
    for name, expected in preserved.items():
        actual = restored.get_parameter(name)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    logits = restored(tokens)
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
    logits.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in restored.parameters())


@pytest.mark.parametrize("kind", ["gated", "ihc", "mhc"])
def test_cached_decode_and_document_isolation(kind, monkeypatch):
    model = OBPM(config(kind)).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    expected = model(tokens)
    _, cache = model(tokens[:, :3], use_cache=True)
    pieces = []
    for index in range(3, 6):
        logits, cache = model(tokens[:, index:index + 1], past_kv=cache, use_cache=True)
        pieces.append(logits)
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected[:, 3:], rtol=1e-5, atol=1e-6)

    # Exercise packed-document plumbing with a CPU SDPA oracle; actual
    # FlashAttention varlen execution requires the CUDA qualification job.
    def packed_attention(q, k, v, *, cu_seqlens_q, cu_seqlens_k,
                         max_seqlen_q, max_seqlen_k, causal):
        assert max_seqlen_q == max_seqlen_k == 3
        assert torch.equal(cu_seqlens_q, cu_seqlens_k)
        bounds = cu_seqlens_q.tolist()
        return torch.cat([
            F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1), k[start:end].transpose(0, 1),
                v[start:end].transpose(0, 1), is_causal=causal,
            ).transpose(0, 1)
            for start, end in zip(bounds, bounds[1:])
        ])

    monkeypatch.setattr(MultiHeadAttention, "flash_attn_varlen_func", staticmethod(packed_attention))
    boundaries = torch.tensor([0, 3, 6], dtype=torch.int32)
    original = model(tokens, cu_doc_len=boundaries, max_doc_len=3)
    changed = tokens.clone()
    changed[:, :3] = torch.tensor([7, 8, 9])
    isolated = model(changed, cu_doc_len=boundaries, max_doc_len=3)
    torch.testing.assert_close(isolated[:, 3:], original[:, 3:], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gated", "ihc", "mhc"])
def test_fullgraph_capture_matches_outputs_and_gradients(kind):
    torch.manual_seed(9)
    model = OBPM(config(kind))
    other = OBPM(config(kind))
    other.load_state_dict(model.state_dict())
    compiled = torch.compile(other, backend="eager", fullgraph=True)
    tokens = torch.tensor([[1, 2, 3]])
    expected, actual = model(tokens), compiled(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for left, right in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)


def test_invalid_combinations_and_legacy_config_defaults():
    for flags in ({"use_attnres": True}, {"use_lrid": True}, {"use_fused_attnres": True}):
        with pytest.raises(ValueError, match="cannot be combined"):
            config("ihc", **flags)
    with pytest.raises(ValueError, match="norm_pos"):
        ModelConfig(residual_type="gated")
    restored = model_config_from_checkpoint({"n_embd": 16, "n_head": 2})
    assert restored.residual_type == "standard"


@pytest.mark.parametrize("kind", ["gated", "ihc", "mhc"])
def test_training_factory_uses_residual_configuration(kind):
    settings = asdict(config(kind, residual_num_streams=2, gated_residual_rank=3, mhc_sinkhorn_iters=7))
    settings.update(init_from="scratch", master_process=False)
    step, checkpoint, model, actual_config = get_model(settings, torch.device("cpu"))
    assert step == 0 and checkpoint is None
    assert actual_config.residual_type == kind
    assert actual_config.residual_num_streams == 2
    assert actual_config.gated_residual_rank == 3
    assert actual_config.mhc_sinkhorn_iters == 7
    assert model(torch.tensor([[1, 2]])).shape == (1, 2, 23)


@pytest.mark.parametrize("field,backend", [("gated_residual_norm_backend", "fla"), ("mhc_sinkhorn_backend", "tile")])
def test_exact_resume_rejects_kernel_changes(field, backend):
    checkpoint = {"train_batches_consumed": 1, "config": {field: "torch"}}
    with pytest.raises(ValueError, match=field):
        validate_exact_resume_data_config(checkpoint, {field: backend})


@pytest.mark.parametrize("kind,flags", [
    ("gated", {"gated_residual_norm_backend": "fla"}),
    ("mhc", {"mhc_sinkhorn_backend": "tile"}),
])
def test_evaluation_rejects_compiling_eager_only_wrappers(kind, flags):
    model = OBPM(config(kind))
    saved_config = config(kind, **flags)
    with mock.patch.object(run_eval, "load_model_checkpoint", return_value=({}, model, saved_config)):
        with pytest.raises(ValueError, match="eager evaluation"):
            run_eval.OBPMWrapper("unused.pt", device="cpu", torch_compile=True, verbose=False)


@pytest.mark.parametrize("filename", ["residual_gated.py", "residual_ihc.py", "residual_mhc.py"])
def test_evaluation_hash_changes_with_controller_source(filename):
    expected = run_eval._evaluation_source_sha256()
    original_open = open

    def changed_source(path, *args, **kwargs):
        if Path(path).name == filename:
            with original_open(path, "rb") as source:
                return io.BytesIO(source.read() + b"\n# source change\n")
        return original_open(path, *args, **kwargs)

    with mock.patch("builtins.open", side_effect=changed_source):
        assert run_eval._evaluation_source_sha256() != expected
