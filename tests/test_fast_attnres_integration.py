from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from fast_attnres import format_fast_attnres_banner, print_fast_attnres_banner
from model import ModelConfig, OBPM
import utils


def _config(backend="fast"):
    return ModelConfig(
        n_layer=1,
        n_head=2,
        n_embd=8,
        mlp_hidden_dim=16,
        vocab_size=17,
        block_size=4,
        flash_attention=False,
        norm_pos="before",
        use_attnres=True,
        attnres_type="full",
        attnres_backend=backend,
        attn_res_query_init="normal",
    )


def _step(model, optimizer, tokens, labels):
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.detach()


def test_old_checkpoint_without_backend_loads_legacy(tmp_path):
    torch.manual_seed(801)
    config = _config("legacy")
    model = OBPM(config).eval()
    old_args = asdict(config)
    old_args.pop("attnres_backend")
    path = tmp_path / "old.pt"
    utils.atomic_torch_save(
        {"model_args": old_args, "model": model.state_dict(), "config": {}, "step": 3},
        path,
    )
    checkpoint, restored, restored_config = utils.load_model_checkpoint(
        path, torch.device("cpu"), verbose=False
    )
    assert checkpoint["step"] == 3
    assert restored_config.attnres_backend == "legacy"
    assert restored.fast_attnres_route_report()["active_reads"] == 0


def test_fast_checkpoint_resume_restores_optimizer_and_next_update(tmp_path):
    torch.manual_seed(802)
    config = _config("fast")
    reference = OBPM(config).train()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    first_tokens = torch.tensor([[0, 1, 2, 3]])
    first_labels = torch.tensor([[1, 2, 3, 4]])
    second_tokens = torch.tensor([[4, 3, 2, 1]])
    second_labels = torch.tensor([[3, 2, 1, 0]])
    _step(reference, optimizer, first_tokens, first_labels)

    path = tmp_path / "fast.pt"
    utils.atomic_torch_save(
        {
            "model_args": asdict(config),
            "model": reference.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {"attnres_backend": "fast"},
            "step": 1,
        },
        path,
    )
    checkpoint, resumed, resumed_config = utils.load_model_checkpoint(
        path, torch.device("cpu"), verbose=False
    )
    assert resumed_config.attnres_backend == "fast"
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])

    expected_loss = _step(reference, optimizer, second_tokens, second_labels)
    actual_loss = _step(resumed, resumed_optimizer, second_tokens, second_labels)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for expected, actual in zip(reference.parameters(), resumed.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exact_resume_rejects_fast_backend_drift():
    checkpoint = {
        "train_batches_consumed": 1,
        "config": {"attnres_backend": "fast"},
    }
    with pytest.raises(ValueError, match="attnres_backend"):
        utils.validate_exact_resume_data_config(
            checkpoint,
            {"attnres_backend": "legacy"},
        )


def test_banner_only_formats_for_active_fast_route():
    active = OBPM(_config("fast")).fast_attnres_startup_report(validate_package=True)
    line = format_fast_attnres_banner(active)
    assert line == (
        "[Fast-AttnRes] backend=fast-attnres version=1.0.0 "
        "active_reads=2/2 legacy_fallback_reads=0"
    )
    inactive = OBPM(_config("legacy")).fast_attnres_startup_report(validate_package=False)
    assert format_fast_attnres_banner(inactive) is None
    printed = []

    def capture(line, **kwargs):
        printed.append((line, kwargs))

    assert not print_fast_attnres_banner(active, is_rank_zero=False, print_fn=capture)
    assert print_fast_attnres_banner(active, is_rank_zero=True, print_fn=capture)
    assert printed == [(line, {"flush": True})]


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile is unavailable")
def test_fast_fullgraph_changed_input_execution():
    torch.manual_seed(803)
    model = OBPM(_config("fast")).eval()
    model.fast_attnres_startup_report(validate_package=True)
    compiled = torch.compile(model, backend="eager", fullgraph=True, dynamic=False)
    first = torch.tensor([[0, 1, 2, 3]])
    second = torch.tensor([[3, 2, 1, 0]])
    with torch.no_grad():
        torch.testing.assert_close(compiled(first), model(first))
        torch.testing.assert_close(compiled(second), model(second))
        assert not torch.equal(compiled(first), compiled(second))
