import pytest

from fast_attnres import fast_attnres_config_decision
from model import ModelConfig, OBPM


def _small_config(**overrides):
    values = {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 8,
        "mlp_hidden_dim": 16,
        "vocab_size": 19,
        "block_size": 4,
        "flash_attention": False,
        "attnres_type": "full",
        "attnres_backend": "auto",
        "lrid_use_logit_scale": False,
    }
    values.update(overrides)
    return ModelConfig(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_attnres": True},
        {
            "use_lrid": True,
            "lrid_key_from_output_tail": True,
            "lrid_rank": 4,
        },
        {
            "use_lrid": True,
            "lrid_key_from_output_tail": True,
            "lrid_rank": 8,
        },
    ],
)
def test_auto_resolves_standard_and_every_valid_sliced_rank_to_fast(overrides):
    config = _small_config(**overrides)
    assert config.attnres_backend == "fast"
    assert config._attnres_backend_requested == "auto"


def test_auto_leaves_non_routed_and_projected_models_on_legacy():
    assert _small_config().attnres_backend == "legacy"
    assert _small_config(use_lrid=True, lrid_rank=4).attnres_backend == "legacy"


def test_explicit_legacy_is_preserved_for_controlled_reference_runs():
    config = _small_config(use_attnres=True, attnres_backend="legacy")
    assert config.attnres_backend == "legacy"
    assert config._attnres_backend_requested == "legacy"


def test_sliced_rank_equal_to_width_is_fast_eligible():
    decision = fast_attnres_config_decision(
        use_attnres=True,
        use_lrid=True,
        attnres_type="full",
        key_norm=True,
        lrid_rank=8,
        n_embd=8,
        lrid_key_from_output_tail=True,
    )
    assert decision.eligible


def test_fast_model_is_disarmed_until_cuda_bf16_qualification():
    model = OBPM(_small_config(use_attnres=True))
    assert not model._fast_attnres_enabled
    with pytest.raises(RuntimeError, match="Fast-AttnRes-only contract failed"):
        model.require_fast_attnres(validate_package=False)
    assert not model._fast_attnres_enabled


def test_nominal_sliced_route_does_not_fall_back_for_unsupported_semantics():
    config = _small_config(
        use_lrid=True,
        lrid_key_from_output_tail=True,
        lrid_rank=4,
        lrid_num_heads=2,
    )
    assert config.attnres_backend == "fast"
    report = OBPM(config)._fast_attnres_config_decision()
    assert not report.eligible
    assert report.reason == "multi_head_lrid"

