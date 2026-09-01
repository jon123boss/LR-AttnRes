import math

from benchmark_fast_attnres import (
    ARMS,
    TARGET,
    _balanced_orders,
    _profile_kwargs,
    _report,
)


def test_benchmark_keeps_public_main_sdpa_fallback_and_only_varies_attnres():
    legacy = _profile_kwargs("full_standard_r1024", "legacy_attnres", TARGET)
    fast = _profile_kwargs("full_standard_r1024", "fast_attnres", TARGET)
    pre_norm = _profile_kwargs("full_standard_r1024", "pre_norm_only", TARGET)

    assert legacy["flash_attention"] is fast["flash_attention"] is pre_norm["flash_attention"] is False
    assert legacy["attnres_backend"] == "legacy"
    assert fast["attnres_backend"] == "fast"
    assert pre_norm["use_attnres"] is False
    ignored = {"attnres_backend", "use_attnres"}
    assert {key: value for key, value in legacy.items() if key not in ignored} == {
        key: value for key, value in fast.items() if key not in ignored
    }


def test_output_tail_profile_is_static_single_head_r64():
    config = _profile_kwargs("full_output_tail_r64", "fast_attnres", TARGET)
    assert config["use_lrid"] is True
    assert config["lrid_rank"] == 64
    assert config["lrid_num_heads"] == 1
    assert config["lrid_input_dependent_query"] is False
    assert config["lrid_key_from_output_tail"] is True


def test_balanced_orders_cover_every_permutation_per_six_rounds():
    orders = _balanced_orders(12, seed=7)
    assert len(orders) == 12
    assert set(orders[:6]) == set(orders[6:])
    assert all(tuple(sorted(order)) == tuple(sorted(ARMS)) for order in orders)


def test_report_uses_paired_arms_and_emits_prenorm_throughput_comparison():
    samples = []
    latencies = {"legacy_attnres": 10.0, "fast_attnres": 8.0, "pre_norm_only": 5.0}
    for seed in (1, 2):
        for pair in range(3):
            digest = f"{seed}:{pair}"
            for arm, latency in latencies.items():
                samples.append(
                    {
                        "phase": "timed",
                        "profile": "full_standard_r1024",
                        "seed": seed,
                        "pair_index": pair,
                        "arm": arm,
                        "input_sha256": digest,
                        "elapsed_ms": latency,
                        "tokens_per_second": 1000.0 / latency,
                    }
                )
    report = _report(samples, ("full_standard_r1024",), replicates=50, seed=9)
    contrasts = report["profiles"]["full_standard_r1024"]["contrasts"]
    assert math.isclose(contrasts["fast_vs_legacy_latency_reduction_percent"]["estimate"], 20.0)
    assert math.isclose(
        contrasts["fast_vs_pre_norm_tokens_per_second_change_percent"]["estimate"],
        -37.5,
    )
