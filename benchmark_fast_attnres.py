"""H100 training-step benchmark for Fast-AttnRes versus the legacy path.

The attention implementation is deliberately not configurable here: every arm
uses the unchanged public-main SDPA fallback. Measurements are paired by input
and balanced across execution order.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import permutations
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from fast_attnres import print_fast_attnres_banner
from model import ModelConfig, MultiHeadAttention, OBPM


ARMS = ("legacy_attnres", "fast_attnres", "pre_norm_only")
PROFILES = ("full_standard_r1024", "full_output_tail_r64")
FULL_SEEDS = (20260827, 20260903, 20260911)


@dataclass(frozen=True)
class Shape:
    layers: int
    width: int
    heads: int
    ffn: int
    batch: int
    tokens: int
    vocab: int


TARGET = Shape(24, 1024, 16, 2816, 2, 2048, 100277)
SMOKE = Shape(2, 128, 4, 256, 2, 32, 257)


def _profile_kwargs(profile: str, arm: str, shape: Shape) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    use_attnres = arm != "pre_norm_only"
    output_tail = profile == "full_output_tail_r64" and use_attnres
    rank = 64 if profile == "full_output_tail_r64" else shape.width
    return {
        "block_size": shape.tokens,
        "vocab_size": shape.vocab,
        "n_layer": shape.layers,
        "n_head": shape.heads,
        "n_embd": shape.width,
        "mlp_hidden_dim": shape.ffn,
        "weight_tying": False,
        "norm_pos": "before",
        "qk_norm": True,
        # Exact public-main SDPA fallback; fixed identically in every arm.
        "flash_attention": False,
        "use_attnres": use_attnres,
        "attnres_type": "full",
        "use_fused_attnres": False,
        "attnres_backend": "fast" if arm == "fast_attnres" else "legacy",
        "attnres_key_norm": True,
        "attn_res_query_norm": False,
        "attn_res_query_init": "zero",
        "use_lrid": output_tail,
        "lrid_rank": rank,
        "lrid_num_heads": 1,
        "lrid_input_dependent_query": False,
        "lrid_key_from_output_tail": output_tail,
        "lrid_key_from_value": False,
        "lrid_key_from_value_shared": False,
        "lrid_query_from_value": False,
        "lrid_query_from_value_shared": False,
        "lrid_static_embedding_key": False,
        "lrid_add_static_embedding_key": False,
        "lrid_add_static_source_key": False,
        "lrid_use_logit_scale": True,
        "attnres_block_count_prior": False,
    }


def _make_model(profile: str, arm: str, shape: Shape, seed: int) -> OBPM:
    torch.manual_seed(seed)
    model = OBPM(ModelConfig(**_profile_kwargs(profile, arm, shape)))
    return model.to_mixed_precision(torch.bfloat16).cuda()


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=False,
        capturable=True,
    )


def _batch(shape: Shape, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(shape.vocab, (shape.batch, shape.tokens), generator=generator)
    y = torch.randint(shape.vocab, (shape.batch, shape.tokens), generator=generator)
    digest = sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
    return {
        "x": x.cuda(non_blocking=False),
        "y": y.cuda(non_blocking=False),
        "cu_doc_len": None,
        "max_doc_len": None,
        "sha256": digest,
    }


def _loss(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    logits = model(
        batch["x"],
        cu_doc_len=batch["cu_doc_len"],
        max_doc_len=batch["max_doc_len"],
    )
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        batch["y"].reshape(-1),
        reduction="mean",
    )


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: dict) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(model, batch)
    loss.backward()
    optimizer.step()
    value = float(loss.detach())
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite training loss: {value}")
    return value


def _timed_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: dict) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = _step(model, optimizer, batch)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), loss


def _balanced_orders(rounds: int, seed: int) -> list[tuple[str, ...]]:
    cycle = list(permutations(ARMS))
    random.Random(seed).shuffle(cycle)
    return [cycle[index % len(cycle)] for index in range(rounds)]


def _assert_public_main_attention() -> dict:
    """Qualify the SDPA fallback used by the unchanged public-main model."""
    torch.manual_seed(20260901)
    tensors = [
        torch.randn(1, 2, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    output = F.scaled_dot_product_attention(*tensors, is_causal=True)
    output.float().square().mean().backward()
    if not all(tensor.grad is not None for tensor in tensors):
        raise RuntimeError("public-main SDPA forward/backward qualification failed")
    return {
        "resolved": "public_main_sdpa_fallback",
        "implementation": "torch.nn.functional.scaled_dot_product_attention",
        "dense_forward_backward": True,
        "document_masking": False,
    }


def _fast_cuda_graph_gate() -> dict:
    """Capture and replay the public Fast-AttnRes operator with changed input."""
    from fast_attnres import load_fast_attnres

    op = load_fast_attnres()
    values = [torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16) for _ in range(5)]
    query = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        op(values, query, eps=float(torch.finfo(torch.bfloat16).eps), scale=1.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(values, query, eps=float(torch.finfo(torch.bfloat16).eps), scale=1.0)
    torch.cuda.synchronize()
    before = captured.clone()
    values[0].add_(0.125)
    graph.replay()
    torch.cuda.synchronize()
    changed = not torch.equal(before, captured)
    if not changed:
        raise RuntimeError("Fast-AttnRes CUDA Graph replay ignored changed input")
    return {"captured": True, "replayed_changed_input": True, "scope": "public_operator"}


def _route_gate(models: dict[str, OBPM], shape: Shape) -> dict:
    if MultiHeadAttention.flash_attn_func is not None or MultiHeadAttention.flash_attn_varlen_func is not None:
        raise RuntimeError("benchmark unexpectedly loaded generic flash-attn")
    fast = models["fast_attnres"].fast_attnres_startup_report(validate_package=True)
    expected_reads = 2 * shape.layers
    if fast["active_reads"] != expected_reads or fast["legacy_fallback_reads"]:
        raise RuntimeError(f"Fast-AttnRes route mismatch: {fast}")
    print_fast_attnres_banner(fast, is_rank_zero=True)
    return {
        "attention": "public_main_sdpa_fallback",
        "fast_attnres": fast,
        "legacy_attnres": {"active_reads": 0, "legacy_reads": expected_reads},
        "pre_norm_only": {"active_reads": 0, "legacy_reads": 0},
    }


def _correctness_gate(models: dict[str, OBPM], profile: str, shape: Shape, seed: int) -> dict:
    """Compare full-model outputs and every shared parameter gradient."""
    legacy = models["legacy_attnres"]
    fast = models["fast_attnres"]
    legacy.load_state_dict(fast.state_dict(), strict=True)
    batch = _batch(shape, seed)
    results = {}
    gradients = {}
    for name, model in (("legacy", legacy), ("fast", fast)):
        model.zero_grad(set_to_none=True)
        logits = model(
            batch["x"],
            cu_doc_len=batch["cu_doc_len"],
            max_doc_len=batch["max_doc_len"],
        )
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), batch["y"].reshape(-1)
        )
        loss.backward()
        results[name] = (logits.detach(), float(loss.detach()))
        gradients[name] = {
            key: parameter.grad.detach().clone()
            for key, parameter in model.named_parameters()
            if parameter.grad is not None
        }
    torch.testing.assert_close(results["fast"][0], results["legacy"][0], rtol=0.05, atol=0.05)
    if gradients["fast"].keys() != gradients["legacy"].keys():
        raise RuntimeError("legacy/Fast gradient key sets differ")
    worst_abs = 0.0
    for key in gradients["fast"]:
        torch.testing.assert_close(
            gradients["fast"][key], gradients["legacy"][key], rtol=0.05, atol=0.05
        )
        worst_abs = max(
            worst_abs,
            float((gradients["fast"][key].float() - gradients["legacy"][key].float()).abs().max()),
        )
    for model in (legacy, fast):
        model.zero_grad(set_to_none=True)
    return {
        "profile": profile,
        "bf16_rtol": 0.05,
        "bf16_atol": 0.05,
        "output_close": True,
        "all_parameter_gradients_close": True,
        "gradient_tensor_count": len(gradients["fast"]),
        "worst_gradient_absolute_difference": worst_abs,
        "losses": {"legacy": results["legacy"][1], "fast": results["fast"][1]},
    }


def _compile_gate(models: dict[str, OBPM], profile: str, shape: Shape, seed: int) -> tuple[dict, dict]:
    compiled = {}
    report = {}
    first = _batch(shape, seed)
    second = _batch(shape, seed + 1)
    for arm, model in models.items():
        wrapped = torch.compile(model, fullgraph=True, dynamic=False)
        with torch.no_grad():
            a = wrapped(first["x"], cu_doc_len=first["cu_doc_len"], max_doc_len=first["max_doc_len"])
            b = wrapped(second["x"], cu_doc_len=second["cu_doc_len"], max_doc_len=second["max_doc_len"])
        torch.cuda.synchronize()
        changed = not torch.equal(a, b)
        if not changed:
            raise RuntimeError(f"compiled {arm} ignored changed input")
        compiled[arm] = wrapped
        report[arm] = {
            "fullgraph": True,
            "dynamic": False,
            "mode": None,
            "changed_input": True,
            "profile": profile,
        }
    return compiled, report


def _shared_state_gate(models: dict[str, OBPM]) -> dict:
    legacy = models["legacy_attnres"].state_dict()
    fast = models["fast_attnres"].state_dict()
    if legacy.keys() != fast.keys():
        raise RuntimeError("legacy and Fast models have different state keys")
    mismatches = [key for key in legacy if not torch.equal(legacy[key], fast[key])]
    if mismatches:
        raise RuntimeError(f"legacy and Fast initial states differ: {mismatches[:3]}")
    return {"legacy_fast_identical": True, "tensor_count": len(legacy)}


def _metric_vector(rows: list[dict], profiles: tuple[str, ...]) -> np.ndarray:
    metrics = []
    for profile in profiles:
        selected = [row for row in rows if row["profile"] == profile]
        mean_ms = {
            arm: float(np.mean([row["elapsed_ms"] for row in selected if row["arm"] == arm]))
            for arm in ARMS
        }
        mean_tps = {
            arm: float(np.mean([row["tokens_per_second"] for row in selected if row["arm"] == arm]))
            for arm in ARMS
        }
        metrics.extend(
            [
                100.0 * (1.0 - mean_ms["fast_attnres"] / mean_ms["legacy_attnres"]),
                100.0 * (mean_tps["fast_attnres"] / mean_tps["pre_norm_only"] - 1.0),
                100.0 * (mean_tps["legacy_attnres"] / mean_tps["pre_norm_only"] - 1.0),
            ]
        )
    return np.asarray(metrics, dtype=np.float64)


def _report(samples: list[dict], profiles: tuple[str, ...], replicates: int, seed: int) -> dict:
    timed = [row for row in samples if row["phase"] == "timed"]
    paired = {}
    for row in timed:
        key = (row["profile"], row["seed"], row["pair_index"])
        paired.setdefault(key, []).append(row)
    for key, rows in paired.items():
        if {row["arm"] for row in rows} != set(ARMS):
            raise RuntimeError(f"incomplete paired arms for {key}")
        if len({row["input_sha256"] for row in rows}) != 1:
            raise RuntimeError(f"input mismatch within paired arms for {key}")
    point = _metric_vector(timed, profiles)
    rng = np.random.default_rng(seed)
    draws = []
    grouped = {
        (profile, run_seed): [
            row for row in timed if row["profile"] == profile and row["seed"] == run_seed
        ]
        for profile in profiles
        for run_seed in sorted({row["seed"] for row in timed})
    }
    pair_indices = {
        key: sorted({row["pair_index"] for row in rows}) for key, rows in grouped.items()
    }
    for _ in range(replicates):
        resampled = []
        for key, rows in grouped.items():
            indices = pair_indices[key]
            chosen = rng.choice(indices, size=len(indices), replace=True)
            by_pair = {}
            for row in rows:
                by_pair.setdefault(row["pair_index"], []).append(row)
            for new_pair, old_pair in enumerate(chosen):
                for row in by_pair[int(old_pair)]:
                    resampled.append({**row, "pair_index": new_pair})
        draws.append(_metric_vector(resampled, profiles))
    draw_matrix = np.stack(draws)
    max_deviation = np.max(np.abs(draw_matrix - point), axis=1)
    critical = float(np.quantile(max_deviation, 0.95))
    output = {
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "method": "seed-stratified common-index paired bootstrap",
            "interval": "simultaneous 95% max-absolute-centered interval",
            "critical_value": critical,
        },
        "profiles": {},
    }
    cursor = 0
    for profile in profiles:
        selected = [row for row in timed if row["profile"] == profile]
        arm_stats = {}
        for arm in ARMS:
            arm_rows = [row for row in selected if row["arm"] == arm]
            arm_stats[arm] = {
                "samples": len(arm_rows),
                "mean_latency_ms": float(np.mean([row["elapsed_ms"] for row in arm_rows])),
                "median_latency_ms": float(np.median([row["elapsed_ms"] for row in arm_rows])),
                "mean_tokens_per_second": float(np.mean([row["tokens_per_second"] for row in arm_rows])),
            }
        names = (
            "fast_vs_legacy_latency_reduction_percent",
            "fast_vs_pre_norm_tokens_per_second_change_percent",
            "legacy_vs_pre_norm_tokens_per_second_change_percent",
        )
        contrasts = {}
        for name in names:
            value = float(point[cursor])
            contrasts[name] = {
                "estimate": value,
                "simultaneous_95_percent_interval": [value - critical, value + critical],
            }
            cursor += 1
        output["profiles"][profile] = {"arms": arm_stats, "contrasts": contrasts}
    return output


def _source_hashes() -> dict:
    root = Path(__file__).resolve().parent
    names = ("model.py", "fast_attnres.py", "attnres_ops.py", "benchmark_fast_attnres.py")
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in names}


def _attention_ast_hash() -> str:
    tree = ast.parse(Path(__file__).with_name("model.py").read_text())
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "MultiHeadAttention"
    )
    return sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def run_benchmark(
    *,
    profiles: tuple[str, ...] = PROFILES,
    seeds: tuple[int, ...] = FULL_SEEDS,
    warmups: int = 10,
    rounds: int = 120,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 20260901,
    smoke: bool = False,
) -> dict:
    if warmups < 0 or rounds < 1 or bootstrap_replicates < 1:
        raise ValueError("warmups >= 0, rounds >= 1, bootstrap_replicates >= 1 required")
    shape = SMOKE if smoke else TARGET
    started = time.time()
    attention = _assert_public_main_attention()
    cuda_graph = _fast_cuda_graph_gate()
    samples = []
    gates = []
    route_reports = {}

    for profile_index, profile in enumerate(profiles):
        for seed in seeds:
            models = {arm: _make_model(profile, arm, shape, seed) for arm in ARMS}
            # The backend selector must not alter initialization or state layout.
            models["fast_attnres"].load_state_dict(models["legacy_attnres"].state_dict(), strict=True)
            gates.append({"profile": profile, "seed": seed, **_shared_state_gate(models)})
            route_reports[f"{profile}:{seed}"] = _route_gate(models, shape)
            gate_shape = shape if smoke else Shape(shape.layers, shape.width, shape.heads, shape.ffn, 1, 32, shape.vocab)
            # Full L/D/H/FFN and vocabulary; shortened tokens keep the oracle gate economical.
            gates.append(_correctness_gate(models, profile, gate_shape, seed + 700_001))
            compiled, compile_report = _compile_gate(
                models, profile, shape, seed + 800_001
            )
            gates.append({"profile": profile, "seed": seed, "compile": compile_report})
            optimizers = {arm: _optimizer(compiled[arm]) for arm in ARMS}
            orders = _balanced_orders(max(warmups, rounds), seed + profile_index * 100_003)

            for phase, count, seed_offset in (
                ("warmup", warmups, -100_000),
                ("timed", rounds, 0),
            ):
                for pair_index in range(count):
                    batch_seed = seed * 1_000_003 + profile_index * 100_000 + seed_offset + pair_index
                    batch = _batch(shape, batch_seed)
                    for order_index, arm in enumerate(orders[pair_index]):
                        elapsed_ms, loss = _timed_step(compiled[arm], optimizers[arm], batch)
                        samples.append(
                            {
                                "record_type": "sample",
                                "phase": phase,
                                "profile": profile,
                                "seed": seed,
                                "initial_state_seed": seed,
                                "pair_index": pair_index,
                                "order_index": order_index,
                                "arm": arm,
                                "attention": "public_main_sdpa_fallback",
                                "attnres": (
                                    "fast" if arm == "fast_attnres" else "legacy" if arm == "legacy_attnres" else "none"
                                ),
                                "input_sha256": batch["sha256"],
                                "elapsed_ms": elapsed_ms,
                                "tokens": int(shape.batch * shape.tokens),
                                "tokens_per_second": float(shape.batch * shape.tokens / (elapsed_ms / 1000.0)),
                                "loss": loss,
                            }
                        )
            del compiled, optimizers, models
            torch.cuda.empty_cache()

    timed_count = sum(row["phase"] == "timed" for row in samples)
    expected = len(profiles) * len(seeds) * rounds * len(ARMS)
    if timed_count != expected:
        raise RuntimeError(f"incomplete samples: got {timed_count}, expected {expected}")
    report = _report(samples, profiles, bootstrap_replicates, bootstrap_seed)
    manifest = {
        "protocol": "fast_attnres_only_h100_v1",
        "status": "smoke" if smoke else "full",
        "profiles": list(profiles),
        "arms": list(ARMS),
        "shape": asdict(shape),
        "dtype": "bfloat16",
        "document_masking": False,
        "seeds": list(seeds),
        "warmups": warmups,
        "rounds": rounds,
        "bootstrap_replicates": bootstrap_replicates,
        "compile": {"fullgraph": True, "dynamic": False, "mode": None},
        "max_autotune": False,
        "expected_timed_samples": expected,
        "attention": "unchanged public-main SDPA fallback",
        "criterion": "torch.nn.functional.cross_entropy(float32 logits, mean)",
    }
    attention_hash = _attention_ast_hash()
    public_main_attention_hash = os.environ.get("LR_ATTNRES_PUBLIC_MAIN_ATTENTION_SHA256", "unknown")
    provenance = {
        "source_commit": os.environ.get("LR_ATTNRES_SOURCE_COMMIT", "unknown"),
        "source_dirty": os.environ.get("LR_ATTNRES_SOURCE_DIRTY", "unknown"),
        "source_hashes": _source_hashes(),
        "attention_ast": {
            "current_sha256": attention_hash,
            "public_main_sha256": public_main_attention_hash,
            "exact_match": attention_hash == public_main_attention_hash,
        },
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("fast-attnres", "numpy", "triton")
        },
        "attention_qualification": attention,
        "fast_attnres_cuda_graph": cuda_graph,
        "routes": route_reports,
        "gates": gates,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    return {"manifest": manifest, "provenance": provenance, "report": report, "samples": samples}


def run_with_failure_record(**kwargs) -> dict:
    try:
        return {"ok": True, **run_benchmark(**kwargs)}
    except Exception as exc:
        return {
            "ok": False,
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def write_artifacts(result: dict, root: str | Path) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(root) / f"fast-attnres-h100-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("manifest", "provenance", "report", "failure"):
        if name in result:
            (run_dir / f"{name}.json").write_text(
                json.dumps(result[name], indent=2, sort_keys=True) + "\n"
            )
    if "samples" in result:
        with (run_dir / "raw_samples.jsonl").open("w") as handle:
            for row in result["samples"]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return run_dir
