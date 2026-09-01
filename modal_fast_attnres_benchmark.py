#!/usr/bin/env python3
"""Run the Fast-AttnRes-only benchmark on an H100 through Modal 1.5.4."""

from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

import modal


REPO_ROOT = Path(__file__).resolve().parent
PROFILES = ("full_standard_r1024", "full_output_tail_r64")


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _public_main_attention_hash() -> str:
    try:
        source = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show", "origin/main:model.py"], text=True
        )
        tree = ast.parse(source)
        node = next(
            item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "MultiHeadAttention"
        )
        return sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    except (OSError, subprocess.CalledProcessError, StopIteration, SyntaxError):
        return "unknown"


def _write_artifacts(result: dict, root: Path) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = root / f"fast-attnres-h100-{timestamp}"
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


app = modal.App("lr-attnres-fast-only-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.13.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .uv_pip_install("fast-attnres==1.0.0", "numpy==2.2.6")
    .env(
        {
            "PYTHONPATH": "/root/LR-AttnRes",
            "LR_ATTNRES_SOURCE_COMMIT": _git_value("rev-parse", "HEAD"),
            "LR_ATTNRES_PUBLIC_MAIN_ATTENTION_SHA256": _public_main_attention_hash(),
            "LR_ATTNRES_SOURCE_DIRTY": (
                "true" if _git_value("status", "--porcelain") not in {"", "unknown"} else "false"
            ),
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/LR-AttnRes",
        ignore=[".git/**", ".pytest_cache/**", "**/__pycache__/**", "benchmark_artifacts/**"],
    )
)


@app.function(image=image, gpu="H100", timeout=24 * 60 * 60)
def benchmark_remote(
    profiles: tuple[str, ...],
    seeds: tuple[int, ...],
    warmups: int,
    rounds: int,
    bootstrap_replicates: int,
    smoke: bool,
):
    os.chdir("/root/LR-AttnRes")
    from benchmark_fast_attnres import run_with_failure_record

    return run_with_failure_record(
        profiles=profiles,
        seeds=seeds,
        warmups=warmups,
        rounds=rounds,
        bootstrap_replicates=bootstrap_replicates,
        smoke=smoke,
    )


@app.function(image=image, timeout=30 * 60)
def aggregate_remote(
    parts: list[dict],
    profiles: tuple[str, ...],
    seeds: tuple[int, ...],
    warmups: int,
    rounds: int,
    bootstrap_replicates: int,
):
    os.chdir("/root/LR-AttnRes")
    from benchmark_fast_attnres import _report

    failed = [part for part in parts if not part["ok"]]
    if failed:
        return {
            "ok": False,
            "failure": {
                "type": "ProfileSeedBenchmarkFailure",
                "message": "one or more profile/seed workers failed",
                "workers": [part.get("failure") for part in failed],
            },
        }
    samples = [row for part in parts for row in part["samples"]]
    manifest = dict(parts[0]["manifest"])
    manifest.update(
        {
            "profiles": list(profiles),
            "seeds": list(seeds),
            "warmups": warmups,
            "rounds": rounds,
            "bootstrap_replicates": bootstrap_replicates,
            "expected_timed_samples": len(profiles) * len(seeds) * rounds * 3,
        }
    )
    return {
        "ok": True,
        "manifest": manifest,
        "provenance": {
            "parallel_profile_seed_workers": True,
            "workers": [part["provenance"] for part in parts],
        },
        "report": _report(samples, profiles, bootstrap_replicates, 20260901),
        "samples": samples,
    }


@app.local_entrypoint()
def main(
    profile: str = "all",
    seeds: str = "20260827,20260903,20260911",
    warmups: int = 10,
    rounds: int = 120,
    bootstrap_replicates: int = 20_000,
    smoke: bool = False,
    artifact_root: str = "benchmark_artifacts",
):
    selected_profiles = PROFILES if profile == "all" else (profile,)
    selected_seeds = tuple(int(value) for value in seeds.split(",") if value.strip())
    calls = [
        benchmark_remote.spawn(
            (selected_profile,),
            (selected_seed,),
            warmups,
            rounds,
            bootstrap_replicates,
            smoke,
        )
        for selected_profile in selected_profiles
        for selected_seed in selected_seeds
    ]
    parts = [call.get() for call in calls]
    result = aggregate_remote.remote(
        parts,
        selected_profiles,
        selected_seeds,
        warmups,
        rounds,
        bootstrap_replicates,
    )
    run_dir = _write_artifacts(result, REPO_ROOT / artifact_root)
    summary = {
        "ok": result["ok"],
        "artifact_dir": str(run_dir),
        "report": result.get("report"),
        "failure": result.get("failure"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
