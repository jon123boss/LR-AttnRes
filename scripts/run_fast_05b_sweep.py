#!/usr/bin/env python3
"""Run the high-rank 0.5B sliced Block LR-AttnRes sweep serially.

The runner is intentionally restartable.  It resumes the newest checkpoint in
each run directory, performs full-shard validation after training, and records
    all state changes atomically in sweep_state.json. Training metrics are logged
    online to W&B, and completed checkpoints are published after validation.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DATASET_DIR = Path("/dev/shm/ultrafineweb20B_gpt4")
RUNS_DIR = Path("/root/sweep-runs")
CACHE_DIR = Path("/root/sweep-cache")
STATE_PATH = RUNS_DIR / "sweep_state.json"
LOCK_PATH = RUNS_DIR / ".sweep.lock"
STOP_PATH = RUNS_DIR / "STOP"
EXPECTED_FINAL_STEP = 38146
EXPECTED_TRAIN_SHARDS = 131
DATASET_REVISION = "2d102ffbc415103c82705a227afba5dad5b9d217"
REFERENCE_WANDB_RUN = (
    "https://wandb.ai/jonnester-german-swiss-international-school-/"
    "LR-AttnRes/runs/ne0tiqb3"
)

# Rank 1024 and rank 64 were reported complete by the owner.  Higher unfinished
# ranks go first to avoid the separate worker covering the lower ranks.
HIGH_PRIORITY_JOBS = tuple(
    (n_blocks, rank)
    for rank in (768, 512, 256, 128)
    for n_blocks in (16, 8, 4)
)
LOWER_RANK_JOBS = tuple(
    (n_blocks, rank)
    for rank in (32, 16)
    for n_blocks in (16, 8, 4)
)
ALL_JOBS = HIGH_PRIORITY_JOBS + LOWER_RANK_JOBS
OWNER_REPORTED_COMPLETE = tuple(
    (n_blocks, rank)
    for rank in (1024, 64)
    for n_blocks in (4, 8, 16)
)
OWNER_REPORTED_RESULTS = {
    (4, 1024): (2.9797, "https://huggingface.co/Jonnester/LR-AttnRes-n4"),
    (8, 1024): (2.9778, "https://huggingface.co/Jonnester/LR-AttnRes-n8"),
    (16, 1024): (2.9673, "https://huggingface.co/Jonnester/LR-AttnRes-n16"),
    (4, 64): (2.9533, "https://huggingface.co/Jonnester/LR-AttnRes-tail-r64-n4"),
    (8, 64): (2.948, "https://huggingface.co/Jonnester/LR-AttnRes-tail-r64-n8"),
    (16, 64): (2.9494, "https://huggingface.co/Jonnester/LR-AttnRes-tail-r64-n16"),
}
PUBLIC_LEGACY_RESULTS = {
    (4, 32): {
        "validation_loss": 2.957,
        "huggingface_url": "https://huggingface.co/Jonnester/LR-AttnRes-sliced-05b-n4-r32",
        "backend": "legacy",
    },
    (8, 32): {
        "validation_loss": 2.9551,
        "huggingface_url": "https://huggingface.co/Jonnester/LR-AttnRes-tail-r32-n8",
        "backend": "legacy",
    },
    (16, 32): {
        "validation_loss": 2.9543,
        "huggingface_url": "https://huggingface.co/Jonnester/LR-AttnRes-n16-r32",
        "backend": "legacy",
    },
    (8, 128): {
        "validation_loss": 2.947662410346825,
        "huggingface_url": (
            "https://huggingface.co/Jonnester/LR-AttnRes-sliced-05b-n4-r32/"
            "tree/main/experiments/sliced-05b-n8-r128"
        ),
        "backend": "legacy",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)


def load_state() -> dict:
    if STATE_PATH.exists():
        with STATE_PATH.open(encoding="utf-8") as source:
            state = json.load(source)
        for job in state.get("jobs", {}).values():
            if job.get("status") == "observed_complete" and job.get("backend") == "legacy":
                job["status"] = "observed_legacy_result"
                job["qualifies_fast_sweep"] = False
        return state
    state = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "dataset_revision": DATASET_REVISION,
        "kernel": "fast-attnres==2.0.1",
        "compile_mode": "fullgraph-static-no-cudagraphs",
        "jobs": {},
    }
    for n_blocks, rank in OWNER_REPORTED_COMPLETE:
        validation_loss, huggingface_url = OWNER_REPORTED_RESULTS[(n_blocks, rank)]
        state["jobs"][job_name(n_blocks, rank)] = {
            "n": n_blocks,
            "rank": rank,
            "status": "observed_complete",
            "validation_loss": validation_loss,
            "huggingface_url": huggingface_url,
        }
    for (n_blocks, rank), result in PUBLIC_LEGACY_RESULTS.items():
        state["jobs"][job_name(n_blocks, rank)] = {
            "n": n_blocks,
            "rank": rank,
            "status": "observed_legacy_result",
            "qualifies_fast_sweep": False,
            **result,
        }
    return state


def save_state(state: dict) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_write_json(STATE_PATH, state)


def job_name(n_blocks: int, rank: int) -> str:
    return f"sliced-fast-05b-n{n_blocks}-r{rank}"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"ckpt_step:(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def newest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("ckpt_step:*.pt"), key=checkpoint_step)
    return checkpoints[-1] if checkpoints else None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def preflight() -> dict:
    if not PYTHON.is_file():
        raise RuntimeError(f"Pinned environment is missing: {PYTHON}")
    train_shards = sorted(DATASET_DIR.glob("finewebedu_train_*.bin"))
    val_shards = sorted(DATASET_DIR.glob("finewebedu_val_*.bin"))
    if len(train_shards) != EXPECTED_TRAIN_SHARDS or len(val_shards) != 1:
        raise RuntimeError(
            f"Dataset is incomplete: expected {EXPECTED_TRAIN_SHARDS} train + 1 val, "
            f"found {len(train_shards)} train + {len(val_shards)} val"
        )
    wrong_sizes = [path for path in train_shards + val_shards if path.stat().st_size != 400_000_000]
    if wrong_sizes:
        raise RuntimeError(f"Dataset contains unexpected shard sizes: {wrong_sizes[:3]}")
    probe = subprocess.check_output(
        [
            str(PYTHON),
            "-c",
            (
                "import json, torch; from importlib.metadata import version; "
                "print(json.dumps({'torch': torch.__version__, "
                "'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), "
                "'fast_attnres': version('fast-attnres')}))"
            ),
        ],
        cwd=ROOT,
        text=True,
    )
    runtime = json.loads(probe)
    if runtime["torch"] not in {"2.10.0+cu130", "2.9.0+cu126"} or runtime["fast_attnres"] != "2.0.1":
        raise RuntimeError(f"Unexpected runtime: {runtime}")
    return {
        **runtime,
        "train_shards": len(train_shards),
        "val_shards": len(val_shards),
        "dataset_bytes": sum(path.stat().st_size for path in train_shards + val_shards),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_diff_sha256": sha256_bytes(
            subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
        ),
    }


def _wandb_cell(run) -> tuple[int | None, int | None]:
    config = dict(run.config or {})
    try:
        n_blocks = int(config.get("attnres_num_blocks"))
        rank = int(config.get("lrid_rank"))
    except (TypeError, ValueError):
        return None, None
    if not config.get("use_lrid") or config.get("attnres_type") != "block":
        return None, None
    if not config.get("lrid_key_from_output_tail"):
        return None, None
    if config.get("attnres_block_average") is not False:
        return None, None
    if config.get("attnres_block_count_prior") is not False:
        return None, None
    if config.get("lrid_use_logit_scale") is not False:
        return None, None
    return n_blocks, rank


def _wandb_fast_qualified(run) -> bool:
    config = dict(run.config or {})
    return (
        config.get("attnres_backend") == "fast"
        and config.get("torch_compile_max_autotune") is False
        and config.get("torch_compile_fullgraph") is True
        and config.get("torch_compile_dynamic") is False
    )


def reconcile_external_lower_runs(state: dict) -> None:
    """Defer lower cells occupied by another worker and record finished Fast runs."""
    import wandb

    api = wandb.Api(timeout=120)
    entity = api.default_entity
    if not entity:
        raise RuntimeError("W&B has no default entity; cannot guard the lower-rank queue.")
    by_cell: dict[tuple[int, int], list] = {}
    for run in api.runs(f"{entity}/LR-AttnRes", order="-created_at", per_page=100):
        cell = _wandb_cell(run)
        if cell in LOWER_RANK_JOBS:
            by_cell.setdefault(cell, []).append(run)

    external_statuses = {"deferred_external_running", "external_fast_finished_pending_import"}
    for n_blocks, rank in LOWER_RANK_JOBS:
        name = job_name(n_blocks, rank)
        job = state["jobs"].setdefault(name, {"n": n_blocks, "rank": rank})
        if job.get("status") in {"complete", "observed_complete"}:
            continue
        runs = by_cell.get((n_blocks, rank), [])
        finished_fast = next(
            (
                run
                for run in runs
                if run.state == "finished"
                and _wandb_fast_qualified(run)
                and int(dict(run.summary or {}).get("tokens_processed", 0)) >= 9_999_745_024
            ),
            None,
        )
        live = next((run for run in runs if run.state in {"running", "pending"}), None)
        if finished_fast is not None:
            job.update(
                {
                    "status": "external_fast_finished_pending_import",
                    "external_wandb_url": finished_fast.url,
                    "external_wandb_run_id": finished_fast.id,
                    "external_backend": dict(finished_fast.config or {}).get("attnres_backend"),
                }
            )
        elif live is not None:
            job.update(
                {
                    "status": "deferred_external_running",
                    "external_wandb_url": live.url,
                    "external_wandb_run_id": live.id,
                    "external_backend": dict(live.config or {}).get("attnres_backend"),
                }
            )
        elif job.get("status") in external_statuses:
            job["status"] = "pending"
            job.pop("external_wandb_url", None)
            job.pop("external_wandb_run_id", None)
            job.pop("external_backend", None)
    state["lower_rank_wandb_audited_at_utc"] = utc_now()


def command_for(n_blocks: int, rank: int, run_dir: Path, resume: bool) -> list[str]:
    command = [
        str(PYTHON),
        "train.py",
        "--out_dir",
        str(run_dir),
        "--dataset_dir",
        str(DATASET_DIR),
        "--no-full_run",
        "--wandb_log",
        "--no-wandb_log_checkpoints",
        "--wandb_project",
        "LR-AttnRes",
        "--wandb_run_name",
        job_name(n_blocks, rank),
        "--use_lrid",
        "--attnres_type",
        "block",
        "--attnres_num_blocks",
        str(n_blocks),
        "--no-attnres_block_average",
        "--no-attnres_block_count_prior",
        "--lrid_rank",
        str(rank),
        "--lrid_key_from_output_tail",
        "--no-lrid_use_logit_scale",
        "--attnres_backend",
        "fast",
        "--no-torch_compile_max_autotune",
        "--no-torch_compile_cudagraphs",
        "--ckpt_interval",
        "5000",
        "--max_local_checkpoints",
        "1",
        "--init_from",
        "resume" if resume else "scratch",
    ]
    if resume:
        command.extend(["--ckpt_file_name", ""])
    return command


def run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{utc_now()}] exec: {json.dumps(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return process.wait()


def publish_pending() -> int:
    environment = os.environ.copy()
    environment["HF_XET_HIGH_PERFORMANCE"] = "1"
    command = [str(PYTHON), str(ROOT / "scripts" / "sync_publish_fast_05b.py")]
    return run_logged(command, RUNS_DIR / "publish.log", environment)


def evaluation_loss(results_path: Path) -> float:
    structured_path = results_path.with_suffix(".json")
    with structured_path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not payload.get("complete"):
        raise RuntimeError(f"Evaluation did not report complete: {structured_path}")
    checkpoint_results = payload.get("checkpoint_results", {})
    if len(checkpoint_results) != 1:
        raise RuntimeError(f"Expected one checkpoint result: {structured_path}")
    result = next(iter(checkpoint_results.values()))
    provenance = result.get("provenance", {})
    fast_report = provenance.get("fast_attnres", {})
    if fast_report.get("resolved_backend") != "fast-attnres":
        raise RuntimeError(f"Evaluation did not use Fast-AttnRes: {fast_report}")
    total_reads = int(fast_report.get("total_reads", 0))
    if total_reads <= 0 or int(fast_report.get("fast_reads", -1)) != total_reads:
        raise RuntimeError(f"Not every evaluation route used Fast-AttnRes: {fast_report}")
    if int(fast_report.get("legacy_reads", -1)) != 0:
        raise RuntimeError(f"Evaluation retained legacy residual routes: {fast_report}")
    compile_report = provenance.get("torch_compile", {})
    expected_compile = {
        "enabled": True,
        "mode": None,
        "fullgraph": True,
        "dynamic": False,
    }
    if compile_report != expected_compile:
        raise RuntimeError(
            f"Evaluation compile contract differs from {expected_compile}: {compile_report}"
        )
    return float(result["validation_loss"]["loss"])


def run_job(state: dict, n_blocks: int, rank: int, max_retries: int) -> None:
    name = job_name(n_blocks, rank)
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    job_state = state["jobs"].setdefault(name, {"n": n_blocks, "rank": rank})
    checkpoint = newest_checkpoint(run_dir)
    if checkpoint is not None and checkpoint_step(checkpoint) >= EXPECTED_FINAL_STEP:
        job_state["status"] = "trained"
        job_state["checkpoint"] = str(checkpoint)
    else:
        for attempt in range(1, max_retries + 1):
            checkpoint = newest_checkpoint(run_dir)
            resume = checkpoint is not None
            command = command_for(n_blocks, rank, run_dir, resume)
            launch = {
                "name": name,
                "n": n_blocks,
                "rank": rank,
                "backend": "fast",
                "compile_mode": "fullgraph-static-no-cudagraphs",
                "torch_compile_fullgraph": True,
                "torch_compile_dynamic": False,
                "dataset_revision": DATASET_REVISION,
                "reference_wandb_run": REFERENCE_WANDB_RUN,
                "intentional_recipe_differences": {
                    "attnres_backend": "legacy -> fast",
                    "torch_compile": "max-autotune -> fullgraph=True,dynamic=False",
                },
                "started_at_utc": utc_now(),
                "attempt": attempt,
                "resume": resume,
                "command": command,
                "runtime": state["runtime"],
            }
            atomic_write_json(run_dir / "launch.json", launch)
            job_state.update(
                {
                    "status": "training",
                    "attempt": attempt,
                    "started_at_utc": launch["started_at_utc"],
                    "run_dir": str(run_dir),
                }
            )
            save_state(state)
            environment = os.environ.copy()
            environment.update(
                {
                    "WANDB_MODE": "online",
                    "WANDB_DIR": str(run_dir),
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "HF_XET_HIGH_PERFORMANCE": "1",
                    "TORCH_COMPILE_CACHE_DIR": str(CACHE_DIR / name),
                }
            )
            return_code = run_logged(command, run_dir / "train.log", environment)
            checkpoint = newest_checkpoint(run_dir)
            if return_code == 0 and checkpoint is not None and checkpoint_step(checkpoint) >= EXPECTED_FINAL_STEP:
                break
            job_state.update(
                {
                    "status": "retrying" if attempt < max_retries else "failed",
                    "return_code": return_code,
                    "latest_checkpoint": str(checkpoint) if checkpoint else None,
                    "failed_at_utc": utc_now(),
                }
            )
            save_state(state)
            if attempt == max_retries:
                raise RuntimeError(f"{name} failed after {max_retries} attempts")
            time.sleep(min(60 * attempt, 300))

        job_state.update(
            {
                "status": "trained",
                "checkpoint": str(checkpoint),
                "trained_at_utc": utc_now(),
            }
        )
        save_state(state)

    results_path = run_dir / "evaluation.txt"
    if not results_path.with_suffix(".json").exists():
        job_state["status"] = "evaluating"
        save_state(state)
        eval_environment = os.environ.copy()
        eval_environment["TORCH_COMPILE_CACHE_DIR"] = str(CACHE_DIR / name / "eval")
        eval_command = [
            str(PYTHON),
            "run_eval.py",
            "--ckpts",
            str(checkpoint),
            "--validation-only",
            "--torch_compile",
            "--results-file",
            str(results_path),
        ]
        return_code = run_logged(eval_command, run_dir / "eval.log", eval_environment)
        if return_code:
            job_state.update({"status": "eval_failed", "eval_return_code": return_code})
            save_state(state)
            raise RuntimeError(f"Evaluation failed for {name} with code {return_code}")

    loss = evaluation_loss(results_path)
    job_state.update(
        {
            "status": "complete_pending_sync_and_upload",
            "backend": "fast-attnres",
            "qualifies_fast_sweep": True,
            "validation_loss": loss,
            "completed_at_utc": utc_now(),
            "evaluation": str(results_path.with_suffix(".json")),
        }
    )
    save_state(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--only", choices=[job_name(n, r) for n, r in ALL_JOBS])
    parser.add_argument(
        "--include-lower",
        action="store_true",
        help="After the high-priority queue, include missing r=32 and r=16 cells.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another sweep controller holds the sweep lock.")
        lock.write(str(os.getpid()))
        lock.flush()
        state = load_state()
        state["runtime"] = preflight()
        state["compile_mode"] = "fullgraph-static-no-cudagraphs"
        if args.include_lower:
            reconcile_external_lower_runs(state)
        save_state(state)
        publish_pending()
        state = load_state()
        selected_jobs = HIGH_PRIORITY_JOBS
        if args.include_lower:
            selected_jobs += LOWER_RANK_JOBS
        if args.only:
            selected_jobs = tuple((n, r) for n, r in ALL_JOBS if job_name(n, r) == args.only)
        for n_blocks, rank in selected_jobs:
            if STOP_PATH.exists():
                state["controller_status"] = "stopped_by_sentinel"
                save_state(state)
                return 0
            name = job_name(n_blocks, rank)
            if state["jobs"].get(name, {}).get("status") in {
                "complete_pending_sync_and_upload",
                "complete",
                "observed_complete",
                "deferred_external_running",
                "external_fast_finished_pending_import",
            }:
                continue
            run_job(state, n_blocks, rank, args.max_retries)
            publish_return_code = publish_pending()
            state = load_state()
            if publish_return_code:
                state["last_publish_return_code"] = publish_return_code
                save_state(state)
        publish_pending()
        state = load_state()
        waiting_external = any(
            job.get("status") in {
                "deferred_external_running",
                "external_fast_finished_pending_import",
            }
            for job in state["jobs"].values()
        )
        if args.include_lower and waiting_external:
            state["controller_status"] = "lower_queue_waiting_external_runs"
        else:
            state["controller_status"] = (
                "full_queue_complete" if args.include_lower else "high_rank_queue_complete"
            )
        save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
