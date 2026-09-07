#!/usr/bin/env python3
"""Sync completed sweep runs to W&B and publish public Hugging Face models."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from huggingface_hub import HfApi
import wandb


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path("/root/sweep-runs")
CACHE_DIR = Path("/root/sweep-cache")
STATE_PATH = RUNS_DIR / "sweep_state.json"
NAMESPACE = "Jonnester"
PROJECT = "LR-AttnRes"
TARGET_BLOCKS = (4, 8, 16)
TARGET_RANKS = (16, 32, 64, 128, 256, 512, 768, 1024)
REFERENCE_WANDB_RUN = (
    "https://wandb.ai/jonnester-german-swiss-international-school-/"
    "LR-AttnRes/runs/ne0tiqb3"
)


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


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)


def write_results_ledger(state: dict) -> None:
    fields = (
        "n",
        "rank",
        "status",
        "backend",
        "validation_loss",
        "wandb_url",
        "huggingface_url",
    )
    rows = []
    for rank in reversed(TARGET_RANKS):
        for n_blocks in reversed(TARGET_BLOCKS):
            name = f"sliced-fast-05b-n{n_blocks}-r{rank}"
            job = state.get("jobs", {}).get(name, {})
            rows.append(
                {
                    "n": n_blocks,
                    "rank": rank,
                    "status": job.get("status", "pending"),
                    "backend": job.get("backend", ""),
                    "validation_loss": job.get("validation_loss", ""),
                    "wandb_url": job.get("wandb_url", ""),
                    "huggingface_url": job.get("huggingface_url", ""),
                }
            )

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(RUNS_DIR / "results.csv", csv_buffer.getvalue())

    markdown = [
        "# LR-AttnRes 0.5B sweep results",
        "",
        f"Updated: {utc_now()}",
        "",
        "| n | rank | status | backend | validation loss | W&B | Hugging Face |",
        "|---:|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        wandb_link = f"[run]({row['wandb_url']})" if row["wandb_url"] else ""
        hf_link = f"[model]({row['huggingface_url']})" if row["huggingface_url"] else ""
        markdown.append(
            f"| {row['n']} | {row['rank']} | {row['status']} | {row['backend']} | "
            f"{row['validation_loss']} | {wandb_link} | {hf_link} |"
        )
    atomic_write_text(RUNS_DIR / "results.md", "\n".join(markdown) + "\n")


def migrate_legacy_observations(state: dict) -> None:
    """Keep legacy checkpoints visible without counting them as Fast results."""
    for job in state.get("jobs", {}).values():
        if job.get("status") == "observed_complete" and job.get("backend") == "legacy":
            job["status"] = "observed_legacy_result"
            job["qualifies_fast_sweep"] = False


def checkpoint_from_job(job: dict) -> Path:
    checkpoint = Path(job["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def load_evaluation(job: dict) -> tuple[dict, dict]:
    evaluation_path = Path(job["evaluation"])
    with evaluation_path.open(encoding="utf-8") as source:
        evaluation = json.load(source)
    result = next(iter(evaluation["checkpoint_results"].values()))
    return evaluation, result


def load_runtime_qualification(run_dir: Path) -> dict:
    path = run_dir / "runtime_qualification.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_model_card(run_dir: Path, name: str, job: dict, result: dict) -> Path:
    validation = result["validation_loss"]
    provenance = result["provenance"]
    fast = provenance["fast_attnres"]
    qualification = load_runtime_qualification(run_dir)
    benchmark_lines = []
    if qualification:
        benchmark_lines = [
            f"- Steady median throughput: `{qualification['steady_median_tokens_per_second']:.2f}` tokens/s",
            (
                "- Controlled Fast routed-read speedup over exact compiled legacy: "
                f"`{qualification['fast_read_speedup_over_exact_legacy']:.3f}x`"
            ),
            f"- Compiled model graphs: `{qualification['model_forward_graph_count']}` forward, "
            f"`{qualification['model_backward_graph_count']}` backward",
        ]
    card = run_dir / "README.md"
    card.write_text(
        "\n".join(
            [
                f"# {name}",
                "",
                "0.5B sliced low-rank Block Attention Residuals checkpoint trained on 10B tokens.",
                "",
                f"- Block count: `{job['n']}`",
                f"- Routing rank: `{job['rank']}`",
                f"- Full validation loss: `{validation['loss']:.10f}`",
                f"- Validation tokens: `{validation['tokens']}`",
                f"- Checkpoint step: `{provenance['checkpoint_step']}`",
                f"- Training tokens: `{provenance['checkpoint_tokens_processed']}`",
                f"- Attention-residual backend: `{fast['resolved_backend']}`",
                f"- Fast-AttnRes version: `{fast['version']}`",
                f"- Checkpoint SHA256: `{provenance['checkpoint_sha256']}`",
                "- Compile mode: `fullgraph=True`, `dynamic=False`, CUDA graphs disabled",
                *benchmark_lines,
                f"- W&B run: {job['wandb_url']}",
                f"- Reference recipe: {REFERENCE_WANDB_RUN}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return card


def update_wandb(
    run_dir: Path,
    api: wandb.Api,
    entity: str,
    job: dict,
    result: dict,
) -> str:
    offline_runs = sorted((run_dir / "wandb").glob("offline-run-*"))
    for offline_run in offline_runs:
        subprocess.run(
            [str(ROOT / ".venv" / "bin" / "wandb"), "sync", str(offline_run)],
            cwd=ROOT,
            check=True,
        )
    run_id = (run_dir / "wandb_run_id.txt").read_text(encoding="utf-8").strip()
    run = None
    for attempt in range(6):
        try:
            run = api.run(f"{entity}/{PROJECT}/{run_id}")
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    validation = result["validation_loss"]
    provenance = result["provenance"]
    run.summary["full_validation/loss"] = float(validation["loss"])
    run.summary["full_validation/tokens"] = int(validation["tokens"])
    run.summary["full_validation/batches"] = int(validation["batches"])
    run.summary["full_validation/checkpoint_sha256"] = provenance["checkpoint_sha256"]
    run.summary["full_validation/backend"] = provenance["fast_attnres"]["resolved_backend"]
    run.summary["sweep/n"] = int(job["n"])
    run.summary["sweep/r"] = int(job["rank"])
    run.summary["sweep/reference_run"] = REFERENCE_WANDB_RUN
    run.summary["compile/fullgraph"] = True
    run.summary["compile/dynamic"] = False
    run.summary["compile/max_autotune"] = False
    qualification = load_runtime_qualification(run_dir)
    if qualification:
        run.summary["sweep/runtime_qualified"] = True
        run.summary["sweep/fast_active_reads"] = int(qualification["fast_active_reads"])
        run.summary["sweep/legacy_fallback_reads"] = int(qualification["legacy_fallback_reads"])
        run.summary["sweep/static_cu_seqlens_size"] = int(qualification["static_cu_seqlens_size"])
        run.summary["sweep/model_forward_graph_count"] = int(qualification["model_forward_graph_count"])
        run.summary["sweep/model_backward_graph_count"] = int(qualification["model_backward_graph_count"])
        run.summary["sweep/steady_median_tokens_per_second"] = float(
            qualification["steady_median_tokens_per_second"]
        )
        run.summary["sweep/fast_read_speedup_over_exact_legacy"] = float(
            qualification["fast_read_speedup_over_exact_legacy"]
        )
    run.summary.update()
    return run.url


def publish_hf(hf: HfApi, run_dir: Path, name: str, job: dict, result: dict) -> str:
    repo_id = f"{NAMESPACE}/LR-AttnRes-{name}"
    hf.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    card = write_model_card(run_dir, name, job, result)
    uploads = [
        (checkpoint_from_job(job), "final_model.pt"),
        (run_dir / "launch.json", "launch.json"),
        (Path(job["evaluation"]), "evaluation.json"),
        (card, "README.md"),
    ]
    qualification_path = run_dir / "runtime_qualification.json"
    if qualification_path.is_file():
        uploads.append((qualification_path, "runtime_qualification.json"))
        qualification = load_runtime_qualification(run_dir)
        benchmark_path = Path(qualification.get("benchmark_local_path", ""))
        if benchmark_path.is_file():
            uploads.append((benchmark_path, "fast_vs_legacy_read_benchmark.json"))
    for local_path, remote_path in uploads:
        hf.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )
    model = hf.model_info(repo_id=repo_id)
    if model.private:
        raise RuntimeError(f"Published repository is unexpectedly private: {repo_id}")
    expected_files = {remote_path for _, remote_path in uploads}
    remote_files = set(hf.list_repo_files(repo_id=repo_id, repo_type="model"))
    missing_files = sorted(expected_files - remote_files)
    if missing_files:
        raise RuntimeError(f"Published repository is missing files: {missing_files}")
    return f"https://huggingface.co/{repo_id}"


def cleanup_published_local_artifacts(name: str, job: dict) -> dict:
    """Reclaim bounded local storage after the public copy is verified."""
    checkpoint = checkpoint_from_job(job)
    checkpoint_size = checkpoint.stat().st_size
    checkpoint.unlink()
    cache_dir = CACHE_DIR / name
    cache_size = 0
    if cache_dir.is_dir():
        cache_size = sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())
        shutil.rmtree(cache_dir)
    return {
        "local_checkpoint_deleted_at_utc": utc_now(),
        "local_checkpoint_deleted_path": str(checkpoint),
        "local_checkpoint_deleted_bytes": checkpoint_size,
        "local_compile_cache_deleted_bytes": cache_size,
    }


def main() -> int:
    wandb_api = wandb.Api(timeout=120)
    entity = wandb_api.default_entity
    if not entity:
        raise RuntimeError("W&B did not return a default entity for the configured account.")
    hf = HfApi()
    identity = hf.whoami()
    if not identity.get("name"):
        raise RuntimeError("Hugging Face did not return an identity for the supplied token.")

    if not STATE_PATH.exists():
        return 0
    with STATE_PATH.open(encoding="utf-8") as source:
        state = json.load(source)
    migrate_legacy_observations(state)
    atomic_write_json(STATE_PATH, state)
    write_results_ledger(state)
    for name, job in state["jobs"].items():
        if job.get("status") not in {"complete_pending_sync_and_upload", "sync_failed"}:
            continue
        run_dir = Path(job["run_dir"])
        try:
            _, result = load_evaluation(job)
            job["wandb_url"] = update_wandb(run_dir, wandb_api, entity, job, result)
            job["huggingface_url"] = publish_hf(hf, run_dir, name, job, result)
            job["status"] = "complete"
            job["backend"] = result["provenance"]["fast_attnres"]["resolved_backend"]
            job["checkpoint_sha256"] = result["provenance"]["checkpoint_sha256"]
            job["checkpoint_remote_file"] = "final_model.pt"
            job["qualifies_fast_sweep"] = True
            job["published_at_utc"] = utc_now()
        except Exception as error:
            job["status"] = "sync_failed"
            job["sync_error"] = f"{type(error).__name__}: {error}"
            atomic_write_json(STATE_PATH, state)
            write_results_ledger(state)
            raise
        atomic_write_json(STATE_PATH, state)
        write_results_ledger(state)
        try:
            job.update(cleanup_published_local_artifacts(name, job))
            job.pop("checkpoint", None)
            job.pop("local_cleanup_error", None)
        except Exception as error:
            # Upload completion is authoritative even if local reclamation
            # fails. Preserve the public result and retry cleanup manually.
            job["local_cleanup_error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(STATE_PATH, state)
        write_results_ledger(state)
    write_results_ledger(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
