#!/usr/bin/env python3
"""Sync completed sweep runs to W&B and publish public Hugging Face models."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from huggingface_hub import HfApi
import wandb


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path("/root/sweep-runs")
STATE_PATH = RUNS_DIR / "sweep_state.json"
NAMESPACE = "Jonnester"
PROJECT = "LR-AttnRes"
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


def write_model_card(run_dir: Path, name: str, job: dict, result: dict) -> Path:
    validation = result["validation_loss"]
    provenance = result["provenance"]
    fast = provenance["fast_attnres"]
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
    return f"https://huggingface.co/{repo_id}"


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
    for name, job in state["jobs"].items():
        if job.get("status") not in {"complete_pending_sync_and_upload", "sync_failed"}:
            continue
        run_dir = Path(job["run_dir"])
        try:
            _, result = load_evaluation(job)
            job["wandb_url"] = update_wandb(run_dir, wandb_api, entity, job, result)
            job["huggingface_url"] = publish_hf(hf, run_dir, name, job, result)
            job["status"] = "complete"
            job["published_at_utc"] = utc_now()
        except Exception as error:
            job["status"] = "sync_failed"
            job["sync_error"] = f"{type(error).__name__}: {error}"
            atomic_write_json(STATE_PATH, state)
            raise
        atomic_write_json(STATE_PATH, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
