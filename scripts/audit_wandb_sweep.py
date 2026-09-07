#!/usr/bin/env python3
"""Print W&B runs that may occupy a cell in the 0.5B LR-AttnRes sweep."""

from __future__ import annotations

import json
import re

import wandb


PROJECT = "LR-AttnRes"
TARGET_BLOCKS = {4, 8, 16}
TARGET_RANKS = {16, 32, 64, 128, 256, 512, 768, 1024}


def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_cell(name: str, config: dict) -> tuple[int | None, int | None]:
    n_blocks = integer(config.get("attnres_num_blocks"))
    rank = integer(config.get("lrid_rank"))
    if n_blocks is None:
        match = re.search(r"(?:^|[-_])n(?:blocks?)?[-_]?([0-9]+)(?:$|[-_])", name.lower())
        n_blocks = integer(match.group(1)) if match else None
    if rank is None:
        match = re.search(r"(?:^|[-_])r(?:ank)?[-_]?([0-9]+)(?:$|[-_])", name.lower())
        rank = integer(match.group(1)) if match else None
    return n_blocks, rank


def summary_value(summary: dict, *keys: str):
    for key in keys:
        if key in summary:
            return summary[key]
    return None


def main() -> int:
    api = wandb.Api(timeout=120)
    entity = api.default_entity
    if not entity:
        raise RuntimeError("The authenticated W&B account has no default entity.")
    selected = []
    for run in api.runs(f"{entity}/{PROJECT}", order="-created_at", per_page=100):
        config = dict(run.config or {})
        summary = dict(run.summary or {})
        n_blocks, rank = infer_cell(run.name or "", config)
        is_target_cell = n_blocks in TARGET_BLOCKS and rank in TARGET_RANKS
        is_lrid_name = any(part in (run.name or "").lower() for part in ("lrid", "attnres", "sliced"))
        if not is_target_cell and not is_lrid_name:
            continue
        selected.append(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "url": run.url,
                "n": n_blocks,
                "rank": rank,
                "backend": config.get("attnres_backend"),
                "compile_max_autotune": config.get("torch_compile_max_autotune"),
                "compile_cudagraphs": config.get("torch_compile_cudagraphs"),
                "tokens_processed": summary_value(summary, "tokens_processed", "_step"),
                "validation_loss": summary_value(
                    summary,
                    "full_validation/loss",
                    "full_validation_loss",
                    "val/loss",
                ),
            }
        )
    print(json.dumps({"entity": entity, "project": PROJECT, "runs": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
