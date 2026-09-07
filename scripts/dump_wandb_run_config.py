#!/usr/bin/env python3
"""Dump selected LR-AttnRes W&B run configurations for recipe comparison."""

from __future__ import annotations

import argparse
import json

import wandb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    api = wandb.Api(timeout=120)
    entity = api.default_entity
    output = {}
    for run_id in args.run_ids:
        run = api.run(f"{entity}/LR-AttnRes/{run_id}")
        output[run_id] = {
            "name": run.name,
            "state": run.state,
            "url": run.url,
            "config": dict(run.config or {}),
            "summary": dict(run.summary or {}),
        }
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
