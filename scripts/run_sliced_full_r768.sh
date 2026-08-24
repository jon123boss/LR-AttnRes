#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY to log the full run to W&B.}"
: "${HF_TOKEN:?Set HF_TOKEN so the final checkpoint can be uploaded.}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

# This source tree is pinned to the exact commit used by W&B run t8tdr5d0.
# Its recorded diff is reproduced in train.py, with only rank 512 -> 768.
exec python3 train.py \
  --init_from scratch \
  --full_run \
  --full_run_hf_repo_id Jonnester/LR-AttnRes-tail-r768
