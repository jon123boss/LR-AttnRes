#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY to log the full run to W&B.}"
: "${HF_TOKEN:?Set HF_TOKEN so the final checkpoint can be uploaded.}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

exec python3 train.py \
  --init_from scratch \
  --wandb_project LR-AttnRes \
  --wandb_run_name "tail r=768" \
  --torch-max-autotune \
  --full_run \
  --full_run_hf_repo_id Jonnester/LR-AttnRes-tail-r768 \
  --use_attnres \
  --use_fused_attnres \
  --attnres_type full \
  --no-attnres_block_average \
  --use_lrid \
  --lrid_rank 768 \
  --lrid_num_heads 1 \
  --lrid_key_from_output_tail \
  --no-lrid_logit_scale \
  "$@"
