# train.py
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_DEFAULT_TORCH_COMPILE_CACHE_DIR = (
    os.environ.get("TORCH_COMPILE_CACHE_DIR")
    or os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    or ""
)
_EXPLICIT_TRITON_CACHE_DIR = os.environ.get("TRITON_CACHE_DIR")
if os.environ.get("TORCH_COMPILE_CACHE_DIR") and os.environ.get("TORCHINDUCTOR_CACHE_DIR") is None:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.environ["TORCH_COMPILE_CACHE_DIR"]
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass, asdict
import math
import time
import os, sys
import copy
import argparse
import subprocess
import gc
import numpy as np
from contextlib import nullcontext
from utils import (
    get_config,
    get_device,
    get_model,
    get_dataloader,
    compute_validation_loss,
    compute_lm_loss,
    loss_to_token_sum,
    unwrap_model,
)
from criterion import get_criterion
from wandb_logger import get_logger
from optimizer import get_optimizers
from schedulers import get_schedulers
from dataloader import create_dataloaders, DataLoaderConfig, warmup_boundaries
from typing import Optional, List, Dict, Any
from model import OBPM
import torch.nn.functional as F
from tokenizer_utils import (
    GPT4_EOT_TOKEN as _GPT4_EOT_TOKEN,
    GPT4_TOKENIZER_MODEL as _GPT4_TOKENIZER_MODEL,
    GPT4_VOCAB_SIZE as _GPT4_VOCAB_SIZE,
    get_tiktoken_encoding,
)


def setup_distributed():
    has_rank = "RANK" in os.environ
    has_world_size = "WORLD_SIZE" in os.environ
    if not has_rank and not has_world_size:
        return False, 0, 1, 0
    if not has_rank or not has_world_size:
        raise RuntimeError("Both RANK and WORLD_SIZE must be set for distributed training.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError("RANK must satisfy 0 <= RANK < WORLD_SIZE")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be >= 0")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(local_rank)
    if world_size == 1:
        return False, rank, world_size, local_rank

    backend = "nccl" if cuda_available else "gloo"
    if cuda_available:
        device = torch.device("cuda", local_rank)
        try:
            dist.init_process_group(backend=backend, device_id=device)
        except TypeError:
            dist.init_process_group(backend=backend)
        dist.barrier(device_ids=[local_rank])
    else:
        dist.init_process_group(backend=backend)
        dist.barrier()
    return True, rank, world_size, local_rank


distributed, rank, world_size, local_rank = setup_distributed()
master_process = rank == 0


def print0(*args, **kwargs):
    if master_process:
        print(*args, **kwargs)


def reduce_float(value, op=None):
    if not distributed:
        return float(value)
    if op is None:
        op = dist.ReduceOp.AVG
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import torch._dynamo as dynamo
if hasattr(dynamo.config, "recompile_limit"):
    dynamo.config.recompile_limit = 64

device = get_device(local_rank if distributed else None, distributed=distributed)
# -------------------------------- Config ------------------------------------
# I/O
out_dir = 'out'
eval_interval = 100
log_interval = 1
eval_steps = 10
eval_only = False
save_checkpoint = True
ckpt_interval = 2500
save_ckpt_at_end = True
interactive_after_train = False
init_from = 'scratch'
ckpt_file_name = ''
# wandb logging
wandb_log = True
wandb_project = "LR-AttnRes"
wandb_run_name = "LRID"
# data
dataset_dir = "ultrafineweb20B_gpt4"
batch_size = 16
block_size = 2048
grad_accum_steps = 8
total_batch_size = batch_size * block_size * grad_accum_steps
tokenizer_model = _GPT4_TOKENIZER_MODEL
data_dtype = "uint32"
# DDP
ddp_preserve_global_batch = True
ddp_find_unused_parameters = False
# torch.compile
torch_compile = True
torch_compile_max_autotune = True
torch_compile_cudagraphs = False
torch_compile_cache_dir = _DEFAULT_TORCH_COMPILE_CACHE_DIR
# Full run automation
full_run = True
full_run_hf_repo_id = ""
full_run_hf_private = False
full_run_eval = True
full_run_eval_mode = "full" # full, validation-only, tasks-only
full_run_eval_torch_max_autotune = False
# Document masking (Dataloader)
use_doc_masking = True
doc_separator_token = _GPT4_EOT_TOKEN
num_workers = 8
pin_memory = True if device.type == "cuda" else False
persistent_workers = False
# model
n_layer = 24
n_head = 16
n_embd = 1024
vocab_size = _GPT4_VOCAB_SIZE
mlp_hidden_dim = 2816
mlp_ratio = None
weight_tying = False
flash_attention = True
init_std = 0.02
init_cutoff_factor = None
# Attention Residuals
use_attnres = False
use_fused_attnres = False
attnres_type = "block" # "full" or "block"
attnres_num_blocks = 8
attnres_block_average = False
attnres_block_average_mode = "count" # count or sqrt
attnres_key_norm = True
attn_res_query_norm = False
attn_res_query_init = "zero" # zero, normal, trunc_normal
attnres_training_cache_phase1 = True
attnres_training_torch_phase2 = True
attnres_fuse_read_norm = True
use_lrid = False
lrid_rank = 32
lrid_projection_rank = None
lrid_num_heads = 1
lrid_input_dependent_query = False
lrid_static_embedding_key = False
lrid_add_static_embedding_key = False
lrid_add_static_source_key = False
lrid_key_from_value = False
lrid_key_from_value_shared = False
lrid_key_from_output_tail = False
lrid_key_value_norm = True
lrid_query_from_value = False
lrid_query_from_value_shared = False
lrid_use_logit_scale = False
lrid_logit_scale = None # None defaults to 1 / sqrt(lrid_rank / lrid_num_heads) when enabled
# rope
rope_theta = 500000.0
# normalization
qk_norm = True
norm_pos = "before" # before, after, both
clip_qkv = None
# optimizer (Muon + AdamW settings)
muon_lr = 0.001
adamw_lr= 0.0003
max_steps = 38146
max_tokens = int(10e9)
muon_weight_decay = 0.1
adamw_weight_decay = 0.0
cautious = True
beta1 = 0.9
beta2 = 0.95
muon_momentum = 0.95
grad_clip = 1.0
# Momentum warmup/cooldown settings
muon_momentum_warmup_steps = 300
muon_momentum_cooldown_steps = 100
muon_momentum_min = 0.85
muon_momentum_max = 0.95
# Cross Entropy Loss
ignore_index = -100
reduction = "mean"
z_loss = True
z_loss_weight = 1e-5
ce_inplace_backward = True
lm_head_chunk_size = 1024
# Scheduler
warmup_steps = 2000
warmdown_steps = int(0.2 * max_steps)
sched_mode = "linear"

# -----------------------------------------------------------------------------

def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def _looks_like_hf_token(value: str) -> bool:
    value = (value or "").strip()
    return value.startswith("hf_")


def configure_torch_compile_cache(cache_dir: str) -> str:
    cache_dir = (cache_dir or "").strip()
    if not cache_dir:
        return ""
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    if _EXPLICIT_TRITON_CACHE_DIR is None:
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_dir, "triton")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    return cache_dir


def prepare_full_run_hf_repo(repo_id: str, private: bool) -> str:
    try:
        from huggingface_hub import HfApi, login
    except Exception as exc:
        raise RuntimeError(
            "full_run requires huggingface-hub. Install it with `pip install huggingface-hub`."
        ) from exc

    api = HfApi()
    try:
        user_info = api.whoami()
    except Exception:
        print("full_run: please sign in to Hugging Face. Your token needs write access.")
        login()
        user_info = api.whoami()

    username = user_info.get("name") or user_info.get("fullname")
    repo_id = (repo_id or os.environ.get("HF_REPO_ID", "")).strip()
    if not repo_id:
        default_repo = f"{username}/LR-AttnRes-{int(time.time())}" if username else ""
        if not sys.stdin.isatty():
            raise RuntimeError(
                "full_run needs a Hugging Face repo id in non-interactive runs. "
                "Pass --full_run_hf_repo_id username/repo or set HF_REPO_ID."
            )
        prompt = "Hugging Face model repo id (namespace/name, not a token)"
        if default_repo:
            prompt += f" [{default_repo}]"
        repo_id = input(f"{prompt}: ").strip() or default_repo

    if _looks_like_hf_token(repo_id):
        raise RuntimeError(
            "That looks like a Hugging Face token, not a model repo id. "
            "The token belongs in the earlier Hugging Face login prompt. "
            "Use a repo id like `username/model-name`, and revoke any token "
            "that was pasted into terminal logs or a repo name."
        )

    if "/" not in repo_id:
        if not username:
            raise RuntimeError("Hugging Face repo id must be `namespace/name`.")
        repo_id = f"{username}/{repo_id}"

    print(f"full_run: creating or reusing Hugging Face model repo {repo_id!r}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            "full_run could not create or access the Hugging Face model repo "
            f"{repo_id!r}. Make sure the token has write permissions and that "
            "the repo namespace is an account or organization you can write to. "
            "You can also create the model repo manually on Hugging Face first, "
            "then rerun with --full_run_hf_repo_id namespace/repo."
        ) from exc
    return repo_id


def upload_full_run_checkpoint(ckpt_path: str, repo_id: str, current_step: int, tokens_seen: int):
    from huggingface_hub import HfApi

    api = HfApi()
    print0(f"full_run: uploading final checkpoint to https://huggingface.co/{repo_id}")
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo="final_model.pt",
        repo_id=repo_id,
        repo_type="model",
    )

    card_path = os.path.join(out_dir, "full_run_model_card.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("# LR-AttnRes full_run checkpoint\n\n")
        f.write(f"- Final checkpoint: `{os.path.basename(ckpt_path)}`\n")
        f.write(f"- Training step: `{current_step}`\n")
        f.write(f"- Tokens processed: `{tokens_seen}`\n")
        f.write("- Uploaded artifact: `final_model.pt`\n")
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print0("full_run: Hugging Face upload complete.")


def run_full_run_eval(ckpt_path: str, current_step: int):
    if not full_run_eval:
        return

    results_file = os.path.join(out_dir, f"full_run_eval_step:{current_step}.txt")
    eval_args = [
        "run_eval.py",
        "--ckpts",
        ckpt_path,
        "--results-file",
        results_file,
    ]
    if torch_compile_cache_dir:
        eval_args.extend(["--torch_compile_cache_dir", torch_compile_cache_dir])
    if full_run_eval_mode == "validation-only":
        eval_args.append("--validation-only")
    elif full_run_eval_mode == "tasks-only":
        eval_args.append("--tasks-only")
    if full_run_eval_torch_max_autotune:
        eval_args.append("--torch-max-autotune")

    if world_size > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world_size}",
            *eval_args,
        ]
    else:
        cmd = [sys.executable, *eval_args]

    env = os.environ.copy()
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)

    print0("full_run: running evaluation:")
    print0(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print0(f"full_run: evaluation results saved to {results_file}")


def release_training_state_for_full_run_eval():
    global model, criterion, optimizers, schedulers, train_loader, val_loader

    del model
    del criterion
    del optimizers
    del schedulers
    del train_loader
    del val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train OBPM.")
    parser.add_argument("--eval_only", type=_str_to_bool, nargs="?", const=True, default=eval_only)
    parser.add_argument("--no-eval_only", dest="eval_only", action="store_false")
    parser.add_argument("--init_from", "--init-from", choices=("scratch", "resume"), default=init_from)
    parser.add_argument("--ckpt_file_name", "--ckpt-file-name", type=str, default=ckpt_file_name)
    parser.add_argument("--wandb_log", type=_str_to_bool, nargs="?", const=True, default=wandb_log)
    parser.add_argument("--no-wandb_log", dest="wandb_log", action="store_false")
    parser.add_argument("--ddp_preserve_global_batch", type=_str_to_bool, nargs="?", const=True, default=ddp_preserve_global_batch)
    parser.add_argument("--no-ddp_preserve_global_batch", dest="ddp_preserve_global_batch", action="store_false")
    parser.add_argument("--ddp_find_unused_parameters", type=_str_to_bool, nargs="?", const=True, default=ddp_find_unused_parameters)
    parser.add_argument("--no-ddp_find_unused_parameters", dest="ddp_find_unused_parameters", action="store_false")
    parser.add_argument("--torch_compile", type=_str_to_bool, nargs="?", const=True, default=torch_compile)
    parser.add_argument("--no-torch_compile", dest="torch_compile", action="store_false")
    parser.add_argument(
        "--torch_compile_max_autotune",
        "--torch-max-autotune",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=torch_compile_max_autotune,
    )
    parser.add_argument(
        "--no-torch_compile_max_autotune",
        "--no-torch-max-autotune",
        dest="torch_compile_max_autotune",
        action="store_false",
    )
    parser.add_argument("--torch_compile_cudagraphs", type=_str_to_bool, nargs="?", const=True, default=torch_compile_cudagraphs)
    parser.add_argument("--no-torch_compile_cudagraphs", dest="torch_compile_cudagraphs", action="store_false")
    parser.add_argument(
        "--torch_compile_cache_dir",
        type=str,
        default=torch_compile_cache_dir,
        help="Directory for TorchInductor/Triton compile caches. Use a large persistent path for max-autotune.",
    )
    parser.add_argument("--full_run", type=_str_to_bool, nargs="?", const=True, default=full_run)
    parser.add_argument("--no-full_run", dest="full_run", action="store_false")
    parser.add_argument("--full_run_hf_repo_id", type=str, default=full_run_hf_repo_id)
    parser.add_argument("--full_run_hf_private", type=_str_to_bool, nargs="?", const=True, default=full_run_hf_private)
    parser.add_argument("--no-full_run_hf_private", dest="full_run_hf_private", action="store_false")
    parser.add_argument("--full_run_eval", type=_str_to_bool, nargs="?", const=True, default=full_run_eval)
    parser.add_argument("--no-full_run_eval", dest="full_run_eval", action="store_false")
    parser.add_argument("--full_run_eval_mode", choices=("full", "validation-only", "tasks-only"), default=full_run_eval_mode)
    parser.add_argument(
        "--full_run_eval_torch_max_autotune",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=full_run_eval_torch_max_autotune,
    )
    parser.add_argument("--no-full_run_eval_torch_max_autotune", dest="full_run_eval_torch_max_autotune", action="store_false")
    parser.add_argument("--use_doc_masking", type=_str_to_bool, nargs="?", const=True, default=use_doc_masking)
    parser.add_argument("--no-use_doc_masking", dest="use_doc_masking", action="store_false")
    parser.add_argument("--use_attnres", type=_str_to_bool, nargs="?", const=True, default=use_attnres)
    parser.add_argument("--no-use_attnres", dest="use_attnres", action="store_false")
    parser.add_argument("--use_fused_attnres", type=_str_to_bool, nargs="?", const=True, default=use_fused_attnres)
    parser.add_argument("--no-use_fused_attnres", dest="use_fused_attnres", action="store_false")
    parser.add_argument("--attnres_type", choices=("full", "block"), default=attnres_type)
    parser.add_argument("--attnres_num_blocks", type=int, default=attnres_num_blocks)
    parser.add_argument("--attnres_block_average", type=_str_to_bool, nargs="?", const=True, default=attnres_block_average)
    parser.add_argument("--no-attnres_block_average", dest="attnres_block_average", action="store_false")
    parser.add_argument("--attnres_block_average_mode", choices=("count", "sqrt"), default=attnres_block_average_mode)
    parser.add_argument("--attnres_key_norm", type=_str_to_bool, nargs="?", const=True, default=attnres_key_norm)
    parser.add_argument("--no-attnres_key_norm", dest="attnres_key_norm", action="store_false")
    parser.add_argument("--attn_res_query_norm", type=_str_to_bool, nargs="?", const=True, default=attn_res_query_norm)
    parser.add_argument("--no-attn_res_query_norm", dest="attn_res_query_norm", action="store_false")
    parser.add_argument("--attn_res_query_init", choices=("zero", "normal", "trunc_normal"), default=attn_res_query_init)
    parser.add_argument("--attnres_training_cache_phase1", type=_str_to_bool, nargs="?", const=True, default=attnres_training_cache_phase1)
    parser.add_argument("--no-attnres_training_cache_phase1", dest="attnres_training_cache_phase1", action="store_false")
    parser.add_argument("--attnres_training_torch_phase2", type=_str_to_bool, nargs="?", const=True, default=attnres_training_torch_phase2)
    parser.add_argument("--no-attnres_training_torch_phase2", dest="attnres_training_torch_phase2", action="store_false")
    parser.add_argument("--attnres_fuse_read_norm", type=_str_to_bool, nargs="?", const=True, default=attnres_fuse_read_norm)
    parser.add_argument("--no-attnres_fuse_read_norm", dest="attnres_fuse_read_norm", action="store_false")
    parser.add_argument("--use_lrid", type=_str_to_bool, nargs="?", const=True, default=use_lrid)
    parser.add_argument("--no-use_lrid", dest="use_lrid", action="store_false")
    parser.add_argument("--lrid_rank", type=int, default=lrid_rank)
    parser.add_argument("--lrid_projection_rank", type=int, default=lrid_projection_rank)
    parser.add_argument("--lrid_num_heads", type=int, default=lrid_num_heads)
    parser.add_argument("--lrid_input_dependent_query", type=_str_to_bool, nargs="?", const=True, default=lrid_input_dependent_query)
    parser.add_argument("--no-lrid_input_dependent_query", dest="lrid_input_dependent_query", action="store_false")
    parser.add_argument("--lrid_static_embedding_key", type=_str_to_bool, nargs="?", const=True, default=lrid_static_embedding_key)
    parser.add_argument("--no-lrid_static_embedding_key", dest="lrid_static_embedding_key", action="store_false")
    parser.add_argument("--lrid_add_static_embedding_key", type=_str_to_bool, nargs="?", const=True, default=lrid_add_static_embedding_key)
    parser.add_argument("--no-lrid_add_static_embedding_key", dest="lrid_add_static_embedding_key", action="store_false")
    parser.add_argument("--lrid_add_static_source_key", type=_str_to_bool, nargs="?", const=True, default=lrid_add_static_source_key)
    parser.add_argument("--no-lrid_add_static_source_key", dest="lrid_add_static_source_key", action="store_false")
    parser.add_argument("--lrid_key_from_value", type=_str_to_bool, nargs="?", const=True, default=lrid_key_from_value)
    parser.add_argument("--no-lrid_key_from_value", dest="lrid_key_from_value", action="store_false")
    parser.add_argument("--lrid_key_from_value_shared", type=_str_to_bool, nargs="?", const=True, default=lrid_key_from_value_shared)
    parser.add_argument("--no-lrid_key_from_value_shared", dest="lrid_key_from_value_shared", action="store_false")
    parser.add_argument("--lrid_key_from_output_tail", type=_str_to_bool, nargs="?", const=True, default=lrid_key_from_output_tail)
    parser.add_argument("--no-lrid_key_from_output_tail", dest="lrid_key_from_output_tail", action="store_false")
    parser.add_argument("--lrid_key_value_norm", type=_str_to_bool, nargs="?", const=True, default=lrid_key_value_norm)
    parser.add_argument("--no-lrid_key_value_norm", dest="lrid_key_value_norm", action="store_false")
    parser.add_argument("--lrid_query_from_value", type=_str_to_bool, nargs="?", const=True, default=lrid_query_from_value)
    parser.add_argument("--no-lrid_query_from_value", dest="lrid_query_from_value", action="store_false")
    parser.add_argument("--lrid_query_from_value_shared", type=_str_to_bool, nargs="?", const=True, default=lrid_query_from_value_shared)
    parser.add_argument("--no-lrid_query_from_value_shared", dest="lrid_query_from_value_shared", action="store_false")
    parser.add_argument("--lrid_use_logit_scale", type=_str_to_bool, nargs="?", const=True, default=lrid_use_logit_scale)
    parser.add_argument("--no-lrid_use_logit_scale", "--no-lrid_logit_scale", dest="lrid_use_logit_scale", action="store_false")
    parser.add_argument("--lrid_logit_scale", type=float, default=lrid_logit_scale)
    parser.add_argument("--interactive_after_train", type=_str_to_bool, nargs="?", const=True, default=interactive_after_train)
    parser.add_argument("--no-interactive_after_train", dest="interactive_after_train", action="store_false")
    parser.add_argument("--ce_inplace_backward", type=_str_to_bool, nargs="?", const=True, default=ce_inplace_backward)
    parser.add_argument("--no-ce_inplace_backward", dest="ce_inplace_backward", action="store_false")
    parser.add_argument(
        "--lm_head_chunk_size",
        type=int,
        default=lm_head_chunk_size,
        help="Number of flattened tokens per LM-head loss chunk. Set 0 to materialize full logits.",
    )
    return parser.parse_args()


args = parse_args()
eval_only = args.eval_only
init_from = args.init_from
ckpt_file_name = args.ckpt_file_name
wandb_log = args.wandb_log
ddp_preserve_global_batch = args.ddp_preserve_global_batch
ddp_find_unused_parameters = args.ddp_find_unused_parameters
torch_compile = args.torch_compile
torch_compile_max_autotune = args.torch_compile_max_autotune
if torch_compile_max_autotune:
    torch_compile = True
torch_compile_cudagraphs = args.torch_compile_cudagraphs
torch_compile_cache_dir = args.torch_compile_cache_dir
full_run = args.full_run
full_run_hf_repo_id = args.full_run_hf_repo_id
full_run_hf_private = args.full_run_hf_private
full_run_eval = args.full_run_eval
full_run_eval_mode = args.full_run_eval_mode
full_run_eval_torch_max_autotune = args.full_run_eval_torch_max_autotune
use_doc_masking = args.use_doc_masking
use_attnres = args.use_attnres
use_fused_attnres = args.use_fused_attnres
attnres_type = args.attnres_type
attnres_num_blocks = args.attnres_num_blocks
attnres_block_average = args.attnres_block_average
attnres_block_average_mode = args.attnres_block_average_mode
attnres_key_norm = args.attnres_key_norm
attn_res_query_norm = args.attn_res_query_norm
attn_res_query_init = args.attn_res_query_init
attnres_training_cache_phase1 = args.attnres_training_cache_phase1
attnres_training_torch_phase2 = args.attnres_training_torch_phase2
attnres_fuse_read_norm = args.attnres_fuse_read_norm
use_lrid = args.use_lrid
lrid_rank = args.lrid_rank
lrid_projection_rank = args.lrid_projection_rank
lrid_num_heads = args.lrid_num_heads
lrid_input_dependent_query = args.lrid_input_dependent_query
lrid_static_embedding_key = args.lrid_static_embedding_key
lrid_add_static_embedding_key = args.lrid_add_static_embedding_key
lrid_add_static_source_key = args.lrid_add_static_source_key
lrid_key_from_value = args.lrid_key_from_value
lrid_key_from_value_shared = args.lrid_key_from_value_shared
if lrid_key_from_value_shared:
    lrid_key_from_value = True
lrid_key_from_output_tail = args.lrid_key_from_output_tail
lrid_key_value_norm = args.lrid_key_value_norm
lrid_query_from_value = args.lrid_query_from_value
lrid_query_from_value_shared = args.lrid_query_from_value_shared
if lrid_query_from_value_shared:
    lrid_query_from_value = True
lrid_use_logit_scale = args.lrid_use_logit_scale
if use_lrid:
    use_attnres = True
lrid_logit_scale = args.lrid_logit_scale
interactive_after_train = args.interactive_after_train
ce_inplace_backward = args.ce_inplace_backward
lm_head_chunk_size = args.lm_head_chunk_size
if lm_head_chunk_size < 0:
    raise ValueError("lm_head_chunk_size must be >= 0")
if distributed and ddp_find_unused_parameters and lm_head_chunk_size > 0:
    raise ValueError(
        "lm_head_chunk_size > 0 computes the LM-head loss outside the DDP forward, "
        "which is incompatible with ddp_find_unused_parameters=True. "
        "Use --no-ddp_find_unused_parameters or set --lm_head_chunk_size 0."
    )

if full_run and eval_only:
    raise ValueError("full_run is for training runs; do not combine it with --eval_only.")
if full_run and interactive_after_train:
    print0("full_run disables interactive_after_train so upload/eval can run unattended.")
    interactive_after_train = False

if full_run:
    try:
        if master_process:
            full_run_hf_repo_id = prepare_full_run_hf_repo(full_run_hf_repo_id, full_run_hf_private)
        if distributed:
            dist.barrier()
    except Exception:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        raise

configured_grad_accum_steps = grad_accum_steps
if distributed and ddp_preserve_global_batch:
    if configured_grad_accum_steps % world_size != 0:
        raise ValueError(
            "grad_accum_steps must be divisible by WORLD_SIZE when "
            "ddp_preserve_global_batch=True. "
            f"Got grad_accum_steps={configured_grad_accum_steps}, WORLD_SIZE={world_size}."
        )
    grad_accum_steps = configured_grad_accum_steps // world_size

global_grad_accum_steps = grad_accum_steps * world_size
total_batch_size = batch_size * block_size * global_grad_accum_steps

wandb_log = wandb_log and master_process
save_checkpoint = save_checkpoint and master_process
interactive_after_train = interactive_after_train and master_process

config = get_config(sys.modules[__name__].__dict__)
start_step, checkpoint, model, model_config = get_model(config, device)
if device.type == "cuda":
    model.to_mixed_precision(dtype=torch.bfloat16)
if distributed:
    for param in model.parameters():
        dist.broadcast(param.detach(), src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.detach(), src=0)
# -----------------------------------------------------------------------------

torch_compile_mode = None
if torch_compile_max_autotune:
    torch_compile_mode = "max-autotune" if torch_compile_cudagraphs else "max-autotune-no-cudagraphs"
if torch_compile:
    torch_compile_cache_dir = configure_torch_compile_cache(torch_compile_cache_dir)
    if not torch_compile_cudagraphs:
        try:
            import torch._inductor.config as inductor_config

            if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                inductor_config.triton.cudagraphs = False
            if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraph_trees"):
                inductor_config.triton.cudagraph_trees = False
        except Exception as exc:
            print0(f"Could not disable torch.compile CUDA graphs through inductor config: {exc}")
    compile_kwargs = {}
    if torch_compile_mode is not None:
        compile_kwargs["mode"] = torch_compile_mode
    model = torch.compile(model, **compile_kwargs)

if distributed:
    ddp_kwargs = dict(find_unused_parameters=ddp_find_unused_parameters)
    if device.type == "cuda":
        ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
    model = DDP(model, **ddp_kwargs)

num_params = unwrap_model(model).get_num_params()
if master_process:
    os.makedirs(out_dir, exist_ok=True)
logger = get_logger(config, num_params=num_params)
print0(f"Device: {device}")
print0(f"Distributed: {distributed} | Rank: {rank}/{world_size} | Local rank: {local_rank}")
print0(f"Total Parameters: {num_params:,}")
print0(f"Total Batch Size: {total_batch_size}")
print0(f"Configured gradient accumulation steps: {configured_grad_accum_steps}")
print0(f"Local gradient accumulation steps: {grad_accum_steps}")
print0(f"Torch compile: {torch_compile} | mode: {torch_compile_mode or 'default'}")
print0(f"Torch compile CUDA graphs: {torch_compile_cudagraphs}")
print0(f"Torch compile cache dir: {torch_compile_cache_dir or 'default'}")
print0(f"Full run: {full_run} | HF repo: {full_run_hf_repo_id or 'N/A'}")

def get_muon_momentum(step):
    momentum_cd_start = max_steps - muon_momentum_cooldown_steps
    if step < muon_momentum_warmup_steps:
        frac = step / muon_momentum_warmup_steps
        momentum = muon_momentum_min + frac * (muon_momentum_max - muon_momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_momentum_cooldown_steps
        momentum = muon_momentum_max - frac * (muon_momentum_max - muon_momentum_min)
    else:
        momentum = muon_momentum_max
    return momentum

criterion = get_criterion(config)
optimizers = get_optimizers(config, model)
muon_optimizer, adamw_optimizer = optimizers
schedulers = get_schedulers(config, muon_optimizer, adamw_optimizer)
muon_scheduler, adamw_scheduler = schedulers
train_loader, val_loader = get_dataloader(config)

if use_doc_masking:
    print0("Warming up document boundary cache...")
    warmup_boundaries(train_loader.dataset, verbose=master_process)
    warmup_boundaries(val_loader.dataset, verbose=master_process)
    print0("Boundary warmup complete.")

tokens_processed = 0
tokens_per_step = batch_size * block_size * grad_accum_steps * world_size
if checkpoint is not None:
    muon_optimizer.load_state_dict(checkpoint["muon_optimizer"])
    adamw_optimizer.load_state_dict(checkpoint["adamw_optimizer"])
    muon_scheduler.load_state_dict(checkpoint["muon_scheduler"])
    adamw_scheduler.load_state_dict(checkpoint["adamw_scheduler"])
    tokens_processed = int(checkpoint["tokens_processed"])

print0(f"Tokens per step: {tokens_per_step:,}")
print0(f"Starting from step {start_step}, tokens seen: {tokens_processed:,}")


def build_checkpoint(current_step: int):
    raw_model = unwrap_model(model)
    return {
        "step": current_step,
        "tokens_processed": tokens_processed,
        "model": raw_model.state_dict(),
        "muon_optimizer": muon_optimizer.state_dict(),
        "adamw_optimizer": adamw_optimizer.state_dict(),
        "muon_scheduler": muon_scheduler.state_dict(),
        "adamw_scheduler": adamw_scheduler.state_dict(),
        "config": config,
        "model_args": asdict(model_config),
    }


def save_training_checkpoint(current_step: int, ckpt_path: str):
    checkpoint_payload = build_checkpoint(current_step)
    torch.save(checkpoint_payload, ckpt_path)
    print0(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path


def infinite_dataloader(dataloader, start_epoch=0):
    epoch = start_epoch
    while True:
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1


@torch.no_grad()
def estimate_loss(current_step):
    out = {}
    model.eval()
    ignore_index = getattr(getattr(criterion, "config", None), "ignore_index", -100)
    reduction = getattr(getattr(criterion, "config", None), "reduction", "mean")

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        total_tokens = 0
        total_batches = 0
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(current_step)
        eval_iter = iter(loader)

        for k in range(eval_steps):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break

            if use_doc_masking:
                x, y, cu_seqlens, max_seqlen = batch
                cu_seqlens = cu_seqlens.to(device)
            else:
                x, y = batch[:2]
                cu_seqlens, max_seqlen = None, None

            if x.max() >= vocab_size or y.max() >= vocab_size:
                print(f"ERROR: Out-of-bounds token detected in training batch!")
                print(f"  x min/max: {x.min()}/{x.max()}")
                print(f"  y min/max: {y.min()}/{y.max()}")
                print(f"  Step: {current_step}")
                raise ValueError("Out-of-bounds token detected in evaluation batch.")

            x, y = x.to(device), y.to(device)
            valid_tokens = int((y != ignore_index).sum().item())
            if valid_tokens == 0:
                continue

            loss = compute_lm_loss(
                model,
                criterion,
                x,
                y,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cast_logits_to_float=True,
            )

            total_loss += loss_to_token_sum(loss, valid_tokens, reduction)
            total_tokens += valid_tokens
            total_batches += 1

        if distributed:
            stats = torch.tensor(
                [total_loss, float(total_tokens), float(total_batches)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total_loss = float(stats[0].item())
            total_tokens = int(stats[1].item())
            total_batches = int(stats[2].item())
            if total_tokens == 0:
                raise RuntimeError(f"No batches available while estimating {split} loss.")
            out[split] = total_loss / total_tokens
        else:
            if total_tokens == 0:
                raise RuntimeError(f"No batches available while estimating {split} loss.")
            out[split] = total_loss / total_tokens
    model.train()
    return out

step = start_step

if eval_only:
    print0("=" * 80)
    print0("Running full validation set evaluation...")
    print0("=" * 80)
    val_metrics = compute_validation_loss(
        model,
        criterion,
        val_loader,
        device,
        vocab_size,
        use_doc_masking=use_doc_masking,
        distributed=distributed,
    )
    print0(
        f"Validation loss: {val_metrics['loss']:.4f} "
        f"({val_metrics['tokens']:,} tokens across {val_metrics['batches']:,} batches)"
    )
    if wandb_log:
        logger.log_validation(
            float(val_metrics["loss"]),
            tokens_processed,
            lr=muon_scheduler.get_last_lr()[0],
        )
        logger.finish()
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
    raise SystemExit

print0("=" * 80)
print0("Starting training...")
print0("=" * 80)

loader_batches_per_epoch = max(1, len(train_loader))
train_iter = infinite_dataloader(
    train_loader,
    start_epoch=(start_step * grad_accum_steps) // loader_batches_per_epoch,
)

while tokens_processed < max_tokens and step < max_steps:
    muon_optimizer.param_groups[0]['momentum'] = get_muon_momentum(step)

    if step != 0 and (step % eval_interval == 0 or step == max_steps - 1):
        losses = estimate_loss(step)
        print0(f"Eval: Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            logger.log_eval(
                step,
                float(losses["train"]),
                float(losses["val"]),
                muon_scheduler.get_last_lr()[0],
                tokens_processed,
            )
    if save_checkpoint and step > 0:
        should_save = (step % ckpt_interval == 0)
        if should_save:
            ckpt_path = os.path.join(out_dir, f"ckpt_step:{step}.pt")
            save_training_checkpoint(step, ckpt_path)
            if wandb_log:
                logger.log_checkpoint(step, ckpt_path, config=config)

    model.train()
    t0 = time.time()

    for opt in optimizers: opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        batch = next(train_iter)

        if use_doc_masking:
            x, y, cu_seqlens, max_seqlen = batch
            cu_seqlens = cu_seqlens.to(device)
        else:
            x, y = batch[:2]
            cu_seqlens, max_seqlen = None, None

        if x.max() >= vocab_size or y.max() >= vocab_size:
            print(f"ERROR: Out-of-bounds token detected in training batch!")
            print(f"  x min/max: {x.min()}/{x.max()}")
            print(f"  y min/max: {y.min()}/{y.max()}")
            print(f"  Step: {step}")
            raise ValueError("Out-of-bounds token detected in training batch.")

        x, y = x.to(device), y.to(device)

        sync_context = (
            model.no_sync()
            if distributed and micro_step < grad_accum_steps - 1
            else nullcontext()
        )
        with sync_context:
            loss = compute_lm_loss(
                model,
                criterion,
                x,
                y,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cast_logits_to_float=False,
            )
            loss = loss / grad_accum_steps

            loss_accum += loss.detach().item()

            loss.backward()

    if grad_clip > 0.0: norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else: norm = None

    for opt in optimizers: opt.step()
    for sched in schedulers: sched.step()

    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.time()
    elapsed = reduce_float(t1 - t0, op=dist.ReduceOp.MAX if distributed else None)
    loss_accum = reduce_float(loss_accum)

    tokens_processed += tokens_per_step

    tokens_per_s = tokens_per_step / elapsed
    ms_per_step = elapsed * 1000.0
    peak_gpu_memory_gb = None
    peak_gpu_memory_reserved_gb = None
    if device.type == "cuda":
        peak_gpu_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_gpu_memory_gb = reduce_float(
            peak_gpu_memory_gb,
            op=dist.ReduceOp.MAX if distributed else None,
        )
        peak_gpu_memory_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        peak_gpu_memory_reserved_gb = reduce_float(
            peak_gpu_memory_reserved_gb,
            op=dist.ReduceOp.MAX if distributed else None,
        )

    if wandb_log:
        logger.log_train(
            step, loss_accum, norm,
            muon_scheduler.get_last_lr()[0],
            ms_per_step, tokens_per_s, tokens_processed,
            peak_gpu_memory_gb=peak_gpu_memory_gb,
            peak_gpu_memory_reserved_gb=peak_gpu_memory_reserved_gb,
        )

    if step % log_interval == 0:
        norm_str = f"{norm:.2f}" if norm is not None else "N/A"
        peak_memory_str = (
            f"Peak VRAM: {peak_gpu_memory_gb:.2f}GB, "
            if peak_gpu_memory_gb is not None
            else ""
        )
        print0(
            f"Step {step}, "
            f"Loss: {loss_accum:.4f}, "
            f"Time: {ms_per_step:.2f}ms, "
            f"Tokens/s: {tokens_per_s:.2f}, "
            f"Tokens seen: {tokens_processed:,}, "
            f"Norm: {norm_str}, "
            f"{peak_memory_str}"
            f"Muon LR: {muon_scheduler.get_last_lr()[0]:.6f}, "
            f"AdamW LR: {adamw_scheduler.get_last_lr()[0]:.6f}"
        )

    step += 1

print0("=" * 80)
print0("Training complete!")
print0("=" * 80)

final_ckpt_path = None
if master_process and (full_run or (save_checkpoint and save_ckpt_at_end)):
    final_ckpt_path = os.path.join(out_dir, f"ckpt_step:{step}.pt")
    save_training_checkpoint(step, final_ckpt_path)
    if wandb_log:
        logger.log_checkpoint(step, final_ckpt_path, config=config)

if distributed and dist.is_initialized():
    dist.barrier()

if wandb_log: logger.finish()

if distributed and dist.is_initialized():
    dist.destroy_process_group()

if master_process and full_run:
    if final_ckpt_path is None:
        raise RuntimeError("full_run could not find a final checkpoint to upload/evaluate.")
    if full_run_eval:
        release_training_state_for_full_run_eval()
    upload_full_run_checkpoint(final_ckpt_path, full_run_hf_repo_id, step, tokens_processed)
    run_full_run_eval(final_ckpt_path, step)

if interactive_after_train and sys.stdin.isatty():
    enc = get_tiktoken_encoding(tokenizer_model)
    generation_model = unwrap_model(model)

    with torch.inference_mode():
        print("\nInteractive generation mode. Type your prompt and press Enter.")
        print("Type 'quit' or press Ctrl-C to exit.\n")
        while True:
            try:
                text = input(">>> ")
                if text.strip().lower() in {"quit", "exit", "q"}:
                    break

                tokens = enc.encode(text)
                if not tokens:
                    continue

                x0 = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

                max_new_tokens = max(1, block_size - len(tokens))
                out_tokens = generation_model.generate(x0, max_new_tokens=max_new_tokens, top_k=5)[0].tolist()
                print(enc.decode(out_tokens))
                print("-" * 80)

            except KeyboardInterrupt:
                print("\nExiting generation mode.")
                break
            except Exception as e:
                print(f"Generation error: {e}")
