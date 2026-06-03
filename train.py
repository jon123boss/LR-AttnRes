# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, asdict
import math
import time
import tiktoken
import os, sys
import copy
import argparse
import numpy as np
from utils import get_config, get_device, get_model, get_dataloader
from criterion import get_criterion
from wandb_logger import get_logger
from optimizer import get_optimizers
from schedulers import get_schedulers
from dataloader import create_dataloaders, DataLoaderConfig, warmup_boundaries
from typing import Optional, List, Dict, Any
from model import OBPM
import torch.nn.functional as F

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import torch._dynamo as dynamo
dynamo.config.recompile_limit = 64

device = get_device()
# -------------------------------- Config ------------------------------------
# I/O
out_dir = 'out'
eval_interval = 100
log_interval = 1
eval_steps = 10
eval_only = False
save_checkpoint = True
ckpt_interval = 10000
save_ckpt_at_end = True
interactive_after_train = False
init_from = 'scratch'
ckpt_file_name = ''
# wandb logging
wandb_log = True
wandb_project = "OBPM"
wandb_run_name = "OBPM"
# data
dataset_dir = "finewebedu10B"
batch_size = 32
block_size = 2048
grad_accum_steps = 4
total_batch_size = batch_size * block_size * grad_accum_steps
# Document masking (Dataloader)
use_doc_masking = True
doc_separator_token = 50256
num_workers = 8
pin_memory = True if device.type == "cuda" else False
persistent_workers = False
# model
n_layer = 12
n_head = 12
n_embd = 768
vocab_size = 57601
mlp_hidden_dim = 2048
mlp_ratio = None
weight_tying = False
flash_attention = True
init_std = 0.02
init_cutoff_factor = None
# rope
rope_theta = 500000.0
# normalization
rmsnorm_eps = 1e-6
qk_norm = True
norm_pos = "before" # before, after, both
clip_qkv = None
# optimizer (Muon + AdamW settings)
muon_lr = 0.01
adamw_lr= 0.003
max_steps = 1000
max_tokens = int(10e9)
muon_weight_decay = 0.1
adamw_weight_decay = 0.0
cautious = True
beta1 = 0.9
beta2 = 0.95
muon_momentum = 0.95
grad_clip = 1.0
# Momentum warmup/cooldown settings
muon_momentum_warmup_steps = 100
muon_momentum_cooldown_steps = 100
muon_momentum_min = 0.85
muon_momentum_max = 0.95
# Cross Entropy Loss
ignore_index = -100
reduction = "mean"
z_loss = True
z_loss_weight = 1e-5
# Scheduler
warmup_steps = 100
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train OBPM.")
    parser.add_argument("--eval_only", type=_str_to_bool, nargs="?", const=True, default=eval_only)
    parser.add_argument("--no-eval_only", dest="eval_only", action="store_false")
    parser.add_argument("--wandb_log", type=_str_to_bool, nargs="?", const=True, default=wandb_log)
    parser.add_argument("--no-wandb_log", dest="wandb_log", action="store_false")
    parser.add_argument("--use_doc_masking", type=_str_to_bool, nargs="?", const=True, default=use_doc_masking)
    parser.add_argument("--no-use_doc_masking", dest="use_doc_masking", action="store_false")
    parser.add_argument("--interactive_after_train", type=_str_to_bool, nargs="?", const=True, default=interactive_after_train)
    parser.add_argument("--no-interactive_after_train", dest="interactive_after_train", action="store_false")
    return parser.parse_args()


args = parse_args()
eval_only = args.eval_only
wandb_log = args.wandb_log
use_doc_masking = args.use_doc_masking
interactive_after_train = args.interactive_after_train

config = get_config(sys.modules[__name__].__dict__)
start_step, checkpoint, model, model_config = get_model(config, device)
if device.type == "cuda":
    model.to_mixed_precision(dtype=torch.bfloat16)
# -----------------------------------------------------------------------------

model = torch.compile(model)

logger = get_logger(config, num_params=model.get_num_params())
print(f"Device: {device}")
print(f"Total Parameters: {model.get_num_params():,}")
print(f"Total Batch Size: {total_batch_size}")
print(f"Gradient accumulation steps: {grad_accum_steps}")

os.makedirs(out_dir, exist_ok=True)

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
    print("Warming up document boundary cache...")
    warmup_boundaries(train_loader.dataset)
    warmup_boundaries(val_loader.dataset)
    print("Boundary warmup complete.")

tokens_processed = 0
tokens_per_step = batch_size * block_size * grad_accum_steps
if checkpoint is not None:
    muon_optimizer.load_state_dict(checkpoint["muon_optimizer"])
    adamw_optimizer.load_state_dict(checkpoint["adamw_optimizer"])
    muon_scheduler.load_state_dict(checkpoint["muon_scheduler"])
    adamw_scheduler.load_state_dict(checkpoint["adamw_scheduler"])
    tokens_processed = int(checkpoint["tokens_processed"])

print(f"Tokens per step: {tokens_per_step:,}")
print(f"Starting from step {start_step}, tokens seen: {tokens_processed:,}")

def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def estimate_loss(current_step):
    out = {}
    model.eval()

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
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
                print(f"  Step: {step}")
                raise ValueError("Out-of-bounds token detected in evaluation batch.")

            x, y = x.to(device), y.to(device)

            logits = model(x, cu_doc_len=cu_seqlens, max_doc_len=max_seqlen)
            logits_for_loss = logits.float()
            loss = criterion(logits_for_loss.view(-1, logits_for_loss.size(-1)), y.view(-1))

            losses.append(float(loss.item()))

        if not losses:
            raise RuntimeError(f"No batches available while estimating {split} loss.")
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

step = start_step

print("=" * 80)
print("Starting training...")
print("=" * 80)

train_iter = infinite_dataloader(train_loader)

while tokens_processed < max_tokens and step < max_steps:
    muon_optimizer.param_groups[0]['momentum'] = get_muon_momentum(step)

    if step != 0 and (step % eval_interval == 0 or step == max_steps - 1):
        losses = estimate_loss(step)
        print(f"Eval: Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            logger.log_eval(
                step,
                float(losses["train"]),
                float(losses["val"]),
                muon_scheduler.get_last_lr()[0],
                tokens_processed,
            )
        if eval_only:
            break

    if save_checkpoint and step > 0:
        should_save = (step % ckpt_interval == 0) or (save_ckpt_at_end and step == max_steps - 1)
        if should_save:
            checkpoint = {
                "step": step,
                "tokens_processed": tokens_processed,
                "model": model.state_dict(),
                "muon_optimizer": muon_optimizer.state_dict(),
                "adamw_optimizer": adamw_optimizer.state_dict(),
                "muon_scheduler": muon_scheduler.state_dict(),
                "adamw_scheduler": adamw_scheduler.state_dict(),
                "config": config,
                "model_args": asdict(model_config),
            }
            ckpt_path = os.path.join(out_dir, f"ckpt_step:{step}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            if wandb_log:
                logger.log_checkpoint(step, ckpt_path, config=config)

    model.train()
    t0 = time.time()

    for opt in optimizers: opt.zero_grad(set_to_none=True)

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

        logits = model(x, cu_doc_len=cu_seqlens, max_doc_len=max_seqlen)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / grad_accum_steps

        loss_accum += loss.detach().item()

        loss.backward()

    if grad_clip > 0.0: norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else: norm = None

    for opt in optimizers: opt.step()
    for sched in schedulers: sched.step()

    tokens_processed += tokens_per_step

    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.time()

    tokens_per_s = tokens_per_step / (t1 - t0)
    ms_per_step = (t1 - t0) * 1000.0

    if wandb_log:
        logger.log_train(
            step, loss_accum, norm,
            muon_scheduler.get_last_lr()[0],
            ms_per_step, tokens_per_s, tokens_processed,
        )

    if step % log_interval == 0:
        print(
            f"Step {step}, "
            f"Loss: {loss_accum:.4f}, "
            f"Time: {ms_per_step:.2f}ms, "
            f"Tokens/s: {tokens_per_s:.2f}, "
            f"Tokens seen: {tokens_processed:,}, "
            f"Norm: {norm:.2f}, "
            f"Muon LR: {muon_scheduler.get_last_lr()[0]:.6f}, "
            f"AdamW LR: {adamw_scheduler.get_last_lr()[0]:.6f}"
        )

    step += 1

print("=" * 80)
print("Training complete!")
print("=" * 80)

if wandb_log: logger.finish()

if interactive_after_train and sys.stdin.isatty():
    enc = tiktoken.get_encoding("gpt2")

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
                out_tokens = model.generate(x0, max_new_tokens=max_new_tokens, top_k=5)[0].tolist()
                print(enc.decode(out_tokens))
                print("-" * 80)

            except KeyboardInterrupt:
                print("\nExiting generation mode.")
                break
            except Exception as e:
                print(f"Generation error: {e}")
