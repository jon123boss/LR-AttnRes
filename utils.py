import torch
import torch.nn as nn
from torch.nn import functional as F
from contextlib import nullcontext
import os
import random
import tempfile
import numpy as np
from model import OBPM, ModelConfig
from dataloader import (
    DataLoaderConfig,
    create_dataloaders,
    create_training_evaluation_dataloader,
    create_validation_dataloader,
)


_ATTNRES_KERNEL_ENV_DEFAULTS = {
    "ATTNRES_TRAIN_KERNEL": "auto",
    "ATTNRES_PHASE2_FUSED_QUERY": "1",
    "ATTNRES_PHASE1_QSPLIT": "0",
    "ATTNRES_PHASE1_TILED": "0",
    "ATTNRES_BLOCK_PHASE1_LIST": "0",
    "ATTNRES_PHASE1_LOGITS_LIST": "0",
    "ATTNRES_SAVE_AUX": "0",
}


def get_config(module_globals=None):
    config_keys = [k for k, v in module_globals.items()  if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
    config = {k: module_globals[k] for k in config_keys} 
    return config

def get_device(local_rank=None, distributed=False):
    if torch.cuda.is_available():
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
    elif distributed:
        device = torch.device("cpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def unwrap_model(model):
    while True:
        if hasattr(model, "module"):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        return model


def strip_compiled_prefix(state_dict, prefix="_orig_mod."):
    """Return a state dict that can be loaded into an uncompiled model."""
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["torch_cuda"]])
    return True


def checkpoint_rng_state_for_rank(checkpoint, rank=0):
    """Return this rank's RNG state, with compatibility for old checkpoints."""
    states = checkpoint.get("rng_states_by_rank")
    if states is None:
        return checkpoint.get("rng_state")
    if rank < 0 or rank >= len(states):
        raise ValueError(
            f"Checkpoint has {len(states)} RNG states but rank {rank} was requested."
        )
    return states[rank]


def capture_attnres_kernel_environment():
    return {
        name: os.environ.get(name, default)
        for name, default in _ATTNRES_KERNEL_ENV_DEFAULTS.items()
    }


def capture_training_runtime(criterion, device):
    runtime = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_type": device.type,
        "criterion_backend": (
            "flash_attn" if getattr(criterion, "flash_attention", False) else "torch"
        ),
        "attnres_kernel_environment": capture_attnres_kernel_environment(),
    }
    if device.type == "cuda":
        runtime["cuda_device_name"] = torch.cuda.get_device_name(device)
        runtime["cuda_device_capability"] = list(torch.cuda.get_device_capability(device))
    return runtime


def validate_training_runtime(checkpoint, current_runtime):
    saved_runtime = checkpoint.get("training_runtime")
    if saved_runtime is not None and saved_runtime != current_runtime:
        raise ValueError(
            "Trajectory-exact checkpoint resume requires the same recorded software, "
            "device, criterion, and Attention-Residual kernel environment. "
            f"saved={saved_runtime!r}, current={current_runtime!r}"
        )


def build_dataset_manifest(dataset):
    """Capture the ordered file identity of a packing dataset's shards."""
    shard_paths = getattr(dataset, "_shard_paths", None)
    if shard_paths is None:
        raise TypeError("Dataset does not expose its ordered shard paths.")
    manifest = []
    for path in shard_paths:
        stat = os.stat(path)
        manifest.append(
            {
                "path": os.path.abspath(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
            }
        )
    return manifest


_EXACT_RESUME_DATA_KEYS = (
    "dataset_dir",
    "data_dtype",
    "batch_size",
    "block_size",
    "grad_accum_steps",
    "world_size",
    "seed",
    "use_doc_masking",
    "doc_separator_token",
)


_EXACT_RESUME_TRAINING_KEYS = (
    # Loss/objective.
    "ignore_index",
    "reduction",
    "z_loss",
    "z_loss_weight",
    "ce_inplace_backward",
    "lm_head_chunk_size",
    # Optimizers and clipping.
    "muon_lr",
    "adamw_lr",
    "muon_weight_decay",
    "adamw_weight_decay",
    "cautious",
    "beta1",
    "beta2",
    "grad_clip",
    # Step-dependent momentum and learning-rate schedules.
    "max_steps",
    "max_tokens",
    "muon_momentum",
    "muon_momentum_warmup_steps",
    "muon_momentum_cooldown_steps",
    "muon_momentum_min",
    "muon_momentum_max",
    "warmup_steps",
    "warmdown_steps",
    "sched_mode",
    # Kernel/backend choices can change floating-point results and gradients.
    "flash_attention",
    "torch_compile",
    "torch_compile_max_autotune",
    "torch_compile_cudagraphs",
)


def validate_exact_resume_data_config(checkpoint, current_config, current_data_manifest=None):
    """Reject configuration changes that invalidate trajectory-exact resume."""
    if "train_batches_consumed" not in checkpoint:
        return
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError(
            "Checkpoint has an exact-resume batch cursor but no saved training config."
        )

    mismatches = []
    for key in _EXACT_RESUME_DATA_KEYS + _EXACT_RESUME_TRAINING_KEYS:
        if key in saved_config and key in current_config:
            saved_value = saved_config[key]
            current_value = current_config[key]
            if saved_value != current_value:
                mismatches.append(f"{key}: saved={saved_value!r}, current={current_value!r}")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Trajectory-exact checkpoint resume requires unchanged data and training configuration; "
            + details
        )
    saved_manifest = checkpoint.get("train_data_manifest")
    if saved_manifest is not None:
        if current_data_manifest is None:
            raise ValueError(
                "Checkpoint records a training shard manifest but the current manifest was not provided."
            )
        if saved_manifest != current_data_manifest:
            raise ValueError(
                "Trajectory-exact checkpoint resume requires the same ordered training shards "
                "with unchanged file identity, size, and modification time."
            )


def atomic_torch_save(payload, path):
    """Write a torch checkpoint atomically so interruptions do not corrupt it."""
    target_path = os.path.abspath(path)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target_path)}.",
        suffix=".tmp",
        dir=target_dir,
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, target_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def load_model_checkpoint(ckpt_path, device, verbose=True, load_training_state=True):
    """Load a training checkpoint using the canonical training/eval path."""
    if verbose:
        print(f"Loading checkpoint from: {ckpt_path}")

    load_kwargs = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        checkpoint = torch.load(ckpt_path, mmap=True, **load_kwargs)
    except (TypeError, RuntimeError):
        # mmap is unavailable on older PyTorch releases and legacy checkpoint
        # formats. CPU loading still prevents unused optimizer state from
        # occupying accelerator memory during evaluation.
        checkpoint = torch.load(ckpt_path, **load_kwargs)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint)!r}")
    if "model_args" not in checkpoint or "model" not in checkpoint:
        raise KeyError("Checkpoint must contain both 'model_args' and 'model'.")

    checkpoint_model_args = checkpoint["model_args"]
    if isinstance(checkpoint_model_args, ModelConfig):
        model_config = checkpoint_model_args
    elif isinstance(checkpoint_model_args, dict):
        model_config = ModelConfig(**checkpoint_model_args)
    else:
        raise TypeError(
            "checkpoint['model_args'] must be a dict or ModelConfig, "
            f"got {type(checkpoint_model_args)!r}"
        )

    model = OBPM(model_config)
    model_state_dict = checkpoint["model"]
    if not isinstance(model_state_dict, dict):
        raise TypeError(
            "checkpoint['model'] must be a state dict, "
            f"got {type(model_state_dict)!r}"
        )
    had_compiled_prefix = any(key.startswith("_orig_mod.") for key in model_state_dict)
    if had_compiled_prefix and verbose:
        print("Detected compiled model checkpoint. Removing '_orig_mod.' prefix from state dict keys.")
    model.load_state_dict(strip_compiled_prefix(model_state_dict), strict=True)
    if not load_training_state:
        retained_keys = {
            "step",
            "tokens_processed",
            "train_batches_consumed",
            "model_args",
            "model",
            "config",
        }
        checkpoint = {
            key: value for key, value in checkpoint.items() if key in retained_keys
        }
    model.to(device)
    return checkpoint, model, model_config


def get_model(config, device):
    verbose = bool(config.get("master_process", True))
    start_step = 0
    checkpoint = None
    if config["init_from"] == 'resume':
        import glob
        import re

        def extract_step_number(path):
            match = re.search(r'ckpt_step:(\d+)\.pt', os.path.basename(path))
            return int(match.group(1)) if match else -1

        def latest_checkpoint_path(out_dir):
            step_ckpts = glob.glob(os.path.join(out_dir, 'ckpt_step:*.pt'))
            if not step_ckpts:
                raise FileNotFoundError(f"No ckpt_step:*.pt checkpoints found in {out_dir!r}")
            step_ckpts.sort(key=extract_step_number)
            return step_ckpts[-1]

        ckpt_file_name = (config.get("ckpt_file_name") or "").strip()
        if ckpt_file_name:
            ckpt_path = ckpt_file_name if os.path.isabs(ckpt_file_name) else os.path.join(config["out_dir"], ckpt_file_name)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Requested checkpoint does not exist: {ckpt_path}")
        else:
            ckpt_path = latest_checkpoint_path(config["out_dir"])
        if verbose:
            print(f"Resuming from {ckpt_path}")
        checkpoint, model, model_config = load_model_checkpoint(
            ckpt_path,
            device,
            verbose=verbose,
        )
        start_step = checkpoint["step"]
    elif config["init_from"] == 'scratch':
        if verbose:
            print("Initializing new model from scratch")
        model_config = ModelConfig(
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_embd=config["n_embd"],
            vocab_size=config["vocab_size"],
            block_size=config["block_size"],
            mlp_hidden_dim=config["mlp_hidden_dim"],
            mlp_ratio=config["mlp_ratio"],
            weight_tying=config["weight_tying"],
            rope_theta=config["rope_theta"],
            norm_pos=config["norm_pos"],
            qk_norm=config["qk_norm"],
            clip_qkv=config["clip_qkv"],
            flash_attention=config["flash_attention"],
            init_std=config["init_std"],
            init_cutoff_factor=config["init_cutoff_factor"],
            use_attnres = config["use_attnres"],
            use_fused_attnres = config.get("use_fused_attnres", False),
            attnres_type = config["attnres_type"],
            attnres_num_blocks = config["attnres_num_blocks"],
            attnres_block_average = config.get("attnres_block_average", True),
            attnres_block_average_mode = config.get("attnres_block_average_mode", "count"),
            attnres_block_count_prior = config.get("attnres_block_count_prior", True),
            attnres_block_alpha = config.get("attnres_block_alpha", "legacy"),
            attnres_block_beta = config.get("attnres_block_beta", "legacy"),
            attnres_block_alpha_learned = config.get("attnres_block_alpha_learned", False),
            attnres_block_beta_learned = config.get("attnres_block_beta_learned", False),
            attnres_block_alpha_scope = config.get("attnres_block_alpha_scope", "shared"),
            attnres_block_beta_scope = config.get("attnres_block_beta_scope", "shared"),
            attnres_block_split_sublayers = config.get("attnres_block_split_sublayers", False),
            attnres_block_learned_scale = config.get("attnres_block_learned_scale", False),
            attnres_block_learned_scale_init = config.get("attnres_block_learned_scale_init", "count"),
            attnres_block_value_norm = config.get("attnres_block_value_norm", False),
            attnres_key_norm = config["attnres_key_norm"],
            attn_res_query_norm = config["attn_res_query_norm"],
            attn_res_query_init = config["attn_res_query_init"],
            attnres_training_cache_phase1 = config.get("attnres_training_cache_phase1", True),
            attnres_training_torch_phase2 = config.get("attnres_training_torch_phase2", True),
            attnres_fuse_read_norm = config.get("attnres_fuse_read_norm", True),
            use_lrid = config["use_lrid"],
            lrid_rank = config["lrid_rank"],
            lrid_projection_rank = config.get("lrid_projection_rank", None),
            lrid_num_heads = config.get("lrid_num_heads", 1),
            lrid_input_dependent_query = config.get("lrid_input_dependent_query", False),
            lrid_static_embedding_key = config.get("lrid_static_embedding_key", False),
            lrid_add_static_embedding_key = config.get("lrid_add_static_embedding_key", False),
            lrid_add_static_source_key = config.get("lrid_add_static_source_key", False),
            lrid_key_from_value = config.get("lrid_key_from_value", False),
            lrid_key_from_value_shared = config.get("lrid_key_from_value_shared", False),
            lrid_key_from_output_tail = config.get("lrid_key_from_output_tail", False),
            lrid_key_value_norm = config.get("lrid_key_value_norm", True),
            lrid_query_from_value = config.get("lrid_query_from_value", False),
            lrid_query_from_value_shared = config.get("lrid_query_from_value_shared", False),
            lrid_use_logit_scale = config["lrid_use_logit_scale"],
            lrid_logit_scale = config["lrid_logit_scale"],
            )
        model = OBPM(model_config)
    else:
        raise Exception("Init_from has to be either 'scratch' or 'resume'")
        
    model.to(device)
    
    return start_step, checkpoint, model, model_config


def _get_dataloader_config(config):
    return DataLoaderConfig(
        data_dir=config["dataset_dir"],
        batch_size=config["batch_size"],
        block_size=config["block_size"],
        grad_accum_steps=config["grad_accum_steps"],
        use_doc_masking=config["use_doc_masking"],
        doc_separator_token=config["doc_separator_token"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"],
        prefetch_factor=int(config.get("prefetch_factor", 2)),
        dtype=np.dtype(config.get("data_dtype", "uint32")),
        rank=int(config.get("rank", 0)),
        world_size=int(config.get("world_size", 1)),
        seed=int(config.get("seed", 42)),
    )


def get_dataloader(config):
    dataloader_config = _get_dataloader_config(config)
    return create_dataloaders(dataloader_config)


def get_validation_dataloader(config):
    dataloader_config = _get_dataloader_config(config)
    return create_validation_dataloader(dataloader_config)


def get_training_evaluation_dataloader(config, train_dataset=None):
    dataloader_config = _get_dataloader_config(config)
    return create_training_evaluation_dataloader(dataloader_config, train_dataset)


def model_for_validation(model):
    """Remove DDP collectives while preserving a torch.compile wrapper."""
    return (
        model.module
        if isinstance(model, nn.parallel.DistributedDataParallel)
        else model
    )


def get_lm_head_for_loss(model):
    core_model = unwrap_model(model)
    return core_model.get_lm_head_weight(), core_model.get_lm_head_bias()


def _compute_chunked_lm_loss(
    hidden,
    y,
    criterion,
    lm_head_weight,
    lm_head_bias=None,
    cast_logits_to_float=True,
):
    labels = y.reshape(-1)
    hidden = hidden.reshape(-1, hidden.size(-1))
    chunk_size = int(getattr(criterion.config, "lm_head_chunk_size", 0))

    if chunk_size <= 0 or chunk_size >= hidden.size(0):
        logits = F.linear(hidden, lm_head_weight, lm_head_bias)
        logits = logits.float() if cast_logits_to_float else logits
        return criterion(logits, labels)

    ignore_index = getattr(criterion.config, "ignore_index", -100)
    reduction = getattr(criterion.config, "reduction", "mean")
    if reduction not in {"mean", "sum"}:
        logits = F.linear(hidden, lm_head_weight, lm_head_bias)
        logits = logits.float() if cast_logits_to_float else logits
        return criterion(logits, labels)

    total_loss = hidden.new_zeros((), dtype=torch.float32)
    total_tokens = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.size(0), chunk_size):
        end = min(start + chunk_size, hidden.size(0))
        logits = F.linear(hidden[start:end], lm_head_weight, lm_head_bias)
        logits = logits.float() if cast_logits_to_float else logits
        chunk_labels = labels[start:end]
        total_loss = total_loss + criterion.sum_loss(logits, chunk_labels)
        total_tokens = total_tokens + (chunk_labels != ignore_index).sum().to(total_tokens.dtype)

    if reduction == "sum":
        return total_loss
    return total_loss / total_tokens.clamp_min(1.0)


def compute_lm_loss(
    model,
    criterion,
    x,
    y,
    cu_seqlens=None,
    max_seqlen=None,
    cast_logits_to_float=True,
):
    chunk_size = int(getattr(getattr(criterion, "config", None), "lm_head_chunk_size", 0))
    if chunk_size > 0 and hasattr(unwrap_model(model), "get_lm_head_weight"):
        hidden = model(
            x,
            cu_doc_len=cu_seqlens,
            max_doc_len=max_seqlen,
            return_hidden=True,
        )
        lm_head_weight, lm_head_bias = get_lm_head_for_loss(model)
        return _compute_chunked_lm_loss(
            hidden,
            y,
            criterion,
            lm_head_weight,
            lm_head_bias=lm_head_bias,
            cast_logits_to_float=cast_logits_to_float,
        )

    logits = model(x, cu_doc_len=cu_seqlens, max_doc_len=max_seqlen)
    logits_for_loss = logits.float() if cast_logits_to_float else logits
    return criterion(logits_for_loss.view(-1, logits_for_loss.size(-1)), y.view(-1))


def loss_to_token_sum(loss, valid_tokens, reduction="mean"):
    if loss.dim() == 0:
        loss_value = float(loss.detach().item())
        if reduction == "sum":
            return loss_value
        return loss_value * valid_tokens
    return float(loss.detach().sum().item())


@torch.no_grad()
def compute_validation_loss(
    model,
    criterion,
    val_loader,
    device,
    vocab_size,
    use_doc_masking=True,
    desc="Validation loss",
    distributed=False,
):
    from tqdm import tqdm

    was_training = model.training
    # Remove only DDP. Keep torch.compile's OptimizedModule so training and
    # validation execute the same compiled forward implementation.
    forward_model = model_for_validation(model)
    model.eval()

    ignore_index = getattr(getattr(criterion, "config", None), "ignore_index", -100)
    reduction = getattr(getattr(criterion, "config", None), "reduction", "mean")
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    disable_tqdm = False
    if distributed:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            disable_tqdm = dist.get_rank() != 0

    try:
        for batch in tqdm(val_loader, desc=desc, leave=False, disable=disable_tqdm):
            if use_doc_masking:
                x, y, cu_seqlens, max_seqlen = batch
                cu_seqlens = cu_seqlens.to(device)
            else:
                x, y = batch[:2]
                cu_seqlens, max_seqlen = None, None

            if x.max() >= vocab_size or y.max() >= vocab_size:
                print("ERROR: Out-of-bounds token detected in validation batch!")
                print(f"  x min/max: {x.min()}/{x.max()}")
                print(f"  y min/max: {y.min()}/{y.max()}")
                raise ValueError("Out-of-bounds token detected in validation batch.")

            valid_tokens = int((y != ignore_index).sum().item())
            if valid_tokens == 0:
                continue

            x, y = x.to(device), y.to(device)

            loss = compute_lm_loss(
                forward_model,
                criterion,
                x,
                y,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            total_loss += loss_to_token_sum(loss, valid_tokens, reduction)

            total_tokens += valid_tokens
            total_batches += 1
    finally:
        if was_training:
            model.train()

    if distributed:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
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
        raise RuntimeError("No validation tokens available while computing validation loss.")

    return {
        "loss": total_loss / total_tokens,
        "tokens": total_tokens,
        "batches": total_batches,
    }
