#!/usr/bin/env python3
"""
Plot validation-set effective AttnRes contribution magnitudes for the n=8 Avg/no-Avg models.

The default probe records ||alpha_i v_i||_2 for each source i at each residual
read site, where alpha_i is the depth-attention weight and v_i is the source
value. This directly measures how much each source contributes to the residual
read.

The legacy raw-write probe is still available with --measure write. Its indexing is:
    0 = input token embedding
    1 = layer 0 attention output
    2 = layer 0 FFN output
    ...

Install:
    pip install huggingface-hub matplotlib tqdm

Prepare validation shards if needed:
    python prepdata.py --repo-id <your-hf-username>/Ultra-FineWeb-en-20B-gpt4

Run:
    python output_magni_analysis.py

Useful quick test:
    python output_magni_analysis.py --max-batches 2 --batch-size 1 --no-doc-masking

Outputs:
    figures/output_magnitude_across_layers_avg_vs_noavg.pdf
    figures/output_magnitude_across_layers_avg_vs_noavg.png
    figures/output_magnitude_across_layers_avg_vs_noavg.csv
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

_MPL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "lr_attnres_matplotlib_cache")
_XDG_CACHE_DIR = os.path.join(tempfile.gettempdir(), "lr_attnres_xdg_cache")
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.makedirs(_XDG_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from dataloader import DataLoaderConfig, create_validation_dataloader, warmup_boundaries
from model import OBPM, ModelConfig, norm
from utils import get_device


# =============================================================================
# User controls
# =============================================================================

MODEL_REPOS = {
    "avg": "Jonnester/LR-AttnRes-n8-Avg",
    "no_avg": "Jonnester/LR-AttnRes-n8",
}

DISPLAY_NAMES = {
    "avg": "Block LR-AttnRes n=8 + avg",
    "no_avg": "Block LR-AttnRes n=8",
}

COLORS = {
    "avg": "#3CB44B",      # green
    "no_avg": "#2E86DE",   # blue
}

CONTRIBUTION_LINESTYLES = {
    "completed": "-",
    "partial": "--",
    "embedding": ":",
    "previous": "-.",
    "all": "-",
}

CONTRIBUTION_DISPLAY_NAMES = {
    "completed": "completed",
    "partial": "partial",
    "embedding": "embedding",
    "previous": "previous",
    "all": "all",
}

PREFERRED_CHECKPOINT_FILES = (
    "final_model.pt",
    "model.pt",
    "checkpoint.pt",
)

OUTPUT_DIR = "figures"
OUTPUT_BASENAME = "output_magnitude_across_layers_avg_vs_noavg"

FIGSIZE = (7.2, 5.0)
DPI = 300
PANEL_LABEL = "(b)"


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update_tensor(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1)
        batch_count = int(values.numel())
        if batch_count == 0:
            return

        batch_mean = float(values.mean().item())
        batch_m2 = float(((values - batch_mean) ** 2).sum().item())
        self.update_batch(batch_count, batch_mean, batch_m2)

    def update_batch(self, batch_count: int, batch_mean: float, batch_m2: float) -> None:
        if batch_count == 0:
            return
        if self.count == 0:
            self.count = int(batch_count)
            self.mean = float(batch_mean)
            self.m2 = float(batch_m2)
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = int(total)

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5


@dataclass
class MagnitudeProfile:
    depth: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    count: np.ndarray
    label: str
    category: str = "all"

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 0.0))


@dataclass
class LoadedModel:
    model: OBPM
    model_config: ModelConfig
    train_config: dict[str, Any]
    checkpoint_path: str


# =============================================================================
# Hugging Face / checkpoint loading
# =============================================================================

def _checkpoint_sort_key(path: str) -> tuple[int, int, str]:
    match = re.search(r"(?:^|/)ckpt_step:(\d+)\.pt$", path)
    if match:
        return (0, int(match.group(1)), path)
    return (1, -1, path)


def choose_checkpoint_file(files: list[str]) -> str:
    available = set(files)
    for preferred in PREFERRED_CHECKPOINT_FILES:
        if preferred in available:
            return preferred

    step_checkpoints = [f for f in files if re.search(r"(?:^|/)ckpt_step:\d+\.pt$", f)]
    if step_checkpoints:
        return sorted(step_checkpoints, key=_checkpoint_sort_key)[-1]

    pt_files = sorted(f for f in files if f.endswith(".pt"))
    if len(pt_files) == 1:
        return pt_files[0]

    sample = "\n".join(f"  - {f}" for f in sorted(files)[:60])
    raise RuntimeError(
        "Could not choose a checkpoint file automatically. "
        "Pass --checkpoint-filename, --avg-checkpoint-filename, or "
        "--no-avg-checkpoint-filename.\n"
        f"First files in repo:\n{sample}"
    )


def resolve_checkpoint_path(
    repo_id: str,
    filename: Optional[str],
    revision: Optional[str],
    cache_dir: Optional[str],
    token: Optional[str],
    local_files_only: bool,
) -> str:
    if os.path.isfile(repo_id):
        return repo_id

    if filename is None:
        if local_files_only:
            filename = "final_model.pt"
        else:
            files = HfApi(token=token).list_repo_files(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
            )
            filename = choose_checkpoint_file(files)

    print(f"Downloading {repo_id}:{filename}")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> LoadedModel:
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_args = checkpoint.get("model_args", {})
    if isinstance(model_args, ModelConfig):
        model_config = model_args
    elif isinstance(model_args, dict):
        model_config = ModelConfig(**model_args)
    else:
        raise RuntimeError(f"Unsupported model_args type in checkpoint: {type(model_args)!r}")

    train_config = dict(checkpoint.get("config", {}))
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    if any(k.startswith(unwanted_prefix) for k in state_dict):
        state_dict = {
            k[len(unwanted_prefix):] if k.startswith(unwanted_prefix) else k: v
            for k, v in state_dict.items()
        }

    model = OBPM(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    del checkpoint
    gc.collect()

    return LoadedModel(
        model=model,
        model_config=model_config,
        train_config=train_config,
        checkpoint_path=checkpoint_path,
    )


# =============================================================================
# Validation data
# =============================================================================

def resolve_optional_bool(auto_value: bool, requested: Optional[bool]) -> bool:
    return bool(auto_value if requested is None else requested)


def build_validation_loader(
    train_config: dict[str, Any],
    model_config: ModelConfig,
    args: argparse.Namespace,
    device: torch.device,
):
    dataset_dir = args.dataset_dir or train_config.get("dataset_dir") or DataLoaderConfig.data_dir
    block_size = args.block_size or int(train_config.get("block_size", model_config.block_size))
    if block_size > model_config.block_size:
        raise ValueError(
            f"Requested block_size={block_size}, but model block_size={model_config.block_size}."
        )

    use_doc_masking = resolve_optional_bool(
        bool(train_config.get("use_doc_masking", True)),
        args.use_doc_masking,
    )
    pin_memory = resolve_optional_bool(device.type == "cuda", args.pin_memory)

    dataloader_config = DataLoaderConfig(
        data_dir=dataset_dir,
        batch_size=args.batch_size,
        block_size=block_size,
        grad_accum_steps=1,
        use_doc_masking=use_doc_masking,
        doc_separator_token=args.doc_separator_token
        if args.doc_separator_token is not None
        else train_config.get("doc_separator_token", 100257),
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor,
        dtype=np.dtype(args.data_dtype or train_config.get("data_dtype", "uint32")),
        rank=0,
        world_size=1,
    )

    val_loader = create_validation_dataloader(dataloader_config)
    if dataloader_config.use_doc_masking and args.warmup_boundaries:
        print("Warming up validation document boundary cache...")
        warmup_boundaries(val_loader.dataset, verbose=True)
        print("Validation boundary warmup complete.")
    return val_loader, dataloader_config.use_doc_masking


# =============================================================================
# Magnitude recording
# =============================================================================

def compute_magnitude(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.detach().float()
    if mode == "l2":
        return torch.linalg.vector_norm(x, ord=2, dim=-1)
    if mode == "rms":
        return x.square().mean(dim=-1).sqrt()
    if mode == "mean_abs":
        return x.abs().mean(dim=-1)
    raise ValueError(f"Unknown magnitude mode: {mode!r}")


def first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"Could not extract tensor output from {type(output)!r}")


class MagnitudeRecorder:
    def __init__(self, model: OBPM, measure: str, magnitude: str):
        self.model = model
        self.measure = measure
        self.magnitude = magnitude
        self.stats: dict[str, dict[int, RunningMoments]] = {}
        self.originals: list[tuple[Any, str, Any]] = []

    def update(self, depth_idx: int, tensor: torch.Tensor, category: str = "all") -> None:
        magnitudes = compute_magnitude(tensor, self.magnitude)
        category_stats = self.stats.setdefault(category, {})
        category_stats.setdefault(int(depth_idx), RunningMoments()).update_tensor(magnitudes)

    def _wrap_instance_method(self, obj: Any, name: str, wrapped: Any) -> None:
        original = getattr(obj, name)
        self.originals.append((obj, name, original))
        setattr(obj, name, wrapped(original))

    def install(self) -> None:
        if self.measure == "contribution":
            self._wrap_effective_contributions()
        elif self.measure == "read":
            self._wrap_read_outputs()
        elif self.measure == "write":
            self._wrap_write_outputs()
        else:
            raise ValueError(f"Unknown measure: {self.measure!r}")

    def restore(self) -> None:
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)
        self.originals.clear()

    def _wrap_read_outputs(self) -> None:
        if hasattr(self.model, "_apply_attnres"):
            def wrap_attnres(original):
                def wrapped(residual_idx, sources, *args, **kwargs):
                    output = original(residual_idx, sources, *args, **kwargs)
                    self.update(int(residual_idx), output)
                    return output
                return wrapped

            self._wrap_instance_method(self.model, "_apply_attnres", wrap_attnres)

        if hasattr(self.model, "_apply_lrid_attnres"):
            def wrap_lrid(original):
                def wrapped(residual_idx, sources, *args, **kwargs):
                    output = original(residual_idx, sources, *args, **kwargs)
                    self.update(int(residual_idx), output)
                    return output
                return wrapped

            self._wrap_instance_method(self.model, "_apply_lrid_attnres", wrap_lrid)

    def _wrap_write_outputs(self) -> None:
        def wrap_embedding(original):
            def wrapped(*args, **kwargs):
                output = original(*args, **kwargs)
                self.update(0, output)
                return output
            return wrapped

        self._wrap_instance_method(self.model.transformer.wte, "forward", wrap_embedding)

        for layer_idx, block in enumerate(self.model.transformer.layers):
            attn_depth = 2 * layer_idx + 1
            mlp_depth = 2 * layer_idx + 2

            def wrap_attention(original, depth_idx=attn_depth):
                def wrapped(*args, **kwargs):
                    output = original(*args, **kwargs)
                    self.update(depth_idx, first_tensor(output))
                    return output
                return wrapped

            def wrap_mlp(original, depth_idx=mlp_depth):
                def wrapped(*args, **kwargs):
                    output = original(*args, **kwargs)
                    self.update(depth_idx, first_tensor(output))
                    return output
                return wrapped

            self._wrap_instance_method(block.attn, "forward", wrap_attention)
            self._wrap_instance_method(block.mlp, "forward", wrap_mlp)

    def _source_categories(self, residual_idx: int, num_sources: int) -> list[str]:
        if num_sources == 1:
            return ["embedding"]

        if self.model.attnres_type == "block":
            block_ends = self.model.attnres_block_ends or frozenset()
            partial_present = residual_idx > 0 and residual_idx not in block_ends
            categories = []
            for source_idx in range(num_sources):
                if source_idx == 0:
                    categories.append("embedding")
                elif partial_present and source_idx == num_sources - 1:
                    categories.append("partial")
                else:
                    categories.append("completed")
            return categories

        return ["embedding"] + ["previous"] * (num_sources - 1)

    def _record_lrid_effective_contributions(
        self,
        residual_idx: int,
        sources: list[Any],
        query_override: Optional[torch.Tensor] = None,
    ) -> None:
        if len(sources) == 1:
            self.update(residual_idx, sources[0][0], category="embedding")
            self.update(residual_idx, sources[0][0], category="all")
            return

        values = torch.stack([source[0] for source in sources], dim=0)
        keys = torch.stack([source[1] for source in sources], dim=0)
        num_heads = self.model.config.lrid_num_heads
        key_head_dim = self.model.config.lrid_rank // num_heads
        value_head_dim = self.model.config.n_embd // num_heads

        keys = keys.reshape(*keys.shape[:-1], num_heads, key_head_dim)
        values_by_head = values.reshape(*values.shape[:-1], num_heads, value_head_dim)
        if self.model.config.attnres_key_norm:
            keys = norm(keys.float()).to(values_by_head.dtype)

        query_idx = self.model._attnres_query_idx(residual_idx)
        static_query = self.model.transformer.lrid_queries[query_idx]
        if self.model.config.lrid_input_dependent_query:
            dynamic_query = query_override if query_override is not None else sources[-1][2]
            if dynamic_query is None:
                raise RuntimeError("Input-dependent LR AttnRes query is missing for contribution recording")
            dynamic_query = dynamic_query.reshape(*dynamic_query.shape[:-1], num_heads, key_head_dim)
            gate = self.model.transformer.lrid_query_gates[query_idx].view(1, 1, num_heads, 1)
            query = static_query.unsqueeze(0).unsqueeze(0) + gate * dynamic_query
        else:
            query = static_query

        if self.model.config.attn_res_query_norm:
            query = norm(query.float())
        query = query.to(keys.dtype)

        if self.model.config.lrid_input_dependent_query:
            logits = torch.einsum("sbthr,bthr->sbth", keys, query) * self.model.config.lrid_logit_scale
        else:
            logits = torch.einsum("sbthr,hr->sbth", keys, query) * self.model.config.lrid_logit_scale
        weights = F.softmax(logits.float(), dim=0).to(values_by_head.dtype)
        contributions = (weights.unsqueeze(-1) * values_by_head).reshape_as(values)
        self.update(residual_idx, contributions, category="all")

        for source_idx, category in enumerate(self._source_categories(residual_idx, len(sources))):
            self.update(residual_idx, contributions[source_idx], category=category)

    def _record_attnres_effective_contributions(
        self,
        residual_idx: int,
        sources: list[torch.Tensor],
    ) -> None:
        if len(sources) == 1:
            self.update(residual_idx, sources[0], category="embedding")
            self.update(residual_idx, sources[0], category="all")
            return

        values = torch.stack(sources, dim=0)
        residual_module = self.model.transformer.attn_residuals[
            self.model._attnres_query_idx(residual_idx)
        ]
        keys = norm(values) if residual_module.use_key_norm else values
        logits = torch.einsum("d,sbtd->sbt", residual_module._query(keys.dtype), keys)
        weights = F.softmax(logits.float(), dim=0).to(values.dtype)
        contributions = weights.unsqueeze(-1) * values
        self.update(residual_idx, contributions, category="all")

        for source_idx, category in enumerate(self._source_categories(residual_idx, len(sources))):
            self.update(residual_idx, contributions[source_idx], category=category)

    def _wrap_effective_contributions(self) -> None:
        if hasattr(self.model, "_apply_attnres"):
            def wrap_attnres(original):
                def wrapped(residual_idx, sources, *args, **kwargs):
                    self._record_attnres_effective_contributions(int(residual_idx), sources)
                    return original(residual_idx, sources, *args, **kwargs)
                return wrapped

            self._wrap_instance_method(self.model, "_apply_attnres", wrap_attnres)

        if hasattr(self.model, "_apply_lrid_attnres"):
            def wrap_lrid(original):
                def wrapped(residual_idx, sources, *args, **kwargs):
                    query_override = kwargs.get("query_override", args[0] if args else None)
                    self._record_lrid_effective_contributions(
                        int(residual_idx),
                        sources,
                        query_override=query_override,
                    )
                    return original(residual_idx, sources, *args, **kwargs)
                return wrapped

            self._wrap_instance_method(self.model, "_apply_lrid_attnres", wrap_lrid)

    def profiles(
        self,
        label: str,
        categories: Optional[list[str]] = None,
    ) -> list[MagnitudeProfile]:
        if not self.stats:
            raise RuntimeError(
                "No activations were recorded. Check that the model path is correct "
                "and that --measure matches the architecture."
            )

        selected_categories = categories if categories is not None else sorted(self.stats)
        profiles = []
        for category in selected_categories:
            category_stats = self.stats.get(category)
            if not category_stats:
                continue

            depth = np.array(sorted(category_stats), dtype=np.int64)
            mean = np.array([category_stats[int(i)].mean for i in depth], dtype=np.float64)
            variance = np.array([category_stats[int(i)].variance for i in depth], dtype=np.float64)
            count = np.array([category_stats[int(i)].count for i in depth], dtype=np.int64)
            profiles.append(
                MagnitudeProfile(
                    depth=depth,
                    mean=mean,
                    variance=variance,
                    count=count,
                    label=label,
                    category=category,
                )
            )

        if not profiles:
            raise RuntimeError(f"No activations were recorded for categories: {selected_categories}")
        return profiles


@torch.no_grad()
def collect_profile(
    loaded: LoadedModel,
    val_loader,
    use_doc_masking: bool,
    args: argparse.Namespace,
    device: torch.device,
    label: str,
) -> list[MagnitudeProfile]:
    recorder = MagnitudeRecorder(
        loaded.model,
        measure=args.measure,
        magnitude=args.magnitude,
    )
    recorder.install()

    try:
        iterator = tqdm(
            val_loader,
            desc=f"Collecting {label}",
            leave=False,
            total=len(val_loader),
        )
        for batch_idx, batch in enumerate(iterator):
            x = batch[0].to(device, non_blocking=True)
            cu_doc_len = None
            max_doc_len = None

            if use_doc_masking:
                cu_doc_len = batch[2].to(device, non_blocking=True)
                max_doc_len = batch[3]

            loaded.model(
                x,
                cu_doc_len=cu_doc_len,
                max_doc_len=max_doc_len,
                return_hidden=True,
            )

            if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
                break
    finally:
        recorder.restore()

    categories = args.contribution_categories if args.measure == "contribution" else None
    return recorder.profiles(label, categories=categories)


# =============================================================================
# Plotting
# =============================================================================

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def style_axis(ax, x_label: str, y_label: str, panel_label: Optional[str] = None) -> None:
    ax.set_facecolor("#fbfbfb")

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        color="#cfcfcf",
        alpha=0.85,
        zorder=0,
    )

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#808080")
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(axis="both", colors="#555555", labelsize=9)
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")

    if panel_label is not None:
        ax.text(
            0.50,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="black",
        )


def apply_limits(
    ax,
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
) -> None:
    if x_min is not None or x_max is not None:
        current = ax.get_xlim()
        ax.set_xlim(
            x_min if x_min is not None else current[0],
            x_max if x_max is not None else current[1],
        )

    if y_min is not None or y_max is not None:
        current = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current[0],
            y_max if y_max is not None else current[1],
        )


def y_label_for_magnitude(mode: str) -> str:
    if mode == "l2":
        return "Output Magnitude (L2 Norm)"
    if mode == "rms":
        return "Output Magnitude (RMS)"
    if mode == "mean_abs":
        return "Output Magnitude (Mean Abs.)"
    return "Output Magnitude"


def x_label_for_measure(measure: str) -> str:
    if measure == "contribution":
        return "Residual Read Site"
    if measure == "read":
        return "Residual Depth Site"
    if measure == "write":
        return "Output Layer Index"
    return "Depth"


def y_label_for_measure(measure: str, magnitude: str) -> str:
    if measure == "contribution":
        if magnitude == "l2":
            return "Effective Contribution Magnitude (L2 Norm)"
        if magnitude == "rms":
            return "Effective Contribution Magnitude (RMS)"
        if magnitude == "mean_abs":
            return "Effective Contribution Magnitude (Mean Abs.)"
        return "Effective Contribution Magnitude"
    return y_label_for_magnitude(magnitude)


def title_for_measure(measure: str, band: str) -> str:
    title_suffix = "shaded +/- 1 std. dev." if band == "std" else "shaded +/- variance"
    if measure == "contribution":
        return f"Effective AttnRes Contribution Across Read Sites ({title_suffix})"
    if measure == "write":
        return f"Output Magnitude Across Layers ({title_suffix})"
    return f"Residual Read Magnitude Across Depth ({title_suffix})"


def plot_profiles(
    profiles: dict[str, list[MagnitudeProfile]],
    args: argparse.Namespace,
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for key in ["no_avg", "avg"]:
        for profile in profiles[key]:
            color = COLORS[key]
            linestyle = CONTRIBUTION_LINESTYLES.get(profile.category, "-")
            category_label = CONTRIBUTION_DISPLAY_NAMES.get(profile.category, profile.category)
            label = DISPLAY_NAMES[key] if profile.category == "all" else f"{DISPLAY_NAMES[key]} ({category_label})"
            band = profile.std if args.band == "std" else profile.variance
            lower = np.maximum(profile.mean - band, 0.0)
            upper = profile.mean + band

            ax.fill_between(
                profile.depth,
                lower,
                upper,
                color=color,
                alpha=0.08 if profile.category != "all" else 0.12,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                profile.depth,
                profile.mean,
                color=color,
                linewidth=2.45 if profile.category in {"all", "completed"} else 2.05,
                linestyle=linestyle,
                alpha=0.97,
                label=label,
                zorder=3,
            )

    style_axis(
        ax,
        x_label=x_label_for_measure(args.measure),
        y_label=y_label_for_measure(args.measure, args.magnitude),
        panel_label=args.panel_label,
    )

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9,
        borderpad=0.7,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_alpha(0.95)

    ax.set_title(title_for_measure(args.measure, args.band), fontsize=12, fontweight="bold", pad=10)

    apply_limits(ax, args.x_min, args.x_max, args.y_min, args.y_max)
    fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, f"{args.output_basename}.pdf")
    png_path = os.path.join(args.output_dir, f"{args.output_basename}.png")

    fig.savefig(pdf_path, bbox_inches="tight", dpi=DPI)
    fig.savefig(png_path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def save_profiles_csv(profiles: dict[str, list[MagnitudeProfile]], args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{args.output_basename}.csv")

    keys = ["no_avg", "avg"]
    rows = ["model_key,display_name,source_category,depth,mean,variance,std,count"]
    for key in keys:
        for profile in profiles[key]:
            for depth, mean, variance, std, count in zip(
                profile.depth,
                profile.mean,
                profile.variance,
                profile.std,
                profile.count,
            ):
                rows.append(
                    f"{key},{DISPLAY_NAMES[key]},{profile.category},{int(depth)},"
                    f"{mean:.10g},{variance:.10g},{std:.10g},{int(count)}"
                )

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Saved {csv_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_optional_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def parse_category_list(value: str) -> list[str]:
    categories = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"completed", "partial", "embedding", "previous", "all"}
    unknown = sorted(set(categories) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown contribution categories {unknown}; allowed: {sorted(allowed)}"
        )
    if not categories:
        raise argparse.ArgumentTypeError("at least one contribution category is required")
    return categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Avg/no-Avg LR-AttnRes checkpoints from Hugging Face, "
            "run them over the validation set, and plot effective AttnRes contribution magnitudes."
        )
    )
    parser.add_argument("--avg-repo", type=str, default=MODEL_REPOS["avg"])
    parser.add_argument("--no-avg-repo", type=str, default=MODEL_REPOS["no_avg"])
    parser.add_argument("--checkpoint-filename", type=str, default=None)
    parser.add_argument("--avg-checkpoint-filename", type=str, default=None)
    parser.add_argument("--no-avg-checkpoint-filename", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--data-dtype", type=str, default=None)
    parser.add_argument("--doc-separator-token", type=int, default=None)
    parser.add_argument("--use-doc-masking", dest="use_doc_masking", type=parse_optional_bool, nargs="?", const=True, default=None)
    parser.add_argument("--no-doc-masking", dest="use_doc_masking", action="store_false")
    parser.add_argument("--warmup-boundaries", type=parse_optional_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-warmup-boundaries", dest="warmup_boundaries", action="store_false")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", type=parse_optional_bool, nargs="?", const=True, default=None)
    parser.add_argument("--persistent-workers", type=parse_optional_bool, nargs="?", const=True, default=False)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--measure", choices=("contribution", "read", "write"), default="contribution")
    parser.add_argument("--magnitude", choices=("l2", "rms", "mean_abs"), default="l2")
    parser.add_argument("--band", choices=("std", "variance"), default="std")
    parser.add_argument(
        "--contribution-categories",
        type=parse_category_list,
        default=parse_category_list("completed,partial"),
        help=(
            "Comma-separated source categories to plot for --measure contribution. "
            "Use completed,partial by default; other options are embedding,previous,all."
        ),
    )

    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--output-basename", type=str, default=OUTPUT_BASENAME)
    parser.add_argument("--panel-label", type=str, default=PANEL_LABEL)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    return parser.parse_args()


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return get_device(distributed=False)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float16":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def free_loaded_model(loaded: Optional[LoadedModel]) -> None:
    if loaded is None:
        return
    del loaded.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"Using device={device}, dtype={dtype}")

    checkpoint_paths = {
        "avg": resolve_checkpoint_path(
            args.avg_repo,
            args.avg_checkpoint_filename or args.checkpoint_filename,
            args.revision,
            args.cache_dir,
            args.hf_token,
            args.local_files_only,
        ),
        "no_avg": resolve_checkpoint_path(
            args.no_avg_repo,
            args.no_avg_checkpoint_filename or args.checkpoint_filename,
            args.revision,
            args.cache_dir,
            args.hf_token,
            args.local_files_only,
        ),
    }

    profiles: dict[str, list[MagnitudeProfile]] = {}
    val_loader = None
    use_doc_masking = None

    for key in ["avg", "no_avg"]:
        loaded: Optional[LoadedModel] = None
        try:
            loaded = load_model_from_checkpoint(checkpoint_paths[key], device=device, dtype=dtype)
            print(
                f"{key}: n_layer={loaded.model_config.n_layer}, "
                f"attnres_type={loaded.model_config.attnres_type}, "
                f"block_average={loaded.model_config.attnres_block_average}"
            )

            if val_loader is None:
                val_loader, use_doc_masking = build_validation_loader(
                    loaded.train_config,
                    loaded.model_config,
                    args,
                    device,
                )

            profiles[key] = collect_profile(
                loaded,
                val_loader,
                bool(use_doc_masking),
                args,
                device,
                DISPLAY_NAMES[key],
            )
        finally:
            free_loaded_model(loaded)

    save_profiles_csv(profiles, args)
    plot_profiles(profiles, args)


if __name__ == "__main__":
    main()
