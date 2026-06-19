#!/usr/bin/env python3
"""
Depthwise routing analysis for AttnRes and LR-AttnRes checkpoints.

The script downloads Jonnester checkpoints, reconstructs depthwise routing
weights during validation forward passes, and writes 300 DPI academic-style
figures plus CSV/JSON metrics.

Examples:
    python analyze_depthwise_routing.py --quick --repos \
        Jonnester/LR-AttnRes-n8 \
        Jonnester/LR-AttnRes-LR-n8-r32 \
        Jonnester/LR-AttnRes-tail-r64-n8

    python analyze_depthwise_routing.py --model-scope paper_core
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_MPL_CACHE_DIR = os.path.join(tempfile.gettempdir(), "lr_attnres_matplotlib_cache")
_XDG_CACHE_DIR = os.path.join(tempfile.gettempdir(), "lr_attnres_xdg_cache")
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.makedirs(_XDG_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE_DIR)

import numpy as np
import torch
from torch.nn import functional as F
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

try:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise
    print(
        "Missing dependency: matplotlib\n\n"
        "Install it with:\n"
        f"  {sys.executable} -m pip install matplotlib",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

try:
    from adjustText import adjust_text
except ModuleNotFoundError:
    adjust_text = None

from dataloader import DataLoaderConfig, create_validation_dataloader, warmup_boundaries
from model import ModelConfig, OBPM, norm
from utils import get_device


# =============================================================================
# User-facing defaults
# =============================================================================

OUTPUT_DIR = "figures/depthwise_analysis"
DPI = 300
FIGSIZE = (6.5, 4.85)

PREFERRED_CHECKPOINT_FILES = (
    "final_model.pt",
    "model.pt",
    "checkpoint.pt",
    "ckpt.pt",
    "pytorch_model.bin",
)

PAPER_CORE_REPOS = [
    "Jonnester/LR-AttnRes-Baseline",
    "Jonnester/LR-AttnRes-Full",
    "Jonnester/LR-AttnRes-n4",
    "Jonnester/LR-AttnRes-n8",
    "Jonnester/LR-AttnRes-n16",
    "Jonnester/LR-AttnRes-Full-16",
    "Jonnester/LR-AttnRes-Full-32",
    "Jonnester/LR-AttnRes-Full-64",
    "Jonnester/LR-AttnRes-n4-r32",
    "Jonnester/LR-AttnRes-LR-n8-r32",
    "Jonnester/LR-AttnRes-n16-r32",
    "Jonnester/LR-AttnRes-tail-r16",
    "Jonnester/LR-AttnRes-tail-r32",
    "Jonnester/LR-AttnRes-tail-r64",
    "Jonnester/LR-AttnRes-tail-r128",
    "Jonnester/LR-AttnRes-tail-r256",
    "Jonnester/LR-AttnRes-tail-r512",
    "Jonnester/LR-AttnRes-tail-r64-n4",
    "Jonnester/LR-AttnRes-tail-r64-n8",
    "Jonnester/LR-AttnRes-tail-r64-n16",
]

PAPER_VAL_LOSS = {
    "Jonnester/LR-AttnRes-Baseline": 3.0009,
    "Jonnester/LR-AttnRes-Full": 2.9752,
    "Jonnester/LR-AttnRes-n4": 2.9797,
    "Jonnester/LR-AttnRes-n8": 2.9778,
    "Jonnester/LR-AttnRes-n16": 2.9673,
    "Jonnester/LR-AttnRes-Full-16": 2.9606,
    "Jonnester/LR-AttnRes-Full-32": 2.9538,
    "Jonnester/LR-AttnRes-Full-64": 2.9501,
    "Jonnester/LR-AttnRes-n4-r32": 2.9488,
    "Jonnester/LR-AttnRes-LR-n8-r32": 2.9477,
    "Jonnester/LR-AttnRes-n16-r32": 2.9543,
    "Jonnester/LR-AttnRes-tail-r16": 2.9762,
    "Jonnester/LR-AttnRes-tail-r32": 2.9682,
    "Jonnester/LR-AttnRes-tail-r64": 2.9638,
    "Jonnester/LR-AttnRes-tail-r128": 2.9630,
    "Jonnester/LR-AttnRes-tail-r256": 2.9626,
    "Jonnester/LR-AttnRes-tail-r512": 2.9617,
    "Jonnester/LR-AttnRes-tail-r64-n4": 2.9533,
    "Jonnester/LR-AttnRes-tail-r64-n8": 2.9480,
    "Jonnester/LR-AttnRes-tail-r64-n16": 2.9494,
}

PAPER_ADDED_FLOPS = {
    "Jonnester/LR-AttnRes-Baseline": 0.0000,
    "Jonnester/LR-AttnRes-Full": 0.813138,
    "Jonnester/LR-AttnRes-n4": 0.111607,
    "Jonnester/LR-AttnRes-n8": 0.175383,
    "Jonnester/LR-AttnRes-n16": 0.302934,
    "Jonnester/LR-AttnRes-Full-16": 0.896552,
    "Jonnester/LR-AttnRes-Full-32": 1.386536,
    "Jonnester/LR-AttnRes-Full-64": 2.366503,
    "Jonnester/LR-AttnRes-n4-r32": 1.024809,
    "Jonnester/LR-AttnRes-LR-n8-r32": 1.057694,
    "Jonnester/LR-AttnRes-n16-r32": 1.123462,
    "Jonnester/LR-AttnRes-tail-r16": 0.412922,
    "Jonnester/LR-AttnRes-tail-r32": 0.419274,
    "Jonnester/LR-AttnRes-tail-r64": 0.431980,
    "Jonnester/LR-AttnRes-tail-r128": 0.457390,
    "Jonnester/LR-AttnRes-tail-r256": 0.508211,
    "Jonnester/LR-AttnRes-tail-r512": 0.609853,
    "Jonnester/LR-AttnRes-tail-r64-n4": 0.059291,
    "Jonnester/LR-AttnRes-tail-r64-n8": 0.093172,
    "Jonnester/LR-AttnRes-tail-r64-n16": 0.160934,
}

STYLE = OrderedDict(
    [
        (
            "baseline_transformer",
            {
                "name": "Baseline Transformer",
                "color": "#7B2CBF",
                "marker": "s",
                "size": 88,
            },
        ),
        (
            "standard_attnres",
            {
                "name": "Standard AttnRes",
                "color": "#2E86DE",
                "marker": "D",
                "size": 92,
            },
        ),
        (
            "sliced_lr_attnres",
            {
                "name": "Sliced LR-AttnRes",
                "color": "#F39C12",
                "marker": "^",
                "size": 94,
            },
        ),
        (
            "lr_attnres",
            {
                "name": "Projected LR-AttnRes",
                "color": "#E84A5F",
                "marker": "o",
                "size": 84,
            },
        ),
    ]
)

LEGEND_ORDER = [
    "baseline_transformer",
    "standard_attnres",
    "sliced_lr_attnres",
    "lr_attnres",
]


# =============================================================================
# Small utilities
# =============================================================================

def set_academic_rcparams() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def style_axis(ax, x_label: str, y_label: str) -> None:
    ax.set_facecolor("#FAFAFA")
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.62,
        color="#D0D0D0",
        alpha=0.90,
    )
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#8A8A8A")
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(axis="both", colors="#555555", length=3.0, width=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def legend_handles(groups: Optional[list[str]] = None) -> list[Line2D]:
    handles: list[Line2D] = []
    for group in groups or LEGEND_ORDER:
        if group not in STYLE:
            continue
        style = STYLE[group]
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=style["marker"],
                markersize=8.5,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                label=style["name"],
            )
        )
    return handles


def finish_legend(ax, handles: Optional[list[Line2D]] = None, loc: str = "best") -> None:
    legend = ax.legend(
        handles=handles,
        loc=loc,
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=0.65,
        handletextpad=0.6,
        labelspacing=0.45,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#C8C8C8")
    legend.get_frame().set_linewidth(0.9)
    legend.get_frame().set_alpha(0.96)


def finish_outside_legend(ax, handles: Optional[list[Line2D]] = None) -> None:
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=0.65,
        handletextpad=0.6,
        labelspacing=0.45,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#C8C8C8")
    legend.get_frame().set_linewidth(0.9)
    legend.get_frame().set_alpha(0.96)


def save_figure(fig, output_dir: Path, basename: str, tight: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(pad=0.8)
    for ext in ("pdf", "png", "svg"):
        path = output_dir / f"{basename}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=DPI)
        print(f"Saved {path}")
    plt.close(fig)


def sanitize_filename(text: str) -> str:
    text = text.replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
    return text.strip("_")


def checkpoint_sort_key(path: str) -> tuple[int, int, str]:
    match = re.search(r"(?:^|/)ckpt_step:(\d+)\.pt$", path)
    if match:
        return (0, int(match.group(1)), path)
    return (1, -1, path)


def choose_checkpoint_file(files: list[str]) -> str:
    available = set(files)
    for filename in PREFERRED_CHECKPOINT_FILES:
        if filename in available:
            return filename
    step_checkpoints = [f for f in files if re.search(r"(?:^|/)ckpt_step:\d+\.pt$", f)]
    if step_checkpoints:
        return sorted(step_checkpoints, key=checkpoint_sort_key)[-1]
    model_files = sorted(
        f
        for f in files
        if f.endswith((".pt", ".pth", ".bin"))
        and any(token in Path(f).name.lower() for token in ("ckpt", "checkpoint", "model"))
    )
    if len(model_files) == 1:
        return model_files[0]
    raise RuntimeError("No obvious checkpoint file found")


def strip_compiled_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def parse_optional_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


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


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def group_from_config(config: ModelConfig) -> str:
    if not config.use_attnres:
        return "baseline_transformer"
    if not config.use_lrid:
        return "standard_attnres"
    if config.lrid_key_from_output_tail:
        return "sliced_lr_attnres"
    return "lr_attnres"


def label_from_config(repo_id: str, config: ModelConfig) -> str:
    group = group_from_config(config)
    if group == "baseline_transformer":
        return "Baseline"
    if config.attnres_type == "full":
        base = "Full"
    else:
        base = f"n={config.attnres_num_blocks}"
    if config.use_lrid:
        base = f"{base} r={config.lrid_rank}"
    return base


def model_sort_key(result: "AnalysisResult") -> tuple[int, int, int, str]:
    group_idx = LEGEND_ORDER.index(result.group) if result.group in LEGEND_ORDER else 99
    full_idx = 0 if result.attnres_type == "full" else 1
    rank = result.lrid_rank if result.lrid_rank is not None else 0
    blocks = result.attnres_num_blocks if result.attnres_num_blocks is not None else 0
    return group_idx, full_idx, blocks, rank, result.repo_id


def participation_ratio(row: np.ndarray) -> float:
    energy = row * row
    denom = float(np.sum(energy * energy))
    if denom <= 0.0:
        return 0.0
    return float((np.sum(energy) ** 2) / denom)


def hoyer_sparsity(row: np.ndarray) -> float:
    row_abs = np.abs(row)
    l2 = float(np.linalg.norm(row_abs))
    if l2 <= 0.0:
        return 1.0
    n = row_abs.size
    if n <= 1:
        return 0.0
    return float((math.sqrt(n) - np.sum(row_abs) / l2) / (math.sqrt(n) - 1.0))


def dims_for_energy(row: np.ndarray, fraction: float) -> int:
    energy = np.sort(row * row)[::-1]
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0
    cumulative = np.cumsum(energy)
    return int(np.searchsorted(cumulative, fraction * total, side="left") + 1)


def pearson_from_arrays(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or y.size < 2:
        return None
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return None
    return float(np.dot(x, y) / denom)


def entropy_np(p: np.ndarray) -> float:
    p = p.astype(np.float64)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


# =============================================================================
# Loading and data
# =============================================================================

@dataclass
class LoadedModel:
    repo_id: str
    checkpoint_path: str
    model: OBPM
    model_config: ModelConfig
    train_config: dict[str, Any]


def resolve_checkpoint_path(
    repo_id: str,
    filename: Optional[str],
    revision: Optional[str],
    cache_dir: Optional[str],
    token: Optional[str],
    local_files_only: bool,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if os.path.isfile(repo_id):
        return repo_id, os.path.basename(repo_id), None

    try:
        if filename is None:
            files = HfApi(token=token).list_repo_files(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
            )
            filename = choose_checkpoint_file(files)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        return path, filename, None
    except Exception as exc:  # noqa: BLE001 - report and keep the run moving.
        return None, filename, str(exc)


def load_model_from_checkpoint(
    repo_id: str,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> LoadedModel:
    print(f"Loading checkpoint: {repo_id} -> {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("model_args", {})
    if isinstance(model_args, ModelConfig):
        model_config = model_args
    elif isinstance(model_args, dict):
        model_config = ModelConfig(**model_args)
    else:
        raise RuntimeError(f"Unsupported model_args type: {type(model_args)!r}")

    state_dict = strip_compiled_prefix(checkpoint["model"])
    model = OBPM(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    train_config = checkpoint.get("config", {})
    if not isinstance(train_config, dict):
        train_config = {}

    del checkpoint, state_dict
    gc.collect()
    return LoadedModel(
        repo_id=repo_id,
        checkpoint_path=checkpoint_path,
        model=model,
        model_config=model_config,
        train_config=dict(train_config),
    )


def prepare_model_for_analysis(model: OBPM) -> dict[str, Any]:
    original = {
        "config_use_fused_attnres": model.config.use_fused_attnres,
        "use_fused_attnres": getattr(model, "use_fused_attnres", None),
    }
    model.config.use_fused_attnres = False
    if hasattr(model, "use_fused_attnres"):
        model.use_fused_attnres = False
    return original


def restore_model_after_analysis(model: OBPM, original: dict[str, Any]) -> None:
    model.config.use_fused_attnres = original["config_use_fused_attnres"]
    if original["use_fused_attnres"] is not None and hasattr(model, "use_fused_attnres"):
        model.use_fused_attnres = original["use_fused_attnres"]


def free_loaded_model(loaded: Optional[LoadedModel]) -> None:
    if loaded is None:
        return
    del loaded.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_validation_loader(
    train_config: dict[str, Any],
    model_config: ModelConfig,
    args: argparse.Namespace,
    device: torch.device,
):
    dataset_dir = args.dataset_dir or train_config.get("dataset_dir") or DataLoaderConfig.data_dir
    block_size = args.block_size or int(train_config.get("block_size", model_config.block_size))
    block_size = min(block_size, model_config.block_size)
    use_doc_masking = bool(train_config.get("use_doc_masking", True))
    if args.use_doc_masking is not None:
        use_doc_masking = bool(args.use_doc_masking)

    config = DataLoaderConfig(
        data_dir=dataset_dir,
        batch_size=args.batch_size,
        block_size=block_size,
        grad_accum_steps=1,
        use_doc_masking=use_doc_masking,
        doc_separator_token=args.doc_separator_token
        if args.doc_separator_token is not None
        else train_config.get("doc_separator_token", 100257),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda") if args.pin_memory is None else bool(args.pin_memory),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor,
        dtype=np.dtype(args.data_dtype or train_config.get("data_dtype", "uint32")),
        rank=0,
        world_size=1,
    )
    loader = create_validation_dataloader(config)
    if config.use_doc_masking and args.warmup_boundaries:
        print("Warming validation document boundaries...")
        warmup_boundaries(loader.dataset, verbose=True)
    return loader, config.use_doc_masking


# =============================================================================
# Running stats
# =============================================================================

@dataclass
class RunningScalar:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def update_tensor(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self.update_sums(
            int(values.numel()),
            float(values.sum().item()),
            float(values.square().sum().item()),
        )

    def update_sums(self, count: int, total: float, total_sq: float) -> None:
        if count <= 0:
            return
        self.count += int(count)
        self.total += float(total)
        self.total_sq += float(total_sq)

    @property
    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.total / self.count

    @property
    def variance(self) -> Optional[float]:
        if self.count == 0:
            return None
        mean = self.total / self.count
        return max(self.total_sq / self.count - mean * mean, 0.0)

    @property
    def std(self) -> Optional[float]:
        var = self.variance
        return None if var is None else math.sqrt(var)


@dataclass
class RunningVector:
    total: Optional[np.ndarray] = None
    total_sq: Optional[np.ndarray] = None
    count: int = 0

    def update_from_tensor(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float().reshape(-1, tensor.size(-1))
        if values.numel() == 0:
            return
        sums = values.sum(dim=0).cpu().numpy().astype(np.float64)
        sums_sq = values.square().sum(dim=0).cpu().numpy().astype(np.float64)
        self.update_sums(sums, sums_sq, values.size(0))

    def update_sums(self, sums: np.ndarray, sums_sq: np.ndarray, count: int) -> None:
        if count <= 0:
            return
        if self.total is None:
            self.total = np.zeros_like(sums, dtype=np.float64)
            self.total_sq = np.zeros_like(sums_sq, dtype=np.float64)
        self.total += sums
        self.total_sq += sums_sq
        self.count += int(count)

    def mean(self) -> Optional[np.ndarray]:
        if self.total is None or self.count == 0:
            return None
        return self.total / self.count

    def variance(self) -> Optional[np.ndarray]:
        if self.total is None or self.total_sq is None or self.count == 0:
            return None
        mean = self.total / self.count
        return np.maximum(self.total_sq / self.count - mean * mean, 0.0)


@dataclass
class RunningPair:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1)
        y = y.detach().float().reshape(-1)
        if x.numel() == 0 or y.numel() == 0:
            return
        if x.numel() != y.numel():
            n = min(x.numel(), y.numel())
            x = x[:n]
            y = y[:n]
        self.count += int(x.numel())
        self.sum_x += float(x.sum().item())
        self.sum_y += float(y.sum().item())
        self.sum_x2 += float(x.square().sum().item())
        self.sum_y2 += float(y.square().sum().item())
        self.sum_xy += float((x * y).sum().item())

    def correlation(self) -> Optional[float]:
        if self.count <= 1:
            return None
        n = float(self.count)
        cov = self.sum_xy - self.sum_x * self.sum_y / n
        var_x = self.sum_x2 - self.sum_x * self.sum_x / n
        var_y = self.sum_y2 - self.sum_y * self.sum_y / n
        denom = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
        if denom <= 0.0:
            return None
        return cov / denom


@dataclass
class SiteStats:
    residual_idx: int
    max_sources: int = 0
    source_weight_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    source_weight_sq_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    source_logit_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    source_logit_sq_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    source_count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    entropy: RunningScalar = field(default_factory=RunningScalar)
    normalized_entropy: RunningScalar = field(default_factory=RunningScalar)
    effective_sources: RunningScalar = field(default_factory=RunningScalar)
    top1_mass: RunningScalar = field(default_factory=RunningScalar)
    top3_mass: RunningScalar = field(default_factory=RunningScalar)
    top5_mass: RunningScalar = field(default_factory=RunningScalar)
    gini: RunningScalar = field(default_factory=RunningScalar)
    simpson_concentration: RunningScalar = field(default_factory=RunningScalar)
    category_mass: dict[str, RunningScalar] = field(default_factory=dict)
    contribution_rms: RunningScalar = field(default_factory=RunningScalar)

    def ensure_sources(self, n_sources: int) -> None:
        if n_sources <= self.max_sources:
            return
        pad = n_sources - self.max_sources
        self.source_weight_sum = np.pad(self.source_weight_sum, (0, pad))
        self.source_weight_sq_sum = np.pad(self.source_weight_sq_sum, (0, pad))
        self.source_logit_sum = np.pad(self.source_logit_sum, (0, pad))
        self.source_logit_sq_sum = np.pad(self.source_logit_sq_sum, (0, pad))
        self.source_count = np.pad(self.source_count, (0, pad))
        self.max_sources = n_sources

    def update_sources(
        self,
        weight_sum: np.ndarray,
        weight_sq_sum: np.ndarray,
        logit_sum: np.ndarray,
        logit_sq_sum: np.ndarray,
        count: int,
    ) -> None:
        n_sources = int(weight_sum.shape[0])
        self.ensure_sources(n_sources)
        self.source_weight_sum[:n_sources] += weight_sum
        self.source_weight_sq_sum[:n_sources] += weight_sq_sum
        self.source_logit_sum[:n_sources] += logit_sum
        self.source_logit_sq_sum[:n_sources] += logit_sq_sum
        self.source_count[:n_sources] += int(count)


# =============================================================================
# Recording internals
# =============================================================================

class RoutingRecorder:
    def __init__(self, model: OBPM, max_pair_samples: int = 512, validate_outputs: bool = False):
        self.model = model
        self.max_pair_samples = int(max_pair_samples)
        self.validate_outputs = bool(validate_outputs)
        self.originals: list[tuple[Any, str, Any]] = []
        self.sites: dict[int, SiteStats] = {}
        self.key_raw_rms = RunningScalar()
        self.score_key_abs = RunningScalar()
        self.value_rms = RunningScalar()
        self.tail_energy_fraction = RunningScalar()
        self.key_value_norm_corr = RunningPair()
        self.score_key_dim = RunningVector()
        self.output_abs_error = RunningScalar()
        self.output_rel_error = RunningScalar()
        self.sample_values: list[torch.Tensor] = []
        self.sample_keys: list[torch.Tensor] = []
        self.distributions_seen = 0

    def install(self) -> None:
        if hasattr(self.model, "_apply_attnres"):
            self._wrap_instance_method(self.model, "_apply_attnres", self._wrap_attnres)
        if hasattr(self.model, "_apply_lrid_attnres"):
            self._wrap_instance_method(self.model, "_apply_lrid_attnres", self._wrap_lrid)

    def restore(self) -> None:
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)
        self.originals.clear()

    def _wrap_instance_method(self, obj: Any, name: str, wrapper: Any) -> None:
        original = getattr(obj, name)
        self.originals.append((obj, name, original))
        setattr(obj, name, wrapper(original))

    def _wrap_attnres(self, original):
        def wrapped(residual_idx, sources, *args, **kwargs):
            expected = self.record_attnres(int(residual_idx), sources, *args, **kwargs)
            output = original(residual_idx, sources, *args, **kwargs)
            self._record_output_error(expected, output)
            return output

        return wrapped

    def _wrap_lrid(self, original):
        def wrapped(residual_idx, sources, *args, **kwargs):
            expected = self.record_lrid(int(residual_idx), sources, *args, **kwargs)
            output = original(residual_idx, sources, *args, **kwargs)
            self._record_output_error(expected, output)
            return output

        return wrapped

    def _record_output_error(self, expected: Optional[torch.Tensor], output: torch.Tensor) -> None:
        if not self.validate_outputs or expected is None:
            return
        diff = (expected.detach().float() - output.detach().float()).abs()
        abs_max = diff.max()
        denom = output.detach().float().abs().max().clamp_min(1e-8)
        rel = abs_max / denom
        self.output_abs_error.update_tensor(abs_max.reshape(1))
        self.output_rel_error.update_tensor(rel.reshape(1))

    def source_categories(self, residual_idx: int, num_sources: int) -> list[str]:
        if num_sources <= 0:
            return []
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

    def _add_biases(
        self,
        logits: torch.Tensor,
        source_counts=None,
        source_logit_biases=None,
    ) -> torch.Tensor:
        if source_counts is not None and source_logit_biases is not None:
            raise RuntimeError("source_counts and source_logit_biases are mutually exclusive")
        if source_counts is not None:
            log_counts = torch.as_tensor(source_counts, device=logits.device, dtype=torch.float32).log()
            view_shape = (logits.size(0),) + (1,) * (logits.ndim - 1)
            logits = logits + log_counts.view(view_shape)
        if source_logit_biases is not None:
            bias_values = [
                bias.to(device=logits.device, dtype=torch.float32).reshape(())
                if torch.is_tensor(bias)
                else torch.tensor(float(bias), device=logits.device, dtype=torch.float32)
                for bias in source_logit_biases
            ]
            logit_bias = torch.stack(bias_values)
            view_shape = (logits.size(0),) + (1,) * (logits.ndim - 1)
            logits = logits + logit_bias.view(view_shape)
        return logits

    def record_attnres(
        self,
        residual_idx: int,
        sources: list[torch.Tensor],
        normalize_output: bool = False,
        average_read: bool = False,
        source_counts=None,
        source_logit_biases=None,
    ) -> Optional[torch.Tensor]:
        if average_read or len(sources) < 2:
            return None
        values = torch.stack(sources, dim=0)
        residual = self.model.transformer.attn_residuals[self.model._attnres_query_idx(residual_idx)]
        score_keys = norm(values.float()).to(values.dtype) if residual.use_key_norm else values
        logits = torch.einsum("d,sbtd->sbt", residual._query(score_keys.dtype), score_keys)
        logits = self._add_biases(logits.float(), source_counts, source_logit_biases)
        weights = F.softmax(logits, dim=0)
        expected = torch.einsum("sbt,sbtd->btd", weights.to(values.dtype), values)
        if normalize_output:
            expected = norm(expected)

        self.record_common(
            residual_idx=residual_idx,
            values=values,
            raw_keys=values.unsqueeze(3),
            score_keys=score_keys.unsqueeze(3),
            logits=logits.unsqueeze(-1),
            weights=weights.unsqueeze(-1),
        )
        return expected

    def record_lrid(
        self,
        residual_idx: int,
        sources: list[Any],
        query_override: Optional[torch.Tensor] = None,
        normalize_output: bool = False,
        average_read: bool = False,
        source_counts=None,
        source_logit_biases=None,
    ) -> Optional[torch.Tensor]:
        if average_read or len(sources) < 2:
            return None

        value_sources = [source[0] for source in sources]
        key_sources = [source[1] for source in sources]
        values = torch.stack(value_sources, dim=0)
        raw_keys_flat = torch.stack(key_sources, dim=0)
        num_heads = self.model.config.lrid_num_heads
        key_head_dim = self.model.config.lrid_rank // num_heads
        value_head_dim = self.model.config.n_embd // num_heads

        raw_keys = raw_keys_flat.reshape(*raw_keys_flat.shape[:-1], num_heads, key_head_dim)
        values_by_head = values.reshape(*values.shape[:-1], num_heads, value_head_dim)
        score_keys = raw_keys
        if self.model.config.attnres_key_norm:
            score_keys = norm(raw_keys.float()).to(values_by_head.dtype)

        query_idx = self.model._attnres_query_idx(residual_idx)
        static_query = self.model.transformer.lrid_queries[query_idx]
        if self.model.config.lrid_input_dependent_query:
            dynamic_query = query_override if query_override is not None else sources[-1][2]
            if dynamic_query is None:
                raise RuntimeError("Input-dependent LR AttnRes query is missing")
            dynamic_query = dynamic_query.reshape(*dynamic_query.shape[:-1], num_heads, key_head_dim)
            gate = self.model.transformer.lrid_query_gates[query_idx].view(1, 1, num_heads, 1)
            query = static_query.unsqueeze(0).unsqueeze(0) + gate * dynamic_query
        else:
            query = static_query
        if self.model.config.attn_res_query_norm:
            query = norm(query.float())
        query = query.to(score_keys.dtype)

        if self.model.config.lrid_input_dependent_query:
            logits = torch.einsum("sbthr,bthr->sbth", score_keys, query)
        else:
            logits = torch.einsum("sbthr,hr->sbth", score_keys, query)
        logits = logits.float() * float(self.model.config.lrid_logit_scale)
        logits = self._add_biases(logits, source_counts, source_logit_biases)
        weights = F.softmax(logits, dim=0)
        output = torch.einsum("sbth,sbthd->bthd", weights.to(values_by_head.dtype), values_by_head)
        expected = output.reshape(output.size(0), output.size(1), self.model.config.n_embd)
        if normalize_output:
            expected = norm(expected)

        self.record_common(
            residual_idx=residual_idx,
            values=values,
            raw_keys=raw_keys,
            score_keys=score_keys,
            logits=logits,
            weights=weights,
        )
        return expected

    def record_common(
        self,
        residual_idx: int,
        values: torch.Tensor,
        raw_keys: torch.Tensor,
        score_keys: torch.Tensor,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        # weights/logits: [S, B, T, H]
        # values: [S, B, T, D]
        # keys: [S, B, T, H, R]
        n_sources = int(weights.size(0))
        if n_sources < 2:
            return
        site = self.sites.setdefault(residual_idx, SiteStats(residual_idx=residual_idx))
        per_source_count = int(weights[0].numel())

        weights_f = weights.detach().float()
        logits_f = logits.detach().float()
        source_dims = tuple(range(1, weights_f.ndim))
        weight_sum = weights_f.sum(dim=source_dims).cpu().numpy().astype(np.float64)
        weight_sq_sum = weights_f.square().sum(dim=source_dims).cpu().numpy().astype(np.float64)
        logit_sum = logits_f.sum(dim=source_dims).cpu().numpy().astype(np.float64)
        logit_sq_sum = logits_f.square().sum(dim=source_dims).cpu().numpy().astype(np.float64)
        site.update_sources(weight_sum, weight_sq_sum, logit_sum, logit_sq_sum, per_source_count)

        dist = weights_f.permute(1, 2, 3, 0).reshape(-1, n_sources)
        self.distributions_seen += int(dist.size(0))
        entropy = -(dist.clamp_min(1e-30) * dist.clamp_min(1e-30).log()).sum(dim=-1)
        site.entropy.update_tensor(entropy)
        if n_sources > 1:
            site.normalized_entropy.update_tensor(entropy / math.log(float(n_sources)))
        site.effective_sources.update_tensor(entropy.exp())
        sorted_dist = torch.sort(dist, dim=-1, descending=True).values
        site.top1_mass.update_tensor(sorted_dist[:, 0])
        site.top3_mass.update_tensor(sorted_dist[:, : min(3, n_sources)].sum(dim=-1))
        site.top5_mass.update_tensor(sorted_dist[:, : min(5, n_sources)].sum(dim=-1))
        site.simpson_concentration.update_tensor(dist.square().sum(dim=-1))
        ascending = torch.sort(dist, dim=-1, descending=False).values
        idx = torch.arange(1, n_sources + 1, device=dist.device, dtype=dist.dtype)
        gini = ((2.0 * idx - n_sources - 1.0) * ascending).sum(dim=-1) / float(n_sources)
        site.gini.update_tensor(gini)

        categories = self.source_categories(residual_idx, n_sources)
        for category in sorted(set(categories)):
            mask = torch.tensor(
                [cat == category for cat in categories],
                device=weights_f.device,
                dtype=torch.bool,
            )
            mass = weights_f[mask].sum(dim=0)
            site.category_mass.setdefault(category, RunningScalar()).update_tensor(mass)

        raw_key_rms = raw_keys.detach().float().square().mean(dim=-1).sqrt()
        score_key_abs = score_keys.detach().float().abs()
        self.key_raw_rms.update_tensor(raw_key_rms)
        self.score_key_abs.update_tensor(score_key_abs)
        self.score_key_dim.update_from_tensor(score_keys.detach().float())

        num_heads = int(score_keys.size(3))
        if num_heads > 1:
            value_by_head = values.reshape(*values.shape[:-1], num_heads, values.size(-1) // num_heads)
        else:
            value_by_head = values.unsqueeze(3)
        value_rms = value_by_head.detach().float().square().mean(dim=-1).sqrt()
        self.value_rms.update_tensor(value_rms)
        self.key_value_norm_corr.update(raw_key_rms, value_rms)

        rank = int(score_keys.size(-1))
        if 0 < rank <= values.size(-1):
            tail = values.detach().float()[..., -rank:]
            tail_energy = tail.square().sum(dim=-1)
            full_energy = values.detach().float().square().sum(dim=-1).clamp_min(1e-30)
            self.tail_energy_fraction.update_tensor(tail_energy / full_energy)

        if num_heads == 1:
            contrib = weights_f.squeeze(-1).unsqueeze(-1) * values.detach().float()
        else:
            contrib_head = weights_f.unsqueeze(-1) * value_by_head.detach().float()
            contrib = contrib_head.reshape_as(values.detach().float())
        site.contribution_rms.update_tensor(contrib.square().mean(dim=-1).sqrt())

        self._maybe_sample_pairwise(values.detach().float(), score_keys.detach().float())

    def _maybe_sample_pairwise(self, values: torch.Tensor, score_keys: torch.Tensor) -> None:
        current = sum(int(chunk.size(0)) for chunk in self.sample_values)
        needed = self.max_pair_samples - current
        if needed <= 0:
            return
        key_flat = score_keys.reshape(score_keys.size(0), score_keys.size(1), score_keys.size(2), -1)
        value_flat = values.reshape(-1, values.size(-1))
        key_flat = key_flat.reshape(-1, key_flat.size(-1))
        if value_flat.size(0) != key_flat.size(0):
            n = min(value_flat.size(0), key_flat.size(0))
            value_flat = value_flat[:n]
            key_flat = key_flat[:n]
        take = min(needed, int(value_flat.size(0)))
        if take <= 0:
            return
        stride = max(1, int(value_flat.size(0)) // take)
        indices = torch.arange(0, value_flat.size(0), stride, device=value_flat.device)[:take]
        self.sample_values.append(value_flat.index_select(0, indices).cpu())
        self.sample_keys.append(key_flat.index_select(0, indices).cpu())


# =============================================================================
# Summaries
# =============================================================================

@dataclass
class AnalysisResult:
    repo_id: str
    checkpoint_path: Optional[str]
    group: str
    label: str
    attnres_type: Optional[str]
    attnres_num_blocks: Optional[int]
    lrid_rank: Optional[int]
    use_lrid: bool
    lrid_key_from_output_tail: bool
    use_attnres: bool
    paper_val_loss: Optional[float]
    paper_added_flops: Optional[float]
    num_batches: int = 0
    tokens_seen: int = 0
    query_matrix: Optional[np.ndarray] = None
    query_site_metrics: list[dict[str, Any]] = field(default_factory=list)
    query_summary: dict[str, Any] = field(default_factory=dict)
    attention_site_metrics: list[dict[str, Any]] = field(default_factory=list)
    source_slot_metrics: list[dict[str, Any]] = field(default_factory=list)
    category_metrics: list[dict[str, Any]] = field(default_factory=list)
    key_summary: dict[str, Any] = field(default_factory=dict)
    output_check: dict[str, Any] = field(default_factory=dict)

    def summary_row(self) -> dict[str, Any]:
        mean_metrics = self.mean_attention_metrics()
        row = {
            "repo_id": self.repo_id,
            "label": self.label,
            "group": self.group,
            "group_name": STYLE.get(self.group, {}).get("name", self.group),
            "attnres_type": self.attnres_type,
            "attnres_num_blocks": self.attnres_num_blocks,
            "lrid_rank": self.lrid_rank,
            "use_lrid": self.use_lrid,
            "lrid_key_from_output_tail": self.lrid_key_from_output_tail,
            "use_attnres": self.use_attnres,
            "paper_val_loss": self.paper_val_loss,
            "paper_added_flops": self.paper_added_flops,
            "num_batches": self.num_batches,
            "tokens_seen": self.tokens_seen,
        }
        row.update({f"query_{k}": v for k, v in self.query_summary.items()})
        row.update({f"attn_{k}": v for k, v in mean_metrics.items()})
        row.update({f"key_{k}": v for k, v in self.key_summary.items()})
        row.update({f"check_{k}": v for k, v in self.output_check.items()})
        return row

    def mean_attention_metrics(self) -> dict[str, Any]:
        if not self.attention_site_metrics:
            return {}
        keys = [
            "entropy_mean",
            "normalized_entropy_mean",
            "effective_sources_mean",
            "top1_mass_mean",
            "top3_mass_mean",
            "top5_mass_mean",
            "gini_mean",
            "simpson_concentration_mean",
            "js_generalized",
            "mean_weight_cv",
            "contribution_rms_mean",
        ]
        out = {}
        for key in keys:
            values = [safe_float(row.get(key)) for row in self.attention_site_metrics]
            values = [value for value in values if value is not None]
            out[key] = float(np.mean(values)) if values else None
        return out


def collect_query_metrics(model: OBPM) -> tuple[Optional[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    if not model.use_attnres:
        return None, [], {}

    if model.use_lrid:
        query = torch.stack([param.detach().float().cpu() for param in model.transformer.lrid_queries], dim=0)
        if model.config.attn_res_query_norm:
            query = norm(query)
        matrix = query.reshape(query.size(0), -1).numpy()
    else:
        rows = []
        for residual in model.transformer.attn_residuals:
            rows.append(residual._query(torch.float32).detach().cpu())
        matrix = torch.stack(rows, dim=0).numpy()

    metrics = []
    rel_thresholds = [0.001, 0.01, 0.05, 0.10]
    for idx, row in enumerate(matrix):
        abs_row = np.abs(row)
        max_abs = float(abs_row.max()) if abs_row.size else 0.0
        row_metrics = {
            "residual_idx": idx + 1,
            "query_dim": int(row.size),
            "query_l2": float(np.linalg.norm(row)),
            "query_l1": float(np.abs(row).sum()),
            "query_abs_mean": float(abs_row.mean()) if abs_row.size else 0.0,
            "query_abs_median": float(np.median(abs_row)) if abs_row.size else 0.0,
            "query_max_abs": max_abs,
            "exact_zero_fraction": float(np.mean(abs_row == 0.0)) if abs_row.size else None,
            "near_zero_1e-8_fraction": float(np.mean(abs_row <= 1e-8)) if abs_row.size else None,
            "participation_ratio": participation_ratio(row),
            "participation_fraction": participation_ratio(row) / row.size if row.size else None,
            "hoyer_sparsity": hoyer_sparsity(row),
            "dims_90_energy": dims_for_energy(row, 0.90),
            "dims_95_energy": dims_for_energy(row, 0.95),
            "dims_99_energy": dims_for_energy(row, 0.99),
        }
        denom = max(max_abs, 1e-30)
        for threshold in rel_thresholds:
            row_metrics[f"used_gt_{threshold:g}x_max_fraction"] = float(np.mean(abs_row > threshold * denom))
        metrics.append(row_metrics)

    summary = {}
    if metrics:
        for key in [
            "query_l2",
            "query_abs_mean",
            "exact_zero_fraction",
            "near_zero_1e-8_fraction",
            "participation_ratio",
            "participation_fraction",
            "hoyer_sparsity",
            "dims_90_energy",
            "dims_95_energy",
            "dims_99_energy",
            "used_gt_0.001x_max_fraction",
            "used_gt_0.01x_max_fraction",
            "used_gt_0.05x_max_fraction",
            "used_gt_0.1x_max_fraction",
        ]:
            values = [safe_float(row.get(key)) for row in metrics]
            values = [value for value in values if value is not None]
            summary[f"{key}_mean"] = float(np.mean(values)) if values else None
            summary[f"{key}_median"] = float(np.median(values)) if values else None
        summary["query_dim"] = int(matrix.shape[1])
        summary["query_sites"] = int(matrix.shape[0])
    return matrix, metrics, summary


def summarize_recorder(recorder: RoutingRecorder) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    site_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    for residual_idx in sorted(recorder.sites):
        site = recorder.sites[residual_idx]
        valid = site.source_count > 0
        mean_weight = np.zeros_like(site.source_weight_sum, dtype=np.float64)
        var_weight = np.zeros_like(site.source_weight_sum, dtype=np.float64)
        mean_logit = np.zeros_like(site.source_logit_sum, dtype=np.float64)
        var_logit = np.zeros_like(site.source_logit_sum, dtype=np.float64)
        mean_weight[valid] = site.source_weight_sum[valid] / site.source_count[valid]
        mean_logit[valid] = site.source_logit_sum[valid] / site.source_count[valid]
        var_weight[valid] = np.maximum(
            site.source_weight_sq_sum[valid] / site.source_count[valid] - mean_weight[valid] ** 2,
            0.0,
        )
        var_logit[valid] = np.maximum(
            site.source_logit_sq_sum[valid] / site.source_count[valid] - mean_logit[valid] ** 2,
            0.0,
        )
        present_weight = mean_weight[valid]
        present_var = var_weight[valid]
        mean_weight_cv = None
        if present_weight.size:
            mean_weight_cv = float(np.mean(np.sqrt(present_var) / np.maximum(present_weight, 1e-12)))
        mean_dist = present_weight / max(float(present_weight.sum()), 1e-30)
        entropy_mean = site.entropy.mean or 0.0
        js_generalized = max(entropy_np(mean_dist) - entropy_mean, 0.0)

        site_rows.append(
            {
                "residual_idx": residual_idx,
                "num_sources": int(valid.sum()),
                "distributions": int(site.entropy.count),
                "entropy_mean": site.entropy.mean,
                "entropy_std": site.entropy.std,
                "normalized_entropy_mean": site.normalized_entropy.mean,
                "effective_sources_mean": site.effective_sources.mean,
                "top1_mass_mean": site.top1_mass.mean,
                "top3_mass_mean": site.top3_mass.mean,
                "top5_mass_mean": site.top5_mass.mean,
                "gini_mean": site.gini.mean,
                "simpson_concentration_mean": site.simpson_concentration.mean,
                "js_generalized": js_generalized,
                "mean_weight_cv": mean_weight_cv,
                "contribution_rms_mean": site.contribution_rms.mean,
            }
        )

        for source_idx in np.flatnonzero(valid):
            source_rows.append(
                {
                    "residual_idx": residual_idx,
                    "source_slot": int(source_idx),
                    "count": int(site.source_count[source_idx]),
                    "weight_mean": float(mean_weight[source_idx]),
                    "weight_std": float(math.sqrt(var_weight[source_idx])),
                    "weight_cv": float(math.sqrt(var_weight[source_idx]) / max(mean_weight[source_idx], 1e-12)),
                    "logit_mean": float(mean_logit[source_idx]),
                    "logit_std": float(math.sqrt(var_logit[source_idx])),
                }
            )

        for category, stat in sorted(site.category_mass.items()):
            category_rows.append(
                {
                    "residual_idx": residual_idx,
                    "category": category,
                    "count": stat.count,
                    "mass_mean": stat.mean,
                    "mass_std": stat.std,
                }
            )

    key_var = recorder.score_key_dim.variance()
    key_mean = recorder.score_key_dim.mean()
    key_dim_std_mean = float(np.sqrt(key_var).mean()) if key_var is not None else None
    key_dim_abs_mean = float(np.abs(key_mean).mean()) if key_mean is not None else None

    pairwise_similarity_corr = None
    if recorder.sample_values and recorder.sample_keys:
        values = torch.cat(recorder.sample_values, dim=0).float()
        keys = torch.cat(recorder.sample_keys, dim=0).float()
        n = min(values.size(0), keys.size(0), recorder.max_pair_samples)
        values = values[:n]
        keys = keys[:n]
        if n >= 4:
            values = F.normalize(values, dim=-1)
            keys = F.normalize(keys, dim=-1)
            value_sim = (values @ values.t()).cpu().numpy()
            key_sim = (keys @ keys.t()).cpu().numpy()
            triu = np.triu_indices(n, k=1)
            pairwise_similarity_corr = pearson_from_arrays(value_sim[triu], key_sim[triu])

    key_summary = {
        "raw_key_rms_mean": recorder.key_raw_rms.mean,
        "score_key_abs_mean": recorder.score_key_abs.mean,
        "score_key_dim_std_mean": key_dim_std_mean,
        "score_key_dim_abs_mean": key_dim_abs_mean,
        "value_rms_mean": recorder.value_rms.mean,
        "tail_energy_fraction_mean": recorder.tail_energy_fraction.mean,
        "key_value_norm_corr": recorder.key_value_norm_corr.correlation(),
        "key_value_pairwise_similarity_corr": pairwise_similarity_corr,
        "pairwise_samples": sum(int(chunk.size(0)) for chunk in recorder.sample_values),
        "distributions_seen": recorder.distributions_seen,
    }
    output_check = {
        "output_abs_error_max_mean": recorder.output_abs_error.mean,
        "output_rel_error_max_mean": recorder.output_rel_error.mean,
    }
    return site_rows, source_rows, category_rows, key_summary, output_check


def make_empty_result(repo_id: str, checkpoint_path: Optional[str], config: ModelConfig) -> AnalysisResult:
    group = group_from_config(config)
    return AnalysisResult(
        repo_id=repo_id,
        checkpoint_path=checkpoint_path,
        group=group,
        label=label_from_config(repo_id, config),
        attnres_type=config.attnres_type if config.use_attnres else None,
        attnres_num_blocks=config.attnres_num_blocks if config.use_attnres and config.attnres_type == "block" else None,
        lrid_rank=config.lrid_rank if config.use_lrid else None,
        use_lrid=bool(config.use_lrid),
        lrid_key_from_output_tail=bool(config.lrid_key_from_output_tail),
        use_attnres=bool(config.use_attnres),
        paper_val_loss=PAPER_VAL_LOSS.get(repo_id),
        paper_added_flops=PAPER_ADDED_FLOPS.get(repo_id),
    )


@torch.no_grad()
def analyze_loaded_model(
    loaded: LoadedModel,
    val_loader,
    use_doc_masking: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> AnalysisResult:
    result = make_empty_result(loaded.repo_id, loaded.checkpoint_path, loaded.model_config)
    query_matrix, query_site_metrics, query_summary = collect_query_metrics(loaded.model)
    result.query_matrix = query_matrix
    result.query_site_metrics = query_site_metrics
    result.query_summary = query_summary

    if not loaded.model_config.use_attnres:
        return result

    original_runtime = prepare_model_for_analysis(loaded.model)
    recorder = RoutingRecorder(
        loaded.model,
        max_pair_samples=args.max_pair_samples,
        validate_outputs=args.validate_outputs,
    )
    recorder.install()
    try:
        iterator = tqdm(val_loader, desc=f"Collecting {result.label}", leave=False, total=len(val_loader))
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
            result.num_batches += 1
            result.tokens_seen += int(x.numel())
            if args.max_batches is not None and result.num_batches >= args.max_batches:
                break
    finally:
        recorder.restore()
        restore_model_after_analysis(loaded.model, original_runtime)

    (
        result.attention_site_metrics,
        result.source_slot_metrics,
        result.category_metrics,
        result.key_summary,
        result.output_check,
    ) = summarize_recorder(recorder)
    return result


# =============================================================================
# Persistence
# =============================================================================

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved {path}")


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_outputs(results: list[AnalysisResult], skipped: list[dict[str, Any]], output_dir: Path) -> None:
    summary_rows = [result.summary_row() for result in results]
    query_rows = []
    site_rows = []
    source_rows = []
    category_rows = []
    for result in results:
        prefix = {
            "repo_id": result.repo_id,
            "label": result.label,
            "group": result.group,
        }
        query_rows.extend([{**prefix, **row} for row in result.query_site_metrics])
        site_rows.extend([{**prefix, **row} for row in result.attention_site_metrics])
        source_rows.extend([{**prefix, **row} for row in result.source_slot_metrics])
        category_rows.extend([{**prefix, **row} for row in result.category_metrics])

    write_csv(output_dir / "routing_metrics.csv", summary_rows)
    write_csv(output_dir / "query_site_metrics.csv", query_rows)
    write_csv(output_dir / "attention_site_metrics.csv", site_rows)
    write_csv(output_dir / "source_slot_metrics.csv", source_rows)
    write_csv(output_dir / "category_mass_metrics.csv", category_rows)

    compact = []
    for result in results:
        row = result.summary_row()
        row["query_matrix_shape"] = None if result.query_matrix is None else list(result.query_matrix.shape)
        compact.append(row)
    (output_dir / "routing_metrics.json").write_text(
        json.dumps(compact, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "skipped_models.json").write_text(
        json.dumps(skipped, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {output_dir / 'routing_metrics.json'}")
    print(f"Saved {output_dir / 'skipped_models.json'}")


def result_payload(result: AnalysisResult) -> dict[str, Any]:
    return {
        "repo_id": result.repo_id,
        "checkpoint_path": result.checkpoint_path,
        "group": result.group,
        "label": result.label,
        "attnres_type": result.attnres_type,
        "attnres_num_blocks": result.attnres_num_blocks,
        "lrid_rank": result.lrid_rank,
        "use_lrid": result.use_lrid,
        "lrid_key_from_output_tail": result.lrid_key_from_output_tail,
        "use_attnres": result.use_attnres,
        "paper_val_loss": result.paper_val_loss,
        "paper_added_flops": result.paper_added_flops,
        "num_batches": result.num_batches,
        "tokens_seen": result.tokens_seen,
        "query_matrix": result.query_matrix,
        "query_site_metrics": result.query_site_metrics,
        "query_summary": result.query_summary,
        "attention_site_metrics": result.attention_site_metrics,
        "source_slot_metrics": result.source_slot_metrics,
        "category_metrics": result.category_metrics,
        "key_summary": result.key_summary,
        "output_check": result.output_check,
    }


def result_from_payload(payload: dict[str, Any]) -> AnalysisResult:
    query_matrix = payload.get("query_matrix")
    if query_matrix is not None:
        query_matrix = np.asarray(query_matrix, dtype=np.float32)
    return AnalysisResult(
        repo_id=payload["repo_id"],
        checkpoint_path=payload.get("checkpoint_path"),
        group=payload["group"],
        label=payload["label"],
        attnres_type=payload.get("attnres_type"),
        attnres_num_blocks=payload.get("attnres_num_blocks"),
        lrid_rank=payload.get("lrid_rank"),
        use_lrid=bool(payload.get("use_lrid")),
        lrid_key_from_output_tail=bool(payload.get("lrid_key_from_output_tail")),
        use_attnres=bool(payload.get("use_attnres")),
        paper_val_loss=payload.get("paper_val_loss"),
        paper_added_flops=payload.get("paper_added_flops"),
        num_batches=int(payload.get("num_batches", 0)),
        tokens_seen=int(payload.get("tokens_seen", 0)),
        query_matrix=query_matrix,
        query_site_metrics=list(payload.get("query_site_metrics", [])),
        query_summary=dict(payload.get("query_summary", {})),
        attention_site_metrics=list(payload.get("attention_site_metrics", [])),
        source_slot_metrics=list(payload.get("source_slot_metrics", [])),
        category_metrics=list(payload.get("category_metrics", [])),
        key_summary=dict(payload.get("key_summary", {})),
        output_check=dict(payload.get("output_check", {})),
    )


def model_result_path(output_dir: Path, repo_id: str) -> Path:
    return output_dir / "model_results" / f"{sanitize_filename(repo_id)}.json"


def save_model_result(result: AnalysisResult, output_dir: Path) -> None:
    path = model_result_path(output_dir, result.repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_payload(result), indent=2, default=json_default) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def load_model_result(output_dir: Path, repo_id: str) -> Optional[AnalysisResult]:
    path = model_result_path(output_dir, repo_id)
    if not path.exists():
        return None
    return result_from_payload(json.loads(path.read_text(encoding="utf-8")))


# =============================================================================
# Plotting
# =============================================================================

def result_display_name(result: AnalysisResult) -> str:
    return f"{result.label}"


def plot_query_heatmaps(results: list[AnalysisResult], output_dir: Path) -> None:
    candidates = choose_representative_models(results)
    candidates = [result for result in candidates if result.query_matrix is not None]
    if not candidates:
        return
    matrices = [np.abs(result.query_matrix) for result in candidates if result.query_matrix is not None]
    all_values = np.concatenate([matrix.ravel() for matrix in matrices])
    vmax = float(np.percentile(all_values, 99.5)) if all_values.size else 1.0
    vmax = max(vmax, 1e-12)
    set_academic_rcparams()
    fig, axes = plt.subplots(
        1,
        len(candidates),
        figsize=(5.3 * len(candidates) + 0.55, 4.85),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, result in zip(axes[0], candidates):
        matrix = np.abs(result.query_matrix)
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(f"{STYLE[result.group]['name']}\n{result.label}", fontweight="bold")
        ax.set_xlabel("Query Dimension")
        ax.set_ylabel("Residual Read Site")
        for boundary in range(1, matrix.shape[0], 2):
            ax.axhline(boundary - 0.5, color="white", linewidth=0.25, alpha=0.22)
        ax.tick_params(axis="both", colors="#555555", length=3.0, width=0.8)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.025, label="|Static Query|")
    save_figure(fig, output_dir, "query_heatmaps_core", tight=False)


def mean_weight_matrix(result: AnalysisResult) -> Optional[np.ndarray]:
    if not result.source_slot_metrics:
        return None
    max_site = max(int(row["residual_idx"]) for row in result.source_slot_metrics)
    max_source = max(int(row["source_slot"]) for row in result.source_slot_metrics) + 1
    matrix = np.full((max_site + 1, max_source), np.nan, dtype=np.float64)
    for row in result.source_slot_metrics:
        matrix[int(row["residual_idx"]), int(row["source_slot"])] = float(row["weight_mean"])
    return matrix


def plot_attention_heatmaps(results: list[AnalysisResult], output_dir: Path) -> None:
    candidates = [result for result in choose_representative_models(results) if result.source_slot_metrics]
    if not candidates:
        return
    set_academic_rcparams()
    fig, axes = plt.subplots(
        1,
        len(candidates),
        figsize=(5.3 * len(candidates) + 0.55, 4.85),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, result in zip(axes[0], candidates):
        matrix = mean_weight_matrix(result)
        if matrix is None:
            continue
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=np.nanmax(matrix))
        ax.set_title(f"{STYLE[result.group]['name']}\n{result.label}", fontweight="bold")
        ax.set_xlabel("Source Slot")
        ax.set_ylabel("Residual Read Site")
        ax.tick_params(axis="both", colors="#555555", length=3.0, width=0.8)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.025, label="Mean Attention Weight")
    save_figure(fig, output_dir, "attention_mean_heatmaps_core", tight=False)


def choose_representative_models(results: list[AnalysisResult]) -> list[AnalysisResult]:
    out: list[AnalysisResult] = []
    for group in ["standard_attnres", "lr_attnres", "sliced_lr_attnres"]:
        group_results = [result for result in results if result.group == group]
        if not group_results:
            continue
        preferred = None
        if group == "standard_attnres":
            preferred = next((r for r in group_results if r.attnres_num_blocks == 8), None)
        elif group == "lr_attnres":
            preferred = next((r for r in group_results if r.attnres_num_blocks == 8 and r.lrid_rank == 32), None)
            preferred = preferred or next((r for r in group_results if r.attnres_type == "full" and r.lrid_rank == 32), None)
        elif group == "sliced_lr_attnres":
            preferred = next((r for r in group_results if r.attnres_num_blocks == 8 and r.lrid_rank == 64), None)
            preferred = preferred or next((r for r in group_results if r.attnres_type == "full" and r.lrid_rank == 64), None)
        out.append(preferred or sorted(group_results, key=model_sort_key)[0])
    return out


def choose_effective_sources_comparison_models(results: list[AnalysisResult]) -> list[AnalysisResult]:
    selected = [
        result
        for result in results
        if (
            result.group == "standard_attnres"
            and (
                result.attnres_type == "full"
                or (result.attnres_type == "block" and result.attnres_num_blocks == 8)
            )
        )
        or (
            result.group == "sliced_lr_attnres"
            and (
                (result.attnres_type == "full" and result.lrid_rank == 64)
                or (result.attnres_type == "block" and result.attnres_num_blocks == 8 and result.lrid_rank == 64)
            )
        )
        or (
            result.group == "lr_attnres"
            and (
                (result.attnres_type == "full" and result.lrid_rank == 32)
                or (result.attnres_type == "block" and result.attnres_num_blocks == 8 and result.lrid_rank == 32)
            )
        )
    ]
    return sorted(selected, key=model_sort_key)


def plot_site_metric_lines(
    results: list[AnalysisResult],
    metric: str,
    ylabel: str,
    basename: str,
    output_dir: Path,
    representative_only: bool = True,
    solid_lines: bool = False,
    selected_results: Optional[list[AnalysisResult]] = None,
) -> None:
    if selected_results is not None:
        selected_ids = {result.repo_id for result in selected_results}
        line_results = [
            result
            for result in results
            if result.attention_site_metrics and result.repo_id in selected_ids
        ]
    elif representative_only:
        representative_ids = {result.repo_id for result in choose_representative_models(results)}
        line_results = [
            result
            for result in results
            if result.attention_site_metrics and result.repo_id in representative_ids
        ]
    else:
        line_results = [result for result in results if result.attention_site_metrics]
    if not line_results:
        return
    line_results = sorted(line_results, key=model_sort_key)
    set_academic_rcparams()
    compact_selection = selected_results is not None
    fig, ax = plt.subplots(figsize=FIGSIZE if representative_only or compact_selection else (8.2, 5.25))
    fig.patch.set_facecolor("white")
    color_by_repo: dict[str, Any] = {}
    if compact_selection and line_results and all(
        result.group in {"standard_attnres", "sliced_lr_attnres", "lr_attnres"} for result in line_results
    ):
        gradient_by_group = {
            "standard_attnres": mpl.colors.LinearSegmentedColormap.from_list(
                "standard_attnres_blue_gradient",
                ["#2563EB", "#0B2E6B"],
            ),
            "sliced_lr_attnres": mpl.colors.LinearSegmentedColormap.from_list(
                "sliced_full_yellow_gradient",
                ["#F59E0B", "#92400E"],
            ),
            "lr_attnres": mpl.colors.LinearSegmentedColormap.from_list(
                "projected_full_red_gradient",
                ["#EF4444", "#7F1D1D"],
            ),
        }
        for group, cmap in gradient_by_group.items():
            group_results = [result for result in line_results if result.group == group]
            denom = max(len(group_results) - 1, 1)
            color_by_repo.update(
                {result.repo_id: cmap(idx / denom) for idx, result in enumerate(group_results)}
            )
    elif not representative_only:
        cmap = plt.get_cmap("tab20")
        color_by_repo = {result.repo_id: cmap(idx % cmap.N) for idx, result in enumerate(line_results)}
    for result in line_results:
        rows = sorted(result.attention_site_metrics, key=lambda row: int(row["residual_idx"]))
        xs = np.array([row["residual_idx"] for row in rows], dtype=float)
        ys = np.array([safe_float(row.get(metric)) for row in rows], dtype=object)
        valid = np.array([value is not None for value in ys], dtype=bool)
        if not valid.any():
            continue
        y_float = np.array([float(value) for value in ys[valid]], dtype=float)
        style = STYLE[result.group]
        color = color_by_repo.get(result.repo_id, style["color"])
        linestyle = "-" if solid_lines or result.attnres_type == "full" else "--"
        label = (
            result_display_name(result)
            if representative_only
            else f"{style['name']}: {result_display_name(result)}"
        )
        ax.plot(
            xs[valid],
            y_float,
            color=color,
            linewidth=1.8 if representative_only or compact_selection else 1.45,
            linestyle=linestyle,
            alpha=0.92 if representative_only or compact_selection else 0.82,
            label=label,
        )
    style_axis(ax, "Residual Read Site", ylabel)
    if representative_only:
        finish_legend(ax, loc="best")
    elif compact_selection:
        finish_legend(ax, loc="best")
    else:
        finish_outside_legend(ax)
    save_figure(fig, output_dir, basename)


def plot_query_summary(results: list[AnalysisResult], output_dir: Path) -> None:
    query_results = [result for result in results if result.query_summary]
    if not query_results:
        return
    query_results = sorted(query_results, key=model_sort_key)
    set_academic_rcparams()
    fig, ax = plt.subplots(figsize=(max(6.5, 0.42 * len(query_results)), 4.85))
    fig.patch.set_facecolor("white")
    xs = np.arange(len(query_results))
    for idx, result in enumerate(query_results):
        style = STYLE[result.group]
        value = result.query_summary.get("participation_fraction_mean")
        ax.scatter(
            [idx],
            [value],
            s=style["size"],
            marker=style["marker"],
            c=style["color"],
            edgecolors="white",
            linewidths=1.1,
            alpha=0.96,
            zorder=3,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([result.label for result in query_results], rotation=35, ha="right")
    style_axis(ax, "Model", "Mean Query Participation Fraction")
    finish_legend(ax, handles=legend_handles(), loc="best")
    save_figure(fig, output_dir, "query_utilization_summary")


def plot_query_energy_by_site(results: list[AnalysisResult], output_dir: Path) -> None:
    representative_ids = {result.repo_id for result in choose_representative_models(results)}
    query_results = [
        result
        for result in results
        if result.query_site_metrics and result.repo_id in representative_ids
    ]
    if not query_results:
        return
    set_academic_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    for result in sorted(query_results, key=model_sort_key):
        rows = sorted(result.query_site_metrics, key=lambda row: int(row["residual_idx"]))
        xs = np.array([row["residual_idx"] for row in rows], dtype=float)
        ys = np.array([row["dims_95_energy"] for row in rows], dtype=float)
        style = STYLE[result.group]
        linestyle = "-" if result.attnres_type == "full" else "--"
        ax.plot(xs, ys, color=style["color"], linewidth=1.75, linestyle=linestyle, alpha=0.88, label=result.label)
    style_axis(ax, "Residual Read Site", "Dims for 95% Query Energy")
    finish_legend(ax, loc="best")
    save_figure(fig, output_dir, "query_energy_dims_by_site")


def annotate_scatter_labels(ax, rows: list[tuple[AnalysisResult, float, float]], use_adjust_text: bool = True) -> None:
    fig = ax.figure
    fig.canvas.draw()
    point_to_px = fig.dpi / 72.0
    max_dx_pt = 112.0
    max_dy_pt = 72.0
    base_offsets = [
        (8, 5),
        (10, 24),
        (10, -17),
        (-44, 5),
        (-54, 24),
        (-54, -17),
        (0, 38),
        (0, -33),
        (46, 38),
        (-76, 38),
        (46, -33),
        (-76, -33),
        (72, 10),
        (-92, 10),
    ]
    family_offsets = {
        "standard_attnres": [(8, 5), (-54, 24), (10, -17)],
        "lr_attnres": [(10, -17), (-54, -17), (46, -33)],
        "sliced_lr_attnres": [(10, 24), (-54, 24), (46, 38)],
        "baseline_transformer": [(8, 5), (10, 24), (10, -17)],
    }

    def expanded_bbox(annotation):
        renderer = fig.canvas.get_renderer()
        return annotation.get_window_extent(renderer=renderer).expanded(1.08, 1.16)

    def compact_label(text: str) -> str:
        if text.startswith("Full r="):
            return text.replace("Full r=", "F r")
        return text.replace("n=", "n").replace(" r=", " r")

    if use_adjust_text and adjust_text is not None:
        texts = []
        target_x = []
        target_y = []
        for result, x, y in sorted(rows, key=lambda item: (item[1], item[2], item[0].label)):
            style = STYLE[result.group]
            target_x.append(x)
            target_y.append(y)
            texts.append(
                ax.text(
                    x,
                    y,
                    compact_label(result.label),
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    fontweight="bold",
                    color=style["color"],
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="white",
                        edgecolor=style["color"],
                        linewidth=0.72,
                        alpha=0.94,
                    ),
                    zorder=4,
                )
            )
        adjust_text(
            texts,
            x=target_x,
            y=target_y,
            target_x=target_x,
            target_y=target_y,
            ax=ax,
            ensure_inside_axes=True,
            prevent_crossings=True,
            expand=(1.12, 1.28),
            force_text=(0.24, 0.38),
            force_static=(0.16, 0.26),
            force_pull=(0.008, 0.008),
            force_explode=(0.18, 0.32),
            max_move=(18, 18),
            min_arrow_len=5,
            iter_lim=350,
            arrowprops=dict(
                arrowstyle="-",
                color="#8A8A8A",
                linewidth=0.45,
                alpha=0.58,
            ),
        )
        return

    placed = []
    annotations = []
    for result, x, y in sorted(rows, key=lambda item: (item[1], item[2], item[0].label)):
        style = STYLE[result.group]
        label = compact_label(result.label)
        offsets = family_offsets.get(result.group, []) + base_offsets
        best = offsets[0]
        best_overlap = float("inf")
        best_escape = float("inf")
        for dx_pt, dy_pt in offsets:
            probe = ax.annotate(
                label,
                xy=(x, y),
                xytext=(dx_pt, dy_pt),
                textcoords="offset points",
                ha="left" if dx_pt >= 0 else "right",
                va="center",
                fontsize=6.6,
                fontweight="bold",
                color=style["color"],
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor=style["color"],
                    linewidth=0.72,
                    alpha=0.94,
                ),
                zorder=4,
            )
            fig.canvas.draw()
            box = expanded_bbox(probe)
            overlap = sum(box.overlaps(existing) for existing in placed)
            ax_box = ax.get_window_extent(renderer=fig.canvas.get_renderer())
            escape = (
                max(0.0, ax_box.x0 - box.x0)
                + max(0.0, box.x1 - ax_box.x1)
                + max(0.0, ax_box.y0 - box.y0)
                + max(0.0, box.y1 - ax_box.y1)
            )
            probe.remove()
            if overlap < best_overlap or (overlap == best_overlap and escape < best_escape):
                best = (dx_pt, dy_pt)
                best_overlap = float(overlap)
                best_escape = float(escape)
            if overlap == 0 and escape == 0:
                break
        dx_pt, dy_pt = best
        annotation = ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            ha="left" if dx_pt >= 0 else "right",
            va="center",
            fontsize=6.6,
            fontweight="bold",
            color=style["color"],
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor=style["color"],
                linewidth=0.72,
                alpha=0.94,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=style["color"],
                linewidth=0.45,
                alpha=0.58,
                shrinkA=0,
                shrinkB=3,
            ),
            zorder=4,
        )
        fig.canvas.draw()
        placed.append(expanded_bbox(annotation))
        annotations.append(annotation)

    # A second pass uses actual rendered text extents to separate labels in dense
    # regions. This keeps the scatter plots readable without hand-tuned offsets
    # for each metric axis.
    for _ in range(80):
        fig.canvas.draw()
        boxes = [expanded_bbox(annotation) for annotation in annotations]
        positions = [list(annotation.get_position()) for annotation in annotations]
        moved = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box_i = boxes[i]
                box_j = boxes[j]
                if not box_i.overlaps(box_j):
                    continue
                x_overlap = min(box_i.x1, box_j.x1) - max(box_i.x0, box_j.x0)
                y_overlap = min(box_i.y1, box_j.y1) - max(box_i.y0, box_j.y0)
                if x_overlap <= 0 or y_overlap <= 0:
                    continue
                ci_x = 0.5 * (box_i.x0 + box_i.x1)
                cj_x = 0.5 * (box_j.x0 + box_j.x1)
                ci_y = 0.5 * (box_i.y0 + box_i.y1)
                cj_y = 0.5 * (box_j.y0 + box_j.y1)
                y_sign = 1.0 if ci_y >= cj_y else -1.0
                x_sign = 1.0 if ci_x >= cj_x else -1.0
                y_push = max(2.5, min(8.0, 0.55 * y_overlap / point_to_px))
                x_push = max(1.2, min(5.0, 0.22 * x_overlap / point_to_px))
                positions[i][1] += y_sign * y_push
                positions[j][1] -= y_sign * y_push
                positions[i][0] += x_sign * x_push
                positions[j][0] -= x_sign * x_push
                moved = True
        if not moved:
            break
        for annotation, position in zip(annotations, positions):
            position[0] = max(-max_dx_pt, min(max_dx_pt, position[0]))
            position[1] = max(-max_dy_pt, min(max_dy_pt, position[1]))
            annotation.set_position(tuple(position))


def plot_metric_vs_validation(results: list[AnalysisResult], output_dir: Path) -> None:
    metric_keys = [
        ("normalized_entropy_mean", "Mean Normalized Attention Entropy", "metric_vs_loss_entropy"),
        ("effective_sources_mean", "Mean Effective Number of Sources", "metric_vs_loss_effective_sources"),
        ("js_generalized", "Mean Input Dependence (Generalized JS)", "metric_vs_loss_input_dependence"),
        ("top1_mass_mean", "Mean Top-1 Source Mass", "metric_vs_loss_top1"),
    ]
    for metric, xlabel, basename in metric_keys:
        rows = []
        for result in results:
            if result.paper_val_loss is None:
                continue
            mean_metrics = result.mean_attention_metrics()
            value = safe_float(mean_metrics.get(metric))
            if value is None:
                continue
            rows.append((result, value, result.paper_val_loss))
        if not rows:
            continue
        set_academic_rcparams()
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        fig.patch.set_facecolor("white")
        for result, x, y in rows:
            style = STYLE[result.group]
            ax.scatter(
                [x],
                [y],
                s=style["size"],
                marker=style["marker"],
                c=style["color"],
                edgecolors="white",
                linewidths=1.1,
                alpha=0.96,
                zorder=3,
            )
        style_axis(ax, xlabel, "Validation Loss")
        xs = np.array([row[1] for row in rows], dtype=float)
        ys = np.array([row[2] for row in rows], dtype=float)
        if metric == "effective_sources_mean":
            x_pad = max((float(xs.max()) - float(xs.min())) * 0.10, 0.70)
            y_pad = max((float(ys.max()) - float(ys.min())) * 0.22, 0.004)
            ax.set_xlim(max(0.0, float(xs.min()) - x_pad), float(xs.max()) + x_pad)
        else:
            x_pad = max((float(xs.max()) - float(xs.min())) * 0.24, 0.025)
            y_pad = max((float(ys.max()) - float(ys.min())) * 0.24, 0.003)
            ax.set_xlim(float(xs.min()) - x_pad, float(xs.max()) + x_pad)
        ax.set_ylim(float(ys.min()) - y_pad, float(ys.max()) + y_pad)
        annotate_scatter_labels(ax, rows, use_adjust_text=(metric != "effective_sources_mean"))
        groups = [group for group in LEGEND_ORDER if any(row[0].group == group for row in rows)]
        finish_outside_legend(ax, handles=legend_handles(groups))
        save_figure(fig, output_dir, basename)


def plot_key_coupling(results: list[AnalysisResult], output_dir: Path) -> None:
    key_results = [result for result in results if result.key_summary]
    if not key_results:
        return
    key_results = sorted(key_results, key=model_sort_key)
    metrics = [
        ("key_value_pairwise_similarity_corr", "Pairwise Key/Value Similarity Corr."),
        ("key_value_norm_corr", "Key/Value Norm Corr."),
        ("tail_energy_fraction_mean", "Tail Energy Fraction"),
    ]
    set_academic_rcparams()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.6), squeeze=False)
    fig.patch.set_facecolor("white")
    xs = np.arange(len(key_results))
    for ax, (metric, ylabel) in zip(axes[0], metrics):
        for idx, result in enumerate(key_results):
            value = safe_float(result.key_summary.get(metric))
            if value is None:
                continue
            style = STYLE[result.group]
            ax.scatter(
                [idx],
                [value],
                s=style["size"],
                marker=style["marker"],
                c=style["color"],
                edgecolors="white",
                linewidths=1.1,
                alpha=0.96,
                zorder=3,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([result.label for result in key_results], rotation=35, ha="right")
        style_axis(ax, "Model", ylabel)
    finish_legend(axes[0, -1], handles=legend_handles(), loc="best")
    save_figure(fig, output_dir, "projected_vs_sliced_key_value_coupling")


def plot_category_mass(results: list[AnalysisResult], output_dir: Path) -> None:
    category_rows = []
    representative_ids = {result.repo_id for result in choose_representative_models(results)}
    for result in results:
        if result.repo_id not in representative_ids:
            continue
        if not result.category_metrics:
            continue
        num_sites = max(1, len(result.attention_site_metrics))
        grouped: dict[str, float] = {}
        for row in result.category_metrics:
            value = safe_float(row.get("mass_mean"))
            if value is not None:
                grouped[str(row["category"])] = grouped.get(str(row["category"]), 0.0) + value
        for category, total in grouped.items():
            category_rows.append((result, category, float(total / num_sites)))
    if not category_rows:
        return
    category_order = ["embedding", "completed", "partial", "previous"]
    categories = [category for category in category_order if any(row[1] == category for row in category_rows)]
    models = sorted({result.repo_id: result for result, _, _ in category_rows}.values(), key=model_sort_key)
    set_academic_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 4.65))
    fig.patch.set_facecolor("white")
    palette = {
        "embedding": "#7B2CBF",
        "completed": "#2E86DE",
        "partial": "#F39C12",
        "previous": "#E84A5F",
    }
    values_by = {(result.repo_id, category): value for result, category, value in category_rows}
    y = np.arange(len(models))
    left = np.zeros(len(models), dtype=float)
    for category in categories:
        widths = np.array([values_by.get((result.repo_id, category), 0.0) for result in models], dtype=float)
        bars = ax.barh(
            y,
            widths,
            left=left,
            height=0.54,
            label=category,
            color=palette.get(category, "#777777"),
            alpha=0.88,
        )
        for idx, (bar, width) in enumerate(zip(bars, widths)):
            if width < 0.055:
                continue
            ax.text(
                left[idx] + width / 2.0,
                bar.get_y() + bar.get_height() / 2.0,
                f"{width:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )
        left += widths
    ax.set_xlim(0.0, max(1.0, float(left.max()) * 1.04))
    style_axis(ax, "Mean Attention Mass", "Representative Model")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{STYLE[result.group]['name']}\n{result.label}" for result in models])
    finish_outside_legend(ax)
    save_figure(fig, output_dir, "source_category_mass_summary")


def generate_all_plots(results: list[AnalysisResult], output_dir: Path) -> None:
    if not results:
        return
    plot_query_heatmaps(results, output_dir)
    plot_attention_heatmaps(results, output_dir)
    plot_query_summary(results, output_dir)
    plot_query_energy_by_site(results, output_dir)
    plot_site_metric_lines(results, "normalized_entropy_mean", "Normalized Attention Entropy", "attention_entropy_by_site", output_dir)
    plot_site_metric_lines(
        results,
        "effective_sources_mean",
        "Effective Number of Sources",
        "attention_effective_sources_by_site",
        output_dir,
        representative_only=False,
        solid_lines=True,
        selected_results=choose_effective_sources_comparison_models(results),
    )
    plot_site_metric_lines(results, "top1_mass_mean", "Top-1 Source Mass", "attention_top1_mass_by_site", output_dir)
    plot_site_metric_lines(results, "js_generalized", "Generalized JS Input Dependence", "attention_input_dependence_by_site", output_dir)
    plot_site_metric_lines(results, "mean_weight_cv", "Mean Source-Weight CV", "attention_weight_cv_by_site", output_dir)
    plot_category_mass(results, output_dir)
    plot_key_coupling(results, output_dir)
    plot_metric_vs_validation(results, output_dir)


# =============================================================================
# Findings
# =============================================================================

def group_average(results: list[AnalysisResult], group: str, key_path: tuple[str, str]) -> Optional[float]:
    values = []
    for result in results:
        if result.group != group:
            continue
        container = result.query_summary if key_path[0] == "query" else result.mean_attention_metrics() if key_path[0] == "attn" else result.key_summary
        value = safe_float(container.get(key_path[1]))
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else None


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def write_findings(results: list[AnalysisResult], skipped: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Depthwise Routing Findings",
        "",
        "These findings are generated from the validation batches used by `analyze_depthwise_routing.py`.",
        "",
        "## Run Summary",
        "",
        f"- Models analyzed: {len(results)}",
        f"- Models skipped: {len(skipped)}",
    ]
    if results:
        batches = sorted({result.num_batches for result in results if result.use_attnres})
        tokens = sorted({result.tokens_seen for result in results if result.use_attnres})
        lines.append(f"- Batches per routed model: {batches}")
        lines.append(f"- Tokens per routed model: {tokens}")

    lines.extend(["", "## Query Sparsity / Utilization", ""])
    for group in ["standard_attnres", "lr_attnres", "sliced_lr_attnres"]:
        participation = group_average(results, group, ("query", "participation_fraction_mean"))
        hoyer = group_average(results, group, ("query", "hoyer_sparsity_mean"))
        used_1pct = group_average(results, group, ("query", "used_gt_0.01x_max_fraction_mean"))
        lines.append(
            f"- {STYLE[group]['name']}: mean participation fraction {fmt(participation)}, "
            f"Hoyer sparsity {fmt(hoyer)}, dims above 1% of per-site max {fmt(used_1pct)}."
        )

    lines.extend(["", "## Depthwise Attention Pattern", ""])
    for group in ["standard_attnres", "lr_attnres", "sliced_lr_attnres"]:
        entropy = group_average(results, group, ("attn", "normalized_entropy_mean"))
        top1 = group_average(results, group, ("attn", "top1_mass_mean"))
        js = group_average(results, group, ("attn", "js_generalized"))
        eff = group_average(results, group, ("attn", "effective_sources_mean"))
        lines.append(
            f"- {STYLE[group]['name']}: normalized entropy {fmt(entropy)}, "
            f"effective sources {fmt(eff)}, top-1 mass {fmt(top1)}, generalized JS {fmt(js)}."
        )

    lines.extend(["", "## Projected vs Sliced Quirks", ""])
    for group in ["lr_attnres", "sliced_lr_attnres", "standard_attnres"]:
        sim = group_average(results, group, ("key", "key_value_pairwise_similarity_corr"))
        norm_corr = group_average(results, group, ("key", "key_value_norm_corr"))
        tail_energy = group_average(results, group, ("key", "tail_energy_fraction_mean"))
        lines.append(
            f"- {STYLE[group]['name']}: pairwise key/value similarity corr {fmt(sim)}, "
            f"key/value norm corr {fmt(norm_corr)}, tail energy fraction {fmt(tail_energy)}."
        )

    lines.extend(["", "## Per-Model Summary", ""])
    lines.append("| Model | Group | Val. loss | Query participation | Entropy | JS | Top-1 | Pair sim corr |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in sorted(results, key=model_sort_key):
        attn = result.mean_attention_metrics()
        lines.append(
            "| "
            f"{result.label} | "
            f"{STYLE[result.group]['name']} | "
            f"{fmt(result.paper_val_loss, 4)} | "
            f"{fmt(safe_float(result.query_summary.get('participation_fraction_mean')), 4)} | "
            f"{fmt(safe_float(attn.get('normalized_entropy_mean')), 4)} | "
            f"{fmt(safe_float(attn.get('js_generalized')), 4)} | "
            f"{fmt(safe_float(attn.get('top1_mass_mean')), 4)} | "
            f"{fmt(safe_float(result.key_summary.get('key_value_pairwise_similarity_corr')), 4)} |"
        )

    if skipped:
        lines.extend(["", "## Skipped Models", ""])
        for item in skipped:
            lines.append(f"- `{item.get('repo_id')}`: {item.get('reason')}")

    path = output_dir / "depthwise_findings.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze depthwise routing sparsity, attention patterns, and projected-vs-sliced quirks."
    )
    parser.add_argument("--model-scope", choices=("paper_core", "custom"), default="paper_core")
    parser.add_argument("--repos", nargs="*", default=None, help="Explicit model repos or local checkpoints.")
    parser.add_argument("--checkpoint-filename", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Process the entire validation loader; overrides --max-batches.",
    )
    parser.add_argument("--quick", action="store_true", help="Shortcut for --max-batches 2.")
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
    parser.add_argument("--max-pair-samples", type=int, default=512)
    parser.add_argument("--validate-outputs", action="store_true")
    parser.add_argument("--resume-existing", action="store_true", help="Reuse per-model JSON results already present in --output-dir.")
    parser.add_argument("--no-save-intermediate", dest="save_intermediate", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.set_defaults(save_intermediate=True)
    return parser.parse_args()


def selected_repos(args: argparse.Namespace) -> list[str]:
    if args.repos:
        return list(args.repos)
    if args.model_scope == "paper_core":
        return list(PAPER_CORE_REPOS)
    raise RuntimeError("--model-scope custom requires --repos")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.max_batches = 2
    if args.full_validation:
        args.max_batches = None
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    set_academic_rcparams()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"Using device={device}, dtype={dtype}")

    repos = selected_repos(args)
    results: list[AnalysisResult] = []
    skipped: list[dict[str, Any]] = []
    val_loader = None
    use_doc_masking = None

    for repo_id in repos:
        if args.resume_existing:
            cached = load_model_result(output_dir, repo_id)
            if cached is not None:
                print(f"Reusing cached result for {repo_id}: {model_result_path(output_dir, repo_id)}")
                results.append(cached)
                continue

        checkpoint_path, checkpoint_filename, error = resolve_checkpoint_path(
            repo_id=repo_id,
            filename=args.checkpoint_filename,
            revision=args.revision,
            cache_dir=args.cache_dir,
            token=args.hf_token,
            local_files_only=args.local_files_only,
        )
        if checkpoint_path is None:
            skipped.append(
                {
                    "repo_id": repo_id,
                    "checkpoint_filename": checkpoint_filename,
                    "reason": error or "checkpoint not found",
                }
            )
            print(f"Skipping {repo_id}: {error}")
            continue

        loaded: Optional[LoadedModel] = None
        try:
            loaded = load_model_from_checkpoint(repo_id, checkpoint_path, device, dtype)
            config = loaded.model_config
            print(
                f"{repo_id}: group={STYLE[group_from_config(config)]['name']}, "
                f"label={label_from_config(repo_id, config)}, "
                f"use_attnres={config.use_attnres}, use_lrid={config.use_lrid}, "
                f"attnres_type={config.attnres_type}, rank={config.lrid_rank}"
            )
            if val_loader is None and config.use_attnres:
                val_loader, use_doc_masking = build_validation_loader(
                    loaded.train_config,
                    loaded.model_config,
                    args,
                    device,
                )
            if config.use_attnres and val_loader is None:
                raise RuntimeError("Validation dataloader was not initialized")
            result = analyze_loaded_model(
                loaded,
                val_loader,
                bool(use_doc_masking),
                args,
                device,
            )
            results.append(result)
            if args.save_intermediate:
                save_model_result(result, output_dir)
                save_outputs(sorted(results, key=model_sort_key), skipped, output_dir)
        except Exception as exc:  # noqa: BLE001 - keep batch analysis moving.
            skipped.append({"repo_id": repo_id, "checkpoint_path": checkpoint_path, "reason": repr(exc)})
            print(f"Skipping {repo_id} after load failure: {exc!r}")
        finally:
            free_loaded_model(loaded)

    results = sorted(results, key=model_sort_key)
    save_outputs(results, skipped, output_dir)
    generate_all_plots(results, output_dir)
    write_findings(results, skipped, output_dir)


if __name__ == "__main__":
    main()
