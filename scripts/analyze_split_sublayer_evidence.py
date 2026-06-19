#!/usr/bin/env python3
"""Evidence plots for --attnres_block_split_sublayers.

This script uses the saved 1M-token routing analysis artifacts. It does not
rerun model validation. The main analysis is an "oracle compression" test:
take the full-mode per-sublayer routing weights, compress them into block
groups, and measure how much attention-vs-MLP preference would be lost if the
group were forced into a single merged block source.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "lr_attnres_matplotlib_cache"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle


TOTAL_SUBLAYERS = 48
TARGET_BLOCK_COUNTS = (4, 8, 16)

FAMILY_ORDER = {
    "standard_attnres": 0,
    "sliced_lr_attnres": 1,
    "lr_attnres": 2,
}

FAMILY_LABEL = {
    "standard_attnres": "Standard AttnRes",
    "sliced_lr_attnres": "Sliced LR-AttnRes",
    "lr_attnres": "Projected LR-AttnRes",
}

FAMILY_COLOR = {
    "standard_attnres": "#2563EB",
    "sliced_lr_attnres": "#D97706",
    "lr_attnres": "#DC2626",
}

TYPE_ORDER = ("embedding", "attention", "mlp")
TYPE_LABEL = {
    "embedding": "Embedding",
    "attention": "Attention outputs",
    "mlp": "MLP outputs",
}
TYPE_COLOR = {
    "embedding": "#6B7280",
    "attention": "#1D4ED8",
    "mlp": "#D97706",
}


@dataclass(frozen=True)
class ModelRow:
    repo_id: str
    label: str
    group: str
    attnres_type: str
    attnres_num_blocks: int | None
    lrid_rank: int | None
    val_loss: float | None
    added_flops: float | None
    effective_sources: float | None
    js_input_dependence: float | None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_bool(value: str) -> bool:
    return str(value).lower() == "true"


def as_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_models(rows: Iterable[dict[str, str]]) -> list[ModelRow]:
    models = []
    for row in rows:
        if not as_bool(row.get("use_attnres", "False")):
            continue
        models.append(
            ModelRow(
                repo_id=row["repo_id"],
                label=row["label"],
                group=row["group"],
                attnres_type=row["attnres_type"],
                attnres_num_blocks=as_int(row.get("attnres_num_blocks")),
                lrid_rank=as_int(row.get("lrid_rank")),
                val_loss=as_float(row.get("paper_val_loss")),
                added_flops=as_float(row.get("paper_added_flops")),
                effective_sources=as_float(row.get("attn_effective_sources_mean")),
                js_input_dependence=as_float(row.get("attn_js_generalized")),
            )
        )
    return models


def model_sort_key(model: ModelRow) -> tuple[int, int, int, int, str]:
    return (
        FAMILY_ORDER.get(model.group, 99),
        0 if model.attnres_type == "full" else 1,
        model.attnres_num_blocks or 0,
        model.lrid_rank or 0,
        model.repo_id,
    )


def display_label(model: ModelRow, multiline: bool = False) -> str:
    sep = "\n" if multiline else " "
    if model.group == "standard_attnres":
        if model.attnres_type == "full":
            return f"Standard AttnRes{sep}Full"
        return f"Standard AttnRes{sep}n={model.attnres_num_blocks}"
    if model.group == "sliced_lr_attnres":
        rank = model.lrid_rank or 0
        if model.attnres_type == "full":
            return f"Sliced LR-AttnRes{sep}Full r={rank}"
        return f"Sliced LR-AttnRes{sep}n={model.attnres_num_blocks} r={rank}"
    if model.group == "lr_attnres":
        rank = model.lrid_rank or 0
        if model.attnres_type == "full":
            return f"Projected LR-AttnRes{sep}Full r={rank}"
        return f"Projected LR-AttnRes{sep}n={model.attnres_num_blocks} r={rank}"
    return model.label


def block_ends(num_blocks: int) -> list[int]:
    return sorted({math.ceil(TOTAL_SUBLAYERS * i / num_blocks) for i in range(1, num_blocks + 1)})


def block_groups_for_site(num_blocks: int, residual_idx: int) -> list[tuple[int, int]]:
    ends = block_ends(num_blocks)
    groups = []
    previous_end = 0
    for end in ends:
        if end <= residual_idx:
            groups.append((previous_end + 1, end))
            previous_end = end
        else:
            break
    if residual_idx > previous_end:
        groups.append((previous_end + 1, residual_idx))
    return groups


def sublayer_type(idx: int) -> str:
    return "attention" if idx % 2 == 1 else "mlp"


def span_type_fractions(start: int, end: int) -> dict[str, float]:
    if end < start:
        return {"embedding": 0.0, "attention": 0.0, "mlp": 0.0}
    count = end - start + 1
    attn = sum(1 for idx in range(start, end + 1) if sublayer_type(idx) == "attention")
    mlp = count - attn
    return {"embedding": 0.0, "attention": attn / count, "mlp": mlp / count}


def block_source_type_fractions(num_blocks: int, residual_idx: int, source_slot: int) -> dict[str, float]:
    if source_slot == 0:
        return {"embedding": 1.0, "attention": 0.0, "mlp": 0.0}
    groups = block_groups_for_site(num_blocks, residual_idx)
    group_idx = source_slot - 1
    if group_idx < 0 or group_idx >= len(groups):
        raise ValueError(f"cannot map source slot {source_slot} at residual {residual_idx} for n={num_blocks}")
    start, end = groups[group_idx]
    return span_type_fractions(start, end)


def load_source_weights(path: Path, repos: set[str]) -> dict[str, dict[int, dict[int, float]]]:
    out: dict[str, dict[int, dict[int, float]]] = {repo: defaultdict(dict) for repo in repos}
    for row in read_csv(path):
        repo_id = row["repo_id"]
        if repo_id not in repos:
            continue
        out[repo_id][int(row["residual_idx"])][int(row["source_slot"])] = float(row["weight_mean"])
    return out


def full_type_makeup(weights_by_site: dict[int, dict[int, float]]) -> dict[str, float]:
    site_values = []
    for slot_weights in weights_by_site.values():
        row = {key: 0.0 for key in TYPE_ORDER}
        for slot, weight in slot_weights.items():
            if slot == 0:
                row["embedding"] += weight
            elif sublayer_type(slot) == "attention":
                row["attention"] += weight
            else:
                row["mlp"] += weight
        site_values.append(row)
    return {key: float(np.mean([row[key] for row in site_values])) for key in TYPE_ORDER}


def block_type_makeup(model: ModelRow, weights_by_site: dict[int, dict[int, float]]) -> dict[str, float]:
    if model.attnres_num_blocks is None:
        raise ValueError("block model is missing attnres_num_blocks")
    site_values = []
    for residual_idx, slot_weights in weights_by_site.items():
        row = {key: 0.0 for key in TYPE_ORDER}
        for slot, weight in slot_weights.items():
            fractions = block_source_type_fractions(model.attnres_num_blocks, residual_idx, slot)
            for key in TYPE_ORDER:
                row[key] += weight * fractions[key]
        site_values.append(row)
    return {key: float(np.mean([row[key] for row in site_values])) for key in TYPE_ORDER}


def oracle_compression_rows(full_models: list[ModelRow], source_weights: dict[str, dict[int, dict[int, float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in full_models:
        weights_by_site = source_weights[model.repo_id]
        for target_n in TARGET_BLOCK_COUNTS:
            site_mixed_mass = []
            site_forced_l1 = []
            mixed_total = 0.0
            abs_share_num = 0.0
            gap_num = 0.0
            observed_attn_num = 0.0
            count_attn_num = 0.0
            forced_l1_total = 0.0
            for residual_idx, slot_weights in weights_by_site.items():
                mixed_mass = 0.0
                forced_l1 = 0.0
                for start, end in block_groups_for_site(target_n, residual_idx):
                    attn_slots = [idx for idx in range(start, end + 1) if sublayer_type(idx) == "attention"]
                    mlp_slots = [idx for idx in range(start, end + 1) if sublayer_type(idx) == "mlp"]
                    if not attn_slots or not mlp_slots:
                        continue
                    attn_mass = sum(slot_weights.get(slot, 0.0) for slot in attn_slots)
                    mlp_mass = sum(slot_weights.get(slot, 0.0) for slot in mlp_slots)
                    total = attn_mass + mlp_mass
                    if total <= 0.0:
                        continue
                    observed_attn_share = attn_mass / total
                    count_attn_share = len(attn_slots) / (end - start + 1)
                    mixed_mass += total
                    group_forced_l1 = 2.0 * total * abs(observed_attn_share - count_attn_share)
                    forced_l1 += group_forced_l1
                    mixed_total += total
                    abs_share_num += total * abs(observed_attn_share - count_attn_share)
                    gap_num += total * abs(attn_mass - mlp_mass) / total
                    observed_attn_num += attn_mass
                    count_attn_num += total * count_attn_share
                    forced_l1_total += group_forced_l1
                site_mixed_mass.append(mixed_mass)
                site_forced_l1.append(forced_l1)

            denom = max(mixed_total, 1e-30)
            rows.append(
                {
                    "repo_id": model.repo_id,
                    "label": display_label(model),
                    "group": model.group,
                    "group_name": FAMILY_LABEL[model.group],
                    "target_block_count": target_n,
                    "merged_max_sources": target_n + 1,
                    "split_max_sources": 2 * target_n + 1,
                    "mixed_mass_mean": float(np.mean(site_mixed_mass)),
                    "forced_mix_l1_mean": float(np.mean(site_forced_l1)),
                    "forced_mix_l1_per_mixed_mass": float(forced_l1_total / denom),
                    "weighted_abs_type_share_distortion": float(abs_share_num / denom),
                    "weighted_attn_mlp_gap": float(gap_num / denom),
                    "observed_attn_share_in_mixed": float(observed_attn_num / denom),
                    "count_attn_share_in_mixed": float(count_attn_num / denom),
                    "paper_val_loss": model.val_loss,
                }
            )
    return rows


def aggregate_oracle_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for group in FAMILY_ORDER:
        for target_n in TARGET_BLOCK_COUNTS:
            subset = [row for row in rows if row["group"] == group and row["target_block_count"] == target_n]
            if not subset:
                continue
            out.append(
                {
                    "group": group,
                    "group_name": FAMILY_LABEL[group],
                    "target_block_count": target_n,
                    "models": len(subset),
                    "mixed_mass_mean": float(np.mean([float(row["mixed_mass_mean"]) for row in subset])),
                    "forced_mix_l1_mean": float(np.mean([float(row["forced_mix_l1_mean"]) for row in subset])),
                    "forced_mix_l1_per_mixed_mass": float(np.mean([float(row["forced_mix_l1_per_mixed_mass"]) for row in subset])),
                    "weighted_abs_type_share_distortion": float(np.mean([float(row["weighted_abs_type_share_distortion"]) for row in subset])),
                    "weighted_attn_mlp_gap": float(np.mean([float(row["weighted_attn_mlp_gap"]) for row in subset])),
                }
            )
    return out


def oracle_site_rows(
    full_models: list[ModelRow],
    source_weights: dict[str, dict[int, dict[int, float]]],
    target_n: int = 8,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in full_models:
        for residual_idx, slot_weights in sorted(source_weights[model.repo_id].items()):
            mixed_total = 0.0
            forced_l1 = 0.0
            observed_attn_num = 0.0
            count_attn_num = 0.0
            for start, end in block_groups_for_site(target_n, residual_idx):
                attn_slots = [idx for idx in range(start, end + 1) if sublayer_type(idx) == "attention"]
                mlp_slots = [idx for idx in range(start, end + 1) if sublayer_type(idx) == "mlp"]
                if not attn_slots or not mlp_slots:
                    continue
                attn_mass = sum(slot_weights.get(slot, 0.0) for slot in attn_slots)
                mlp_mass = sum(slot_weights.get(slot, 0.0) for slot in mlp_slots)
                total = attn_mass + mlp_mass
                if total <= 0.0:
                    continue
                count_attn_share = len(attn_slots) / (end - start + 1)
                forced_l1 += 2.0 * total * abs(attn_mass / total - count_attn_share)
                mixed_total += total
                observed_attn_num += attn_mass
                count_attn_num += total * count_attn_share
            rows.append(
                {
                    "repo_id": model.repo_id,
                    "label": display_label(model),
                    "group": model.group,
                    "group_name": FAMILY_LABEL[model.group],
                    "target_block_count": target_n,
                    "residual_idx": residual_idx,
                    "mixed_mass": mixed_total,
                    "forced_mix_l1": forced_l1,
                    "forced_mix_l1_per_mixed_mass": forced_l1 / mixed_total if mixed_total > 0.0 else 0.0,
                    "observed_attn_share": observed_attn_num / mixed_total if mixed_total > 0.0 else 0.0,
                    "count_attn_share": count_attn_num / mixed_total if mixed_total > 0.0 else 0.0,
                }
            )
    return rows


def matched_full_model(block_model: ModelRow, full_models: list[ModelRow]) -> ModelRow:
    candidates = [model for model in full_models if model.group == block_model.group]
    if block_model.group == "standard_attnres":
        return candidates[0]
    if block_model.group == "sliced_lr_attnres":
        rank = block_model.lrid_rank
        exact = [model for model in candidates if model.lrid_rank == rank]
        if exact:
            return exact[0]
        return min(candidates, key=lambda model: abs((model.lrid_rank or 0) - (rank or 0)))
    if block_model.group == "lr_attnres":
        rank = block_model.lrid_rank
        exact = [model for model in candidates if model.lrid_rank == rank]
        if exact:
            return exact[0]
        return min(candidates, key=lambda model: abs((model.lrid_rank or 0) - (rank or 0)))
    raise ValueError(f"cannot match full model for {block_model.repo_id}")


def block_full_distance_rows(
    block_models: list[ModelRow],
    full_models: list[ModelRow],
    source_weights: dict[str, dict[int, dict[int, float]]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    full_makeups = {model.repo_id: full_type_makeup(source_weights[model.repo_id]) for model in full_models}
    rows = []
    for model in block_models:
        full_model = matched_full_model(model, full_models)
        block_makeup = block_type_makeup(model, source_weights[model.repo_id])
        full_makeup = full_makeups[full_model.repo_id]
        deltas = {key: block_makeup[key] - full_makeup[key] for key in TYPE_ORDER}
        rows.append(
            {
                "repo_id": model.repo_id,
                "label": display_label(model),
                "group": model.group,
                "group_name": FAMILY_LABEL[model.group],
                "attnres_num_blocks": model.attnres_num_blocks,
                "lrid_rank": model.lrid_rank,
                "matched_full_repo_id": full_model.repo_id,
                "matched_full_label": display_label(full_model),
                "block_embedding_mean": block_makeup["embedding"],
                "block_attention_mean": block_makeup["attention"],
                "block_mlp_mean": block_makeup["mlp"],
                "full_embedding_mean": full_makeup["embedding"],
                "full_attention_mean": full_makeup["attention"],
                "full_mlp_mean": full_makeup["mlp"],
                "delta_embedding": deltas["embedding"],
                "delta_attention": deltas["attention"],
                "delta_mlp": deltas["mlp"],
                "type_makeup_l1_distance": sum(abs(deltas[key]) for key in TYPE_ORDER),
                "paper_val_loss": model.val_loss,
                "matched_full_val_loss": full_model.val_loss,
                "effective_sources": model.effective_sources,
            }
        )
    return rows, full_makeups


def source_budget_rows(models: list[ModelRow]) -> list[dict[str, object]]:
    rows = []
    for num_blocks in TARGET_BLOCK_COUNTS:
        rows.append(
            {
                "variant": f"non-split n={num_blocks}",
                "attnres_num_blocks": num_blocks,
                "split_sublayers": False,
                "max_sources": num_blocks + 1,
                "completed_depth_sources": num_blocks,
            }
        )
        rows.append(
            {
                "variant": f"split n={num_blocks}",
                "attnres_num_blocks": num_blocks,
                "split_sublayers": True,
                "max_sources": 2 * num_blocks + 1,
                "completed_depth_sources": 2 * num_blocks,
            }
        )
    for model in models:
        if model.attnres_type != "block":
            continue
        rows.append(
            {
                "variant": display_label(model),
                "attnres_num_blocks": model.attnres_num_blocks,
                "split_sublayers": False,
                "max_sources": (model.attnres_num_blocks or 0) + 1,
                "completed_depth_sources": model.attnres_num_blocks,
                "paper_val_loss": model.val_loss,
                "effective_sources": model.effective_sources,
                "paper_added_flops": model.added_flops,
                "group": model.group,
                "group_name": FAMILY_LABEL.get(model.group, model.group),
            }
        )
    return rows


def non_split_source_count_sum(num_blocks: int) -> int:
    return sum(1 + len(block_groups_for_site(num_blocks, residual_idx)) for residual_idx in range(1, TOTAL_SUBLAYERS + 1))


def split_source_count_at_site(num_blocks: int, residual_idx: int) -> int:
    count = 1
    for start, end in block_groups_for_site(num_blocks, residual_idx):
        is_completed = end in block_ends(num_blocks) and end <= residual_idx
        if is_completed:
            count += 2
            continue
        types = {sublayer_type(idx) for idx in range(start, end + 1)}
        count += len(types)
    return count


def split_source_count_sum(num_blocks: int) -> int:
    return sum(split_source_count_at_site(num_blocks, residual_idx) for residual_idx in range(1, TOTAL_SUBLAYERS + 1))


def controlled_split_flop_rows(models: list[ModelRow]) -> list[dict[str, object]]:
    full_source_sum = sum(residual_idx + 1 for residual_idx in range(1, TOTAL_SUBLAYERS + 1))

    def by_family(attnres_type: str, group: str, rank: int | None = None) -> ModelRow:
        matches = [
            model
            for model in models
            if model.attnres_type == attnres_type
            and model.group == group
            and (rank is None or model.lrid_rank == rank)
        ]
        if not matches:
            raise RuntimeError(f"missing model for {group}, {attnres_type}, rank={rank}")
        return sorted(matches, key=model_sort_key)[0]

    standard_full = by_family("full", "standard_attnres")
    sliced_full_r64 = by_family("full", "sliced_lr_attnres", 64)
    sliced_full_r32 = by_family("full", "sliced_lr_attnres", 32)
    projected_full_r32 = by_family("full", "lr_attnres", 32)
    projected_key_const = (projected_full_r32.added_flops or 0.0) - (sliced_full_r32.added_flops or 0.0)

    families = [
        {
            "family": "standard_attnres",
            "family_name": "Standard AttnRes",
            "rank": None,
            "full_kernel_flops": standard_full.added_flops or 0.0,
            "projected_key_const": 0.0,
        },
        {
            "family": "sliced_lr_attnres",
            "family_name": "Sliced LR-AttnRes r=64",
            "rank": 64,
            "full_kernel_flops": sliced_full_r64.added_flops or 0.0,
            "projected_key_const": 0.0,
        },
        {
            "family": "lr_attnres",
            "family_name": "Projected LR-AttnRes r=32",
            "rank": 32,
            "full_kernel_flops": sliced_full_r32.added_flops or 0.0,
            "projected_key_const": projected_key_const,
        },
    ]

    rows = []
    for target_n in TARGET_BLOCK_COUNTS:
        if target_n % 2 != 0:
            continue
        split_blocks = target_n // 2
        non_split_sum = non_split_source_count_sum(target_n)
        split_sum = split_source_count_sum(split_blocks)
        for family in families:
            kernel_per_source_sum = family["full_kernel_flops"] / full_source_sum
            non_split_kernel = kernel_per_source_sum * non_split_sum
            split_kernel = kernel_per_source_sum * split_sum
            non_split_total = non_split_kernel + family["projected_key_const"]
            split_total = split_kernel + family["projected_key_const"]
            rows.append(
                {
                    "family": family["family"],
                    "family_name": family["family_name"],
                    "rank": family["rank"],
                    "non_split_blocks": target_n,
                    "split_blocks": split_blocks,
                    "final_max_sources": target_n + 1,
                    "non_split_source_sum": non_split_sum,
                    "split_source_sum": split_sum,
                    "non_split_avg_sources": non_split_sum / TOTAL_SUBLAYERS,
                    "split_avg_sources": split_sum / TOTAL_SUBLAYERS,
                    "source_sum_overhead_pct": 100.0 * (split_sum / non_split_sum - 1.0),
                    "non_split_added_flops_pct": non_split_total,
                    "split_added_flops_pct": split_total,
                    "absolute_flop_delta_pct": split_total - non_split_total,
                    "relative_total_flop_overhead_pct": 100.0 * (split_total / non_split_total - 1.0),
                    "relative_depth_kernel_overhead_pct": 100.0 * (split_kernel / non_split_kernel - 1.0),
                    "projected_key_const_pct": family["projected_key_const"],
                }
            )
    return rows


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
        }
    )


def save_figure(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_structure_diagram(output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 3.4), sharex=True)
    fig.patch.set_facecolor("white")
    sublayers = list(range(1, 13))
    block_ranges = [(1, 6), (7, 12)]

    for ax, title, split in [
        (axes[0], "Default block compression: one mixed source per block", False),
        (axes[1], "Split-sublayer compression: separate attention and MLP summaries per block", True),
    ]:
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlim(0.5, 12.5)
        ax.grid(False)
        for idx in sublayers:
            kind = sublayer_type(idx)
            ax.add_patch(
                Rectangle(
                    (idx - 0.38, 0.08),
                    0.76,
                    0.18,
                    facecolor=TYPE_COLOR[kind],
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.95,
                )
            )
            ax.text(idx, 0.17, "A" if kind == "attention" else "M", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        for block_idx, (start, end) in enumerate(block_ranges, start=1):
            if split:
                ax.add_patch(Rectangle((start - 0.45, 0.50), end - start + 0.9, 0.15, facecolor="#DBEAFE", edgecolor="#1D4ED8", linewidth=1.0))
                ax.add_patch(Rectangle((start - 0.45, 0.70), end - start + 0.9, 0.15, facecolor="#FFEDD5", edgecolor="#D97706", linewidth=1.0))
                ax.text((start + end) / 2, 0.575, f"Block {block_idx} attention source", ha="center", va="center", color="#1E3A8A", fontsize=8)
                ax.text((start + end) / 2, 0.775, f"Block {block_idx} MLP source", ha="center", va="center", color="#7C2D12", fontsize=8)
            else:
                ax.add_patch(Rectangle((start - 0.45, 0.58), end - start + 0.9, 0.22, facecolor="#E5E7EB", edgecolor="#4B5563", linewidth=1.0))
                ax.text((start + end) / 2, 0.69, f"Block {block_idx} mixed source", ha="center", va="center", color="#111827", fontsize=8)
    axes[-1].set_xticks(sublayers)
    axes[-1].set_xlabel("Residual-writing sublayer index")
    legend = [
        Patch(facecolor=TYPE_COLOR["attention"], label="Attention output"),
        Patch(facecolor=TYPE_COLOR["mlp"], label="MLP output"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, output_dir, "split_sublayer_structure_diagram")


def plot_oracle_family_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    configure_matplotlib()
    metrics = [
        ("mixed_mass_mean", "Mass in Mixed Blocks"),
        ("forced_mix_l1_mean", "Forced-Mix L1 Error"),
        ("forced_mix_l1_per_mixed_mass", "Error / Mixed Mass"),
        ("weighted_attn_mlp_gap", "Attn/MLP Gap in Mixed Blocks"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.2, 3.7), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        for group in FAMILY_ORDER:
            subset = sorted([row for row in rows if row["group"] == group], key=lambda row: int(row["target_block_count"]))
            xs = [int(row["target_block_count"]) for row in subset]
            ys = [float(row[metric]) for row in subset]
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=FAMILY_COLOR[group], label=FAMILY_LABEL[group])
        ax.set_title(title)
        ax.set_xticks(list(TARGET_BLOCK_COUNTS))
        ax.set_xlabel("Target block count")
    axes[0].set_ylabel("Mean over full models")
    axes[-1].legend(loc="best")
    fig.suptitle("Oracle Test: What Full Routing Loses Under Merged Block Compression", y=1.05, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "oracle_compression_family_summary")


def plot_oracle_model_heatmap(rows: list[dict[str, object]], output_dir: Path) -> None:
    configure_matplotlib()
    models = []
    seen = set()
    for row in rows:
        repo_id = str(row["repo_id"])
        if repo_id not in seen:
            models.append((repo_id, str(row["label"]), str(row["group"])))
            seen.add(repo_id)
    data = np.zeros((len(models), len(TARGET_BLOCK_COUNTS)), dtype=float)
    for i, (repo_id, _, _) in enumerate(models):
        for j, target_n in enumerate(TARGET_BLOCK_COUNTS):
            match = next(row for row in rows if row["repo_id"] == repo_id and row["target_block_count"] == target_n)
            data[i, j] = float(match["forced_mix_l1_mean"])

    fig, ax = plt.subplots(figsize=(6.8, max(4.4, 0.36 * len(models))))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(TARGET_BLOCK_COUNTS)))
    ax.set_xticklabels([f"n={n}" for n in TARGET_BLOCK_COUNTS])
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([label for _, label, _ in models])
    ax.set_title("Forced-Mix L1 Error if Full Routing Is Compressed")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7, color="#111827")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean L1 source-type mass error")
    fig.tight_layout()
    save_figure(fig, output_dir, "oracle_compression_model_heatmap")


def plot_oracle_site_error(rows: list[dict[str, object]], output_dir: Path) -> None:
    configure_matplotlib()
    selected_labels = {
        "Standard AttnRes Full",
        "Sliced LR-AttnRes Full r=64",
        "Sliced LR-AttnRes Full r=512",
        "Projected LR-AttnRes Full r=32",
    }
    selected_rows = [row for row in rows if row["label"] in selected_labels]
    labels = []
    for row in selected_rows:
        label = str(row["label"])
        if label not in labels:
            labels.append(label)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), sharex=True)
    for label in labels:
        subset = sorted([row for row in selected_rows if row["label"] == label], key=lambda row: int(row["residual_idx"]))
        group = str(subset[0]["group"])
        color = FAMILY_COLOR[group]
        if "r=512" in label:
            linestyle = "--"
        elif "r=64" in label:
            linestyle = "-."
        else:
            linestyle = "-"
        xs = [int(row["residual_idx"]) for row in subset]
        axes[0].plot(xs, [float(row["forced_mix_l1"]) for row in subset], color=color, linestyle=linestyle, linewidth=1.7, label=label)
        axes[1].plot(
            xs,
            [float(row["observed_attn_share"]) - float(row["count_attn_share"]) for row in subset],
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            label=label,
        )
    axes[0].set_title("Depth-Wise Forced-Mix Error at n=8")
    axes[0].set_ylabel("Per-site L1 source-type mass error")
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_title("Observed Attn Share minus Count-Implied Share")
    axes[1].set_ylabel("Share delta")
    for ax in axes:
        ax.set_xlabel("Residual read site")
    axes[1].legend(loc="best", fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_dir, "oracle_n8_site_error_by_depth")


def plot_block_full_distance(rows: list[dict[str, object]], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for group in FAMILY_ORDER:
        subset = sorted([row for row in rows if row["group"] == group], key=lambda row: int(row["attnres_num_blocks"]))
        xs = [int(row["attnres_num_blocks"]) for row in subset]
        ys = [float(row["type_makeup_l1_distance"]) for row in subset]
        axes[0].plot(xs, ys, marker="o", linewidth=2.0, color=FAMILY_COLOR[group], label=FAMILY_LABEL[group])
    axes[0].set_xticks(list(TARGET_BLOCK_COUNTS))
    axes[0].set_xlabel("Block count")
    axes[0].set_ylabel("L1 distance to matched full model")
    axes[0].set_title("Actual Non-Split Blocks Distort Source-Type Makeup")
    axes[0].legend(loc="best")

    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(rows))
    bottom_pos = np.zeros(len(rows))
    bottom_neg = np.zeros(len(rows))
    for key, color, label in [
        ("delta_embedding", TYPE_COLOR["embedding"], "Embedding delta"),
        ("delta_attention", TYPE_COLOR["attention"], "Attention delta"),
        ("delta_mlp", TYPE_COLOR["mlp"], "MLP delta"),
    ]:
        vals = np.array([float(row[key]) for row in rows])
        pos = np.maximum(vals, 0.0)
        neg = np.minimum(vals, 0.0)
        axes[1].bar(x, pos, bottom=bottom_pos, color=color, edgecolor="white", linewidth=0.5, label=label)
        axes[1].bar(x, neg, bottom=bottom_neg, color=color, edgecolor="white", linewidth=0.5)
        bottom_pos += pos
        bottom_neg += neg
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("Block minus matched full")
    axes[1].set_title("Direction of the Distortion")
    axes[1].legend(loc="best", fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_dir, "block_vs_full_type_makeup_distance")


def plot_validation_source_budget(block_models: list[ModelRow], full_models: list[ModelRow], output_dir: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))

    for group in FAMILY_ORDER:
        subset = sorted([m for m in block_models if m.group == group], key=lambda m: m.attnres_num_blocks or 0)
        xs = [(m.attnres_num_blocks or 0) + 1 for m in subset]
        ys = [m.val_loss for m in subset]
        axes[0].plot(xs, ys, marker="o", linewidth=2.0, color=FAMILY_COLOR[group], label=FAMILY_LABEL[group])
        for x, y, model in zip(xs, ys, subset):
            axes[0].annotate(f"n={model.attnres_num_blocks}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[0].set_xlabel("Non-split max sources")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Existing Non-Split Block Results")
    axes[0].legend(loc="best")

    width = 0.34
    ns = np.array(list(TARGET_BLOCK_COUNTS), dtype=float)
    non_split_sources = ns + 1
    split_sources = 2 * ns + 1
    axes[1].bar(ns - width / 2, non_split_sources, width=width, color="#94A3B8", label="Non-split")
    axes[1].bar(ns + width / 2, split_sources, width=width, color="#F97316", label="Split sublayers")
    axes[1].axhline(17, color="#111827", linestyle="--", linewidth=1.0)
    axes[1].text(4.15, 17.5, "split n=8 ~= non-split n=16 source budget", fontsize=8, color="#111827")
    axes[1].set_xticks(ns)
    axes[1].set_xticklabels([f"n={int(n)}" for n in ns])
    axes[1].set_ylabel("Max depth sources")
    axes[1].set_title("Source Budget Equivalence")
    axes[1].legend(loc="best")
    fig.tight_layout()
    save_figure(fig, output_dir, "validation_and_source_budget")


def plot_controlled_split_flops(rows: list[dict[str, object]], output_dir: Path) -> None:
    configure_matplotlib()
    families = []
    for row in rows:
        family_name = str(row["family_name"])
        if family_name not in families:
            families.append(family_name)

    fig, axes = plt.subplots(1, len(families), figsize=(14.2, 4.0), sharey=False)
    if len(families) == 1:
        axes = [axes]
    width = 0.34
    for ax, family_name in zip(axes, families):
        subset = [row for row in rows if row["family_name"] == family_name]
        xs = np.arange(len(subset))
        non_split = np.array([float(row["non_split_added_flops_pct"]) for row in subset])
        split = np.array([float(row["split_added_flops_pct"]) for row in subset])
        ax.bar(xs - width / 2, non_split, width=width, color="#94A3B8", label="non-split N")
        ax.bar(xs + width / 2, split, width=width, color="#F97316", label="split N/2")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"N={row['non_split_blocks']}\\nfinal {row['final_max_sources']} src" for row in subset])
        ax.set_title(family_name)
        ax.set_ylabel("Added FLOPs (%)")
        for x, row in zip(xs, subset):
            delta = float(row["absolute_flop_delta_pct"])
            ax.text(x, max(float(row["split_added_flops_pct"]), float(row["non_split_added_flops_pct"])) * 1.015, f"+{delta:.4f}", ha="center", va="bottom", fontsize=7)
    axes[0].legend(loc="best")
    fig.suptitle("Controlled Final Source Count: Non-Split N vs Split N/2", y=1.04, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "controlled_split_flops")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("figures/depthwise_analysis_1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/depthwise_analysis_1m/split_sublayer_evidence"))
    args = parser.parse_args()

    routing_path = args.input_dir / "routing_metrics.csv"
    source_path = args.input_dir / "source_slot_metrics.csv"
    models = parse_models(read_csv(routing_path))
    full_models = sorted([model for model in models if model.attnres_type == "full"], key=model_sort_key)
    block_models = sorted([model for model in models if model.attnres_type == "block"], key=model_sort_key)
    source_weights = load_source_weights(source_path, {model.repo_id for model in full_models + block_models})

    oracle_rows = oracle_compression_rows(full_models, source_weights)
    oracle_family_rows = aggregate_oracle_rows(oracle_rows)
    oracle_n8_site_rows = oracle_site_rows(full_models, source_weights, target_n=8)
    distance_rows, full_makeups = block_full_distance_rows(block_models, full_models, source_weights)
    budget_rows = source_budget_rows(models)
    controlled_flop_rows = controlled_split_flop_rows(models)

    full_makeup_rows = [
        {
            "repo_id": model.repo_id,
            "label": display_label(model),
            "group": model.group,
            "group_name": FAMILY_LABEL[model.group],
            **full_makeups[model.repo_id],
        }
        for model in full_models
    ]

    write_csv(
        args.output_dir / "oracle_compression_by_model.csv",
        oracle_rows,
        [
            "repo_id",
            "label",
            "group",
            "group_name",
            "target_block_count",
            "merged_max_sources",
            "split_max_sources",
            "mixed_mass_mean",
            "forced_mix_l1_mean",
            "forced_mix_l1_per_mixed_mass",
            "weighted_abs_type_share_distortion",
            "weighted_attn_mlp_gap",
            "observed_attn_share_in_mixed",
            "count_attn_share_in_mixed",
            "paper_val_loss",
        ],
    )
    write_csv(
        args.output_dir / "oracle_compression_family_summary.csv",
        oracle_family_rows,
        [
            "group",
            "group_name",
            "target_block_count",
            "models",
            "mixed_mass_mean",
            "forced_mix_l1_mean",
            "forced_mix_l1_per_mixed_mass",
            "weighted_abs_type_share_distortion",
            "weighted_attn_mlp_gap",
        ],
    )
    write_csv(
        args.output_dir / "oracle_n8_site_error.csv",
        oracle_n8_site_rows,
        [
            "repo_id",
            "label",
            "group",
            "group_name",
            "target_block_count",
            "residual_idx",
            "mixed_mass",
            "forced_mix_l1",
            "forced_mix_l1_per_mixed_mass",
            "observed_attn_share",
            "count_attn_share",
        ],
    )
    write_csv(
        args.output_dir / "block_vs_full_type_makeup_distance.csv",
        distance_rows,
        [
            "repo_id",
            "label",
            "group",
            "group_name",
            "attnres_num_blocks",
            "lrid_rank",
            "matched_full_repo_id",
            "matched_full_label",
            "block_embedding_mean",
            "block_attention_mean",
            "block_mlp_mean",
            "full_embedding_mean",
            "full_attention_mean",
            "full_mlp_mean",
            "delta_embedding",
            "delta_attention",
            "delta_mlp",
            "type_makeup_l1_distance",
            "paper_val_loss",
            "matched_full_val_loss",
            "effective_sources",
        ],
    )
    write_csv(
        args.output_dir / "full_type_makeup.csv",
        full_makeup_rows,
        ["repo_id", "label", "group", "group_name", "embedding", "attention", "mlp"],
    )
    write_csv(
        args.output_dir / "source_budget.csv",
        budget_rows,
        [
            "variant",
            "attnres_num_blocks",
            "split_sublayers",
            "max_sources",
            "completed_depth_sources",
            "paper_val_loss",
            "effective_sources",
            "paper_added_flops",
            "group",
            "group_name",
        ],
    )
    write_csv(
        args.output_dir / "controlled_split_flops.csv",
        controlled_flop_rows,
        [
            "family",
            "family_name",
            "rank",
            "non_split_blocks",
            "split_blocks",
            "final_max_sources",
            "non_split_source_sum",
            "split_source_sum",
            "non_split_avg_sources",
            "split_avg_sources",
            "source_sum_overhead_pct",
            "non_split_added_flops_pct",
            "split_added_flops_pct",
            "absolute_flop_delta_pct",
            "relative_total_flop_overhead_pct",
            "relative_depth_kernel_overhead_pct",
            "projected_key_const_pct",
        ],
    )

    plot_structure_diagram(args.output_dir)
    plot_oracle_family_summary(oracle_family_rows, args.output_dir)
    plot_oracle_model_heatmap(oracle_rows, args.output_dir)
    plot_oracle_site_error(oracle_n8_site_rows, args.output_dir)
    plot_block_full_distance(distance_rows, args.output_dir)
    plot_validation_source_budget(block_models, full_models, args.output_dir)
    plot_controlled_split_flops(controlled_flop_rows, args.output_dir)

    print(f"Wrote split-sublayer evidence artifacts to {args.output_dir}")
    print("Top oracle forced-mix errors:")
    for row in sorted(oracle_rows, key=lambda item: float(item["forced_mix_l1_mean"]), reverse=True)[:8]:
        print(
            f"  {row['label']} compressed to n={row['target_block_count']}: "
            f"L1={float(row['forced_mix_l1_mean']):.4f}, "
            f"mixed_mass={float(row['mixed_mass_mean']):.4f}, "
            f"error/mixed={float(row['forced_mix_l1_per_mixed_mass']):.4f}"
        )


if __name__ == "__main__":
    main()
