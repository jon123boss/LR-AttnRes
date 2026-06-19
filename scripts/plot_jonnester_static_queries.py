#!/usr/bin/env python3
"""Plot static AttnRes/LR-AttnRes query heatmaps for Jonnester Hub models."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "lr_attnres_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "lr_attnres_xdg_cache"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_depthwise_routing import (  # noqa: E402
    PAPER_ADDED_FLOPS,
    PAPER_VAL_LOSS,
    STYLE,
    choose_checkpoint_file,
    collect_query_metrics,
    group_from_config,
    json_default,
    label_from_config,
    load_model_from_checkpoint,
    model_result_path,
    model_sort_key,
    result_from_payload,
    result_payload,
    sanitize_filename,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", default="Jonnester")
    parser.add_argument(
        "--candidate-substring",
        default="LR-AttnRes",
        help="Only download/inspect repos whose id contains this substring. Empty string inspects every repo.",
    )
    parser.add_argument(
        "--existing-results-dir",
        type=Path,
        default=Path("figures/depthwise_analysis_1m"),
        help="Reuse query matrices from this analysis directory before downloading checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/pdf/jonnester_static_queries"))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--model-list-file",
        type=Path,
        default=None,
        help="Read repo ids from a saved text file instead of listing Hugging Face models.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only use cached query JSON; do not download missing checkpoints.",
    )
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--normalization",
        choices=("none", "dim", "sqrt_dim", "row_rms"),
        default="none",
        help=(
            "Transform plotted values: none=|q|, dim=|q|/D, "
            "sqrt_dim=|q|/sqrt(D), row_rms=|q|/RMS(q_site)."
        ),
    )
    parser.add_argument(
        "--scale-mode",
        choices=("global", "per_model"),
        default="global",
        help="Use one shared color scale or a separate percentile-clipped colorbar per model.",
    )
    parser.add_argument(
        "--scale-percentile",
        type=float,
        default=99.5,
        help="Percentile used to clip heatmap colors.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep downloaded checkpoint files in the Hugging Face cache. By default they are deleted after loading.",
    )
    return parser.parse_args()


def list_author_models(api: HfApi, author: str, token: str | None) -> list[str]:
    models = api.list_models(author=author, token=token, full=False)
    ids = []
    for model in models:
        model_id = getattr(model, "modelId", None) or getattr(model, "id", None)
        if model_id:
            ids.append(str(model_id))
    return sorted(set(ids), key=str.lower)


def load_cached_result(repo_id: str, existing_results_dir: Path, output_dir: Path):
    for base in (output_dir, existing_results_dir):
        path = model_result_path(base, repo_id)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return result_from_payload(payload), path
    return None, None


def lightweight_result(repo_id: str, checkpoint_path: str, loaded) -> Any:
    config = loaded.model_config
    result = type("StaticQueryResult", (), {})()
    result.repo_id = repo_id
    result.checkpoint_path = checkpoint_path
    result.checkpoint_filename = Path(checkpoint_path).name
    result.group = group_from_config(config)
    result.label = label_from_config(repo_id, config)
    result.use_attnres = bool(config.use_attnres)
    result.use_lrid = bool(config.use_lrid)
    result.attnres_type = config.attnres_type if config.use_attnres else None
    result.attnres_num_blocks = config.attnres_num_blocks if config.use_attnres else None
    result.lrid_rank = config.lrid_rank if config.use_lrid else None
    result.lrid_num_heads = config.lrid_num_heads if config.use_lrid else None
    result.lrid_input_dependent_query = bool(getattr(config, "lrid_input_dependent_query", False))
    result.lrid_key_from_output_tail = bool(getattr(config, "lrid_key_from_output_tail", False))
    result.paper_val_loss = PAPER_VAL_LOSS.get(repo_id)
    result.paper_added_flops = PAPER_ADDED_FLOPS.get(repo_id)
    result.num_batches = 0
    result.tokens_seen = 0
    result.query_matrix, result.query_site_metrics, result.query_summary = collect_query_metrics(loaded.model)
    result.attention_site_metrics = []
    result.source_slot_metrics = []
    result.category_metrics = []
    result.key_summary = {}
    result.output_check = {}

    def mean_attention_metrics():
        return {}

    result.mean_attention_metrics = mean_attention_metrics
    result.summary_row = lambda: {}
    return result


def save_lightweight_result(result: Any, output_dir: Path) -> None:
    path = model_result_path(output_dir, result.repo_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_payload(result), indent=2, default=json_default) + "\n", encoding="utf-8")


def delete_downloaded_checkpoint(checkpoint_path: str | None) -> list[str]:
    if not checkpoint_path:
        return []
    deleted: list[str] = []
    path = Path(checkpoint_path)
    candidates = [path]
    try:
        resolved = path.resolve(strict=True)
        if resolved != path and "huggingface" in resolved.parts and "blobs" in resolved.parts:
            candidates.append(resolved)
    except OSError:
        pass

    for candidate in candidates:
        try:
            if candidate.is_file() or candidate.is_symlink():
                candidate.unlink()
                deleted.append(str(candidate))
            elif candidate.is_dir():
                shutil.rmtree(candidate)
                deleted.append(str(candidate))
        except FileNotFoundError:
            continue
        except OSError as exc:
            deleted.append(f"{candidate} [delete failed: {exc}]")
    return deleted


def title_for(result: Any) -> str:
    group = getattr(result, "group", "unknown")
    group_name = STYLE.get(group, {}).get("name", group)
    repo_name = result.repo_id.split("/", 1)[-1]
    details = []
    attnres_num_blocks = getattr(result, "attnres_num_blocks", None)
    lrid_rank = getattr(result, "lrid_rank", None)
    lrid_num_heads = getattr(result, "lrid_num_heads", None)
    if attnres_num_blocks:
        details.append(f"n={attnres_num_blocks}")
    if lrid_rank:
        details.append(f"r={lrid_rank}")
    if lrid_num_heads:
        details.append(f"h={lrid_num_heads}")
    detail_text = " ".join(details)
    second = repo_name if not detail_text else f"{repo_name} ({detail_text})"
    return f"{group_name}\n{textwrap.shorten(second, width=36, placeholder='...')}"


def plot_matrix(result: Any, normalization: str) -> np.ndarray:
    matrix = np.abs(result.query_matrix)
    if normalization == "dim":
        return matrix / max(float(matrix.shape[1]), 1.0)
    if normalization == "sqrt_dim":
        return matrix / math.sqrt(max(float(matrix.shape[1]), 1.0))
    if normalization == "row_rms":
        denom = np.sqrt(np.mean(np.square(result.query_matrix), axis=1, keepdims=True))
        return matrix / np.maximum(denom, 1e-12)
    return matrix


def normalization_label(normalization: str) -> str:
    if normalization == "dim":
        return "|Static Query| / D"
    if normalization == "sqrt_dim":
        return "|Static Query| / sqrt(D)"
    if normalization == "row_rms":
        return "|Static Query| / RMS(query at read site)"
    return "|Static Query|"


def plot_title(normalization: str) -> str:
    if normalization == "dim":
        return "Jonnester Hugging Face Models with Static AttnRes Queries Normalized by Query Dimension"
    if normalization == "sqrt_dim":
        return "Jonnester Hugging Face Models with Static AttnRes Queries Normalized by sqrt(Query Dimension)"
    if normalization == "row_rms":
        return "Jonnester Hugging Face Models with Static AttnRes Queries Normalized by Per-Site RMS"
    return "Jonnester Hugging Face Models with Static AttnRes Queries"


def plot_results(
    results: list[Any],
    pdf_path: Path,
    png_path: Path,
    dpi: int,
    cols: int,
    normalization: str,
    scale_mode: str,
    scale_percentile: float,
) -> None:
    matrices = [
        plot_matrix(result, normalization)
        for result in results
        if result.query_matrix is not None
    ]
    if not matrices:
        raise RuntimeError("No query matrices to plot")

    vmax = None
    if scale_mode == "global":
        all_values = np.concatenate([matrix.ravel() for matrix in matrices])
        vmax = float(np.percentile(all_values, scale_percentile)) if all_values.size else 1.0
        vmax = max(vmax, 1e-12)

    n = len(results)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    fig_w = 5.25 * cols + 0.85
    fig_h = 3.35 * rows + 0.65

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    last_image = None
    for ax, result in zip(axes.ravel(), results):
        matrix = plot_matrix(result, normalization)
        panel_vmax = vmax
        if scale_mode == "per_model":
            panel_vmax = float(np.percentile(matrix.ravel(), scale_percentile)) if matrix.size else 1.0
            panel_vmax = max(panel_vmax, 1e-12)
        last_image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=panel_vmax,
        )
        ax.set_title(title_for(result), fontweight="bold", pad=4)
        ax.set_xlabel("Query Dimension")
        ax.set_ylabel("Residual Read Site")
        for boundary in range(1, matrix.shape[0], 2):
            ax.axhline(boundary - 0.5, color="white", linewidth=0.22, alpha=0.20)
        ax.tick_params(axis="both", colors="#555555", length=2.5, width=0.7)
        if scale_mode == "per_model":
            cbar = fig.colorbar(last_image, ax=ax, fraction=0.045, pad=0.012)
            cbar.ax.tick_params(labelsize=5.0, length=1.6, width=0.5)
            cbar.ax.yaxis.offsetText.set_fontsize(5.0)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if last_image is not None and scale_mode == "global":
        fig.colorbar(
            last_image,
            ax=axes.ravel().tolist(),
            shrink=0.74,
            pad=0.012,
            label=f"{normalization_label(normalization)}, shared {scale_percentile:g}th pct. scale",
        )
    scale_note = "shared scale" if scale_mode == "global" else f"per-model {scale_percentile:g}th pct. scales"
    fig.suptitle(f"{plot_title(normalization)} ({scale_note})", fontsize=14, fontweight="bold")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=min(dpi, 180))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = None
    if args.model_list_file is not None:
        all_repo_ids = [
            line.strip()
            for line in args.model_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        api = HfApi(token=args.hf_token)
        all_repo_ids = list_author_models(api, args.author, args.hf_token)
    if args.max_models is not None:
        all_repo_ids = all_repo_ids[: args.max_models]
    print(f"Found {len(all_repo_ids)} public models under {args.author}", flush=True)

    candidate_substring = args.candidate_substring or ""
    filtered_repo_ids = [
        repo_id
        for repo_id in all_repo_ids
        if not candidate_substring or candidate_substring in repo_id
    ]
    filtered_repo_ids = sorted(
        filtered_repo_ids,
        key=lambda repo_id: (
            not model_result_path(args.existing_results_dir, repo_id).exists(),
            repo_id.lower(),
        ),
    )
    results: list[Any] = []
    skipped: list[dict[str, Any]] = []
    cleanup: list[dict[str, Any]] = []

    temp_cache = None
    download_cache_dir = args.cache_dir
    try:
        if download_cache_dir is None and not args.local_files_only and not args.cache_only:
            temp_cache = tempfile.TemporaryDirectory(prefix="jonnester_static_queries_hf_")
            download_cache_dir = temp_cache.name
            print(f"Using temporary Hugging Face cache: {download_cache_dir}", flush=True)

        for repo_id in all_repo_ids:
            if candidate_substring and candidate_substring not in repo_id:
                skipped.append({"repo_id": repo_id, "stage": "filter", "reason": f"id does not contain {candidate_substring!r}"})
        total_candidates = len(filtered_repo_ids)

        for idx, repo_id in enumerate(filtered_repo_ids, start=1):
            print(f"[{idx}/{total_candidates}] {repo_id}", flush=True)
            cached, cached_path = load_cached_result(repo_id, args.existing_results_dir, args.output_dir)
            if cached is not None:
                if cached.query_matrix is None:
                    skipped.append({"repo_id": repo_id, "stage": "cache", "reason": "cached result has no query matrix"})
                    print("  skipped cached result without query matrix", flush=True)
                else:
                    results.append(cached)
                print(f"  reused cached query matrix: {cached_path}", flush=True)
                continue
            if args.cache_only:
                skipped.append({"repo_id": repo_id, "stage": "cache", "reason": "no cached query matrix found"})
                print("  skipped: no cached query matrix found", flush=True)
                continue
            if api is None:
                api = HfApi(token=args.hf_token)

            checkpoint_path = None
            loaded = None
            try:
                print("  listing files", flush=True)
                files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=args.hf_token)
                checkpoint_filename = choose_checkpoint_file(files)
                if checkpoint_filename is None:
                    skipped.append({"repo_id": repo_id, "stage": "files", "reason": "no preferred checkpoint file found"})
                    print("  skipped: no preferred checkpoint file found", flush=True)
                    continue
                print(f"  downloading {checkpoint_filename}", flush=True)
                checkpoint_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=checkpoint_filename,
                    repo_type="model",
                    cache_dir=download_cache_dir,
                    token=args.hf_token,
                    local_files_only=args.local_files_only,
                )
                print("  loading checkpoint and extracting static query matrix", flush=True)
                loaded = load_model_from_checkpoint(repo_id, checkpoint_path, torch.device("cpu"), torch.float32)
                result = lightweight_result(repo_id, checkpoint_path, loaded)
                if result.query_matrix is None:
                    skipped.append({"repo_id": repo_id, "stage": "load", "reason": "model has no AttnRes static query matrix"})
                    print("  skipped: no AttnRes static query matrix", flush=True)
                    continue
                results.append(result)
                save_lightweight_result(result, args.output_dir)
                print(f"  included: query matrix {tuple(result.query_matrix.shape)}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep surveying the owner namespace.
                skipped.append(
                    {
                        "repo_id": repo_id,
                        "stage": "load",
                        "checkpoint_path": checkpoint_path,
                        "reason": repr(exc),
                    }
                )
                print(f"  skipped after error: {exc!r}", flush=True)
            finally:
                if loaded is not None:
                    del loaded
                gc.collect()
                if not args.keep_checkpoints:
                    deleted_paths = delete_downloaded_checkpoint(checkpoint_path)
                    if deleted_paths:
                        cleanup.append(
                            {
                                "repo_id": repo_id,
                                "paths": deleted_paths,
                            }
                        )
                        print(f"  deleted {len(deleted_paths)} checkpoint path(s)", flush=True)
    finally:
        if temp_cache is not None:
            temp_cache.cleanup()

    results = sorted(results, key=model_sort_key)
    suffix = "" if args.normalization == "none" else f"_{args.normalization}_normalized"
    if args.scale_mode == "per_model":
        suffix += "_per_model_scale"
    manifest = {
        "author": args.author,
        "candidate_substring": candidate_substring,
        "all_repo_ids": all_repo_ids,
        "included_repo_ids": [result.repo_id for result in results],
        "skipped": skipped,
        "cleanup": cleanup,
        "included_count": len(results),
        "skipped_count": len(skipped),
        "cleanup_count": len(cleanup),
        "normalization": args.normalization,
        "scale_mode": args.scale_mode,
        "scale_percentile": args.scale_percentile,
    }
    manifest_path = args.output_dir / f"jonnester_static_query_manifest{suffix}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "jonnester_static_query_models.txt").write_text(
        "\n".join(result.repo_id for result in results) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "jonnester_all_models.txt").write_text(
        "\n".join(all_repo_ids) + "\n",
        encoding="utf-8",
    )

    pdf_path = args.output_dir / f"jonnester_all_static_queries{suffix}.pdf"
    png_path = args.output_dir / f"jonnester_all_static_queries{suffix}.png"
    plot_results(
        results,
        pdf_path,
        png_path,
        args.dpi,
        args.cols,
        args.normalization,
        args.scale_mode,
        args.scale_percentile,
    )
    print(f"Found {len(all_repo_ids)} public models under {args.author}")
    print(f"Included {len(results)} models with static query matrices")
    print(f"Skipped {len(skipped)} models")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
