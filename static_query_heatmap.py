import argparse
from dataclasses import dataclass
import gc
import importlib
import json
import math
from pathlib import Path
import re
import sys


DEFAULT_REPOS = (
    "Jonnester/LR-AttnRes-Full",
    "Jonnester/LR-AttnRes-Full-32",
)


def require_module(import_name, package_name=None):
    try:
        return importlib.import_module(import_name)
    except ModuleNotFoundError as exc:
        root_name = import_name.split(".", 1)[0]
        if exc.name not in {import_name, root_name}:
            raise
        package_name = package_name or root_name
        print(
            f"Missing dependency: {package_name}\n\n"
            f"Install the dependencies for this Python with:\n"
            f"  {sys.executable} -m pip install torch huggingface_hub matplotlib numpy\n",
            file=sys.stderr,
        )
        if root_name == "torch" and sys.version_info >= (3, 14):
            print(
                "Note: you are using Python 3.14. If pip cannot find a torch wheel, "
                "run this script with a Python version supported by PyTorch, such as "
                "Python 3.12 or 3.13.",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc


np = require_module("numpy")
torch = require_module("torch")
snapshot_download = require_module("huggingface_hub").snapshot_download


@dataclass
class QueryAnalysis:
    repo_id: str
    model_dir: Path
    checkpoint_path: Path
    config: dict
    raw_queries: object
    effective_queries: object
    flat_effective_queries: object
    gates: object | None


def clean_state_key(key):
    for prefix in ("_orig_mod.", "module."):
        if key.startswith(prefix):
            return clean_state_key(key[len(prefix) :])
    return key


def load_json_if_present(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def checkpoint_candidates(model_dir):
    suffixes = {".pt", ".pth", ".bin", ".ckpt", ".safetensors"}
    candidates = [path for path in model_dir.rglob("*") if path.is_file() and path.suffix in suffixes]

    def score(path):
        name = path.name.lower()
        likely_model = any(token in name for token in ("ckpt", "checkpoint", "model", "pytorch_model"))
        likely_optimizer = any(token in name for token in ("optimizer", "optim", "scheduler"))
        return (
            1 if likely_optimizer else 0,
            0 if likely_model else 1,
            -path.stat().st_size,
            str(path),
        )

    return sorted(candidates, key=score)


def state_dict_from_loaded_object(loaded):
    if isinstance(loaded, dict):
        for key in ("model", "state_dict", "module"):
            value = loaded.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(key, str) for key in loaded.keys()):
            return loaded
    raise TypeError("Could not find a state dict in the loaded checkpoint.")


def load_checkpoint_or_state_dict(path):
    if path.suffix == ".safetensors":
        safetensors_torch = require_module("safetensors.torch", "safetensors")
        return {}, safetensors_torch.load_file(str(path), device="cpu")

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    return loaded if isinstance(loaded, dict) else {}, state_dict_from_loaded_object(loaded)


def extract_indexed_tensors(state_dict, base_name):
    pattern = re.compile(rf"^{re.escape(base_name)}\.(\d+)$")
    tensors = {}
    for key, value in state_dict.items():
        match = pattern.match(clean_state_key(key))
        if match is None:
            continue
        tensors[int(match.group(1))] = value.detach().cpu().float()
    if not tensors:
        return None

    max_idx = max(tensors)
    missing = [idx for idx in range(max_idx + 1) if idx not in tensors]
    if missing:
        raise ValueError(f"Missing {base_name} entries: {missing}")
    return torch.stack([tensors[idx] for idx in range(max_idx + 1)], dim=0).numpy()


def rms_norm_last_dim(array):
    eps = np.finfo(np.float32).eps
    return array / np.sqrt(np.mean(array * array, axis=-1, keepdims=True) + eps)


def merged_config(checkpoint, config_json):
    model_args = checkpoint.get("model_args", {})
    if not isinstance(model_args, dict):
        model_args = getattr(model_args, "__dict__", {})
    config = dict(config_json)
    config.update(model_args)
    train_config = checkpoint.get("config", {})
    if isinstance(train_config, dict):
        for key in (
            "n_layer",
            "lrid_rank",
            "lrid_num_heads",
            "attn_res_query_norm",
            "lrid_logit_scale",
            "attnres_type",
            "use_lrid",
            "lrid_input_dependent_query",
        ):
            config.setdefault(key, train_config.get(key))
    return {key: value for key, value in config.items() if value is not None}


def load_query_analysis(repo_id, local_root):
    model_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_root / repo_id.split("/")[-1]),
        )
    )
    config_json = load_json_if_present(model_dir / "config.json")

    for candidate in checkpoint_candidates(model_dir):
        checkpoint, state_dict = load_checkpoint_or_state_dict(candidate)
        queries = extract_indexed_tensors(state_dict, "transformer.lrid_queries")
        gates = extract_indexed_tensors(state_dict, "transformer.lrid_query_gates")
        if queries is None:
            del checkpoint, state_dict
            gc.collect()
            continue

        config = merged_config(checkpoint, config_json)
        effective = rms_norm_last_dim(queries) if config.get("attn_res_query_norm", False) else queries.copy()
        flat_effective = effective.reshape(effective.shape[0], -1)

        del checkpoint, state_dict
        gc.collect()
        return QueryAnalysis(
            repo_id=repo_id,
            model_dir=model_dir,
            checkpoint_path=candidate,
            config=config,
            raw_queries=queries,
            effective_queries=effective,
            flat_effective_queries=flat_effective,
            gates=gates,
        )

    raise FileNotFoundError(f"No checkpoint with transformer.lrid_queries.* found in {model_dir}")


def layer_site_labels(num_sites):
    labels = []
    for site_idx in range(num_sites):
        layer_idx = site_idx // 2
        sublayer = "attn" if site_idx % 2 == 0 else "mlp"
        labels.append(f"L{layer_idx:02d} {sublayer}")
    return labels


def participation_ratio(row):
    energy = row * row
    denom = np.sum(energy * energy)
    if denom <= 0:
        return 0.0
    return float((np.sum(energy) ** 2) / denom)


def hoyer_sparsity(row):
    row_abs = np.abs(row)
    l2 = np.linalg.norm(row_abs)
    if l2 <= 0:
        return 1.0
    n = row_abs.size
    if n <= 1:
        return 0.0
    return float((math.sqrt(n) - np.sum(row_abs) / l2) / (math.sqrt(n) - 1.0))


def dims_for_energy(row, fraction):
    energy = np.sort(row * row)[::-1]
    total = np.sum(energy)
    if total <= 0:
        return 0
    cumulative = np.cumsum(energy)
    return int(np.searchsorted(cumulative, fraction * total, side="left") + 1)


def summarize_analysis(analysis, rel_thresholds):
    matrix = analysis.flat_effective_queries
    abs_matrix = np.abs(matrix)
    num_sites, rank = matrix.shape
    max_abs = float(abs_matrix.max()) if abs_matrix.size else 0.0
    exact_zero = float(np.mean(abs_matrix == 0.0))

    row_max = abs_matrix.max(axis=1, keepdims=True)
    row_max[row_max == 0.0] = 1.0
    used_by_threshold = {
        threshold: np.mean(abs_matrix > threshold * row_max, axis=1)
        for threshold in rel_thresholds
    }

    participation = np.array([participation_ratio(row) for row in matrix])
    hoyer = np.array([hoyer_sparsity(row) for row in matrix])
    energy_90 = np.array([dims_for_energy(row, 0.90) for row in matrix])
    energy_95 = np.array([dims_for_energy(row, 0.95) for row in matrix])
    energy_99 = np.array([dims_for_energy(row, 0.99) for row in matrix])

    short_name = analysis.repo_id.split("/")[-1]
    print(f"\n=== {analysis.repo_id} ===")
    print(f"downloaded: {analysis.model_dir}")
    print(f"checkpoint: {analysis.checkpoint_path}")
    print(
        "config: "
        f"attnres_type={analysis.config.get('attnres_type', 'unknown')} | "
        f"n_layer={analysis.config.get('n_layer', num_sites // 2)} | "
        f"sites={num_sites} | "
        f"heads={analysis.effective_queries.shape[1]} | "
        f"rank={rank} | "
        f"attn_res_query_norm={analysis.config.get('attn_res_query_norm', False)} | "
        f"lrid_logit_scale={analysis.config.get('lrid_logit_scale', 'unknown')}"
    )
    print("effective static query magnitude:")
    print(f"  max |q|:        {max_abs:.6g}")
    print(f"  mean |q|:       {float(abs_matrix.mean()):.6g}")
    print(f"  median |q|:     {float(np.median(abs_matrix)):.6g}")
    print(f"  exact zeros:    {exact_zero:.2%}")

    print("relative used dimensions per site, using |q| > threshold * site_max(|q|):")
    for threshold, fractions in used_by_threshold.items():
        print(
            f"  > {threshold:g}x site max: "
            f"mean {float(fractions.mean() * rank):.1f}/{rank} dims "
            f"({float(fractions.mean()):.2%}), "
            f"min {float(fractions.min() * rank):.1f}, "
            f"max {float(fractions.max() * rank):.1f}"
        )

    print("energy concentration:")
    print(
        f"  L2 participation: mean {float(participation.mean()):.1f}/{rank} dims "
        f"({float(participation.mean() / rank):.2%}), "
        f"median {float(np.median(participation)):.1f}"
    )
    print(
        f"  dims for 90/95/99% L2 energy: "
        f"median {int(np.median(energy_90))}/"
        f"{int(np.median(energy_95))}/"
        f"{int(np.median(energy_99))} of {rank}"
    )
    print(f"  Hoyer sparsity: mean {float(hoyer.mean()):.3f} (0=dense, 1=sparse)")

    if analysis.gates is not None:
        gates_abs = np.abs(analysis.gates)
        print("input-dependent query gates:")
        print(
            f"  mean |gate|={float(gates_abs.mean()):.6g}, "
            f"max |gate|={float(gates_abs.max()):.6g}"
        )
        print("  heatmap shows the static component only; dynamic query use is input-dependent.")

    return {
        "name": short_name,
        "participation_mean": float(participation.mean()),
        "participation_fraction": float(participation.mean() / rank),
        "used_1pct_fraction": float(used_by_threshold.get(0.01, np.array([np.nan])).mean()),
    }


def plot_heatmaps(analyses, output_path, clip_percentile, absolute, show):
    if not show:
        matplotlib = require_module("matplotlib")
        matplotlib.use("Agg")
    plt = require_module("matplotlib.pyplot", "matplotlib")

    matrices = [
        np.abs(analysis.flat_effective_queries) if absolute else analysis.flat_effective_queries
        for analysis in analyses
    ]
    if absolute:
        all_values = np.concatenate([matrix.ravel() for matrix in matrices])
        vmax = float(np.percentile(all_values, clip_percentile))
        vmin = 0.0
        cmap = "magma"
    else:
        all_values = np.concatenate([np.abs(matrix).ravel() for matrix in matrices])
        vmax = float(np.percentile(all_values, clip_percentile))
        vmin = -vmax
        cmap = "RdBu_r"

    if vmax <= 0.0:
        vmax = 1.0
        if not absolute:
            vmin = -1.0

    fig_height = max(5.0, max(matrix.shape[0] for matrix in matrices) * 0.18)
    fig, axes = plt.subplots(
        1,
        len(analyses),
        figsize=(7.2 * len(analyses), fig_height),
        constrained_layout=True,
        squeeze=False,
    )

    image = None
    for axis, analysis, matrix in zip(axes[0], analyses, matrices):
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        short_name = analysis.repo_id.split("/")[-1]
        heads = analysis.effective_queries.shape[1]
        head_dim = analysis.effective_queries.shape[2]
        axis.set_title(f"{short_name}\neffective static query")
        axis.set_xlabel("query dimension")
        axis.set_ylabel("residual site")

        for layer_boundary in range(2, matrix.shape[0], 2):
            axis.axhline(layer_boundary - 0.5, color="black", linewidth=0.35, alpha=0.25)
        if heads > 1:
            for head_boundary in range(head_dim, matrix.shape[1], head_dim):
                axis.axvline(head_boundary - 0.5, color="black", linewidth=0.35, alpha=0.35)

        labels = layer_site_labels(matrix.shape[0])
        tick_stride = max(1, math.ceil(matrix.shape[0] / 32))
        tick_positions = list(range(0, matrix.shape[0], tick_stride))
        axis.set_yticks(tick_positions)
        axis.set_yticklabels([labels[idx] for idx in tick_positions], fontsize=8)

    colorbar_label = "|query|" if absolute else "query value"
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92, label=colorbar_label)
    fig.suptitle(
        "LR AttnRes Static Query Heatmaps\n"
        f"values clipped at the {clip_percentile:g}th percentile for display",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"\nSaved heatmap: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download LR AttnRes checkpoints, plot static LRID query heatmaps, "
            "and print sparsity/usage statistics."
        )
    )
    parser.add_argument("--repos", nargs="+", default=list(DEFAULT_REPOS), help="Hugging Face repo IDs to inspect.")
    parser.add_argument("--local-root", type=Path, default=Path("hf_models"), help="Directory for downloaded snapshots.")
    parser.add_argument("--output", type=Path, default=Path("static_query_heatmap.png"), help="Output PNG path.")
    parser.add_argument(
        "--rel-thresholds",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.10],
        help="Relative per-site magnitude thresholds used for used-dimension stats.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Percentile used to clip heatmap colors for display.",
    )
    parser.add_argument("--absolute", action="store_true", help="Plot absolute query magnitude instead of signed values.")
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window after saving.")
    return parser.parse_args()


def main():
    args = parse_args()
    analyses = [load_query_analysis(repo_id, args.local_root) for repo_id in args.repos]
    summaries = [summarize_analysis(analysis, args.rel_thresholds) for analysis in analyses]
    plot_heatmaps(analyses, args.output, args.clip_percentile, args.absolute, args.show)

    print("\n=== quick comparison ===")
    for summary in summaries:
        print(
            f"{summary['name']}: "
            f"L2 participation {summary['participation_mean']:.1f} dims "
            f"({summary['participation_fraction']:.2%}); "
            f">1% site-max used fraction {summary['used_1pct_fraction']:.2%}"
        )


if __name__ == "__main__":
    main()
