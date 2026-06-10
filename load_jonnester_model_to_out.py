import argparse
import os
import re
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional


def require_package(import_name, package_name=None):
    try:
        return __import__(import_name)
    except ModuleNotFoundError as exc:
        if exc.name != import_name:
            raise
        package_name = package_name or import_name
        print(
            f"Missing dependency: {package_name}\n\n"
            f"Install it with:\n"
            f"  {sys.executable} -m pip install {package_name}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


torch = require_package("torch")
huggingface_hub = require_package("huggingface_hub", "huggingface-hub")
HfApi = huggingface_hub.HfApi
hf_hub_download = huggingface_hub.hf_hub_download

from model import ModelConfig, OBPM


DEFAULT_REPO_ID = "Jonnester/LR-AttnRes-n16"
PREFERRED_CHECKPOINT_FILES = (
    "final_model.pt",
    "checkpoint.pt",
    "ckpt.pt",
    "model.pt",
    "pytorch_model.bin",
)


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

    sample = "\n".join(f"  - {f}" for f in sorted(files)[:60])
    raise RuntimeError(
        "Could not choose a checkpoint file automatically. "
        "Pass --filename explicitly.\n"
        f"First files in repo:\n{sample}"
    )


def resolve_checkpoint_path(
    repo_id: str,
    filename: Optional[str],
    revision: Optional[str],
    cache_dir: Optional[str],
    token: Optional[str],
    local_files_only: bool,
) -> tuple[str, str]:
    if os.path.isfile(repo_id):
        return repo_id, os.path.basename(repo_id)

    if filename is None:
        files = HfApi(token=token).list_repo_files(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
        )
        filename = choose_checkpoint_file(files)

    print(f"Downloading {repo_id}:{filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    return path, filename


def normalize_model_args(model_args):
    if isinstance(model_args, dict):
        return asdict(ModelConfig(**model_args))
    if isinstance(model_args, ModelConfig):
        return asdict(model_args)
    if is_dataclass(model_args):
        return asdict(model_args)
    raise TypeError(f"Unsupported checkpoint model_args type: {type(model_args)!r}")


def strip_compiled_prefix(state_dict: dict) -> dict:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    print(f"Detected compiled model checkpoint. Stripping '{prefix}' from state dict keys.")
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def default_output_name(source_filename: str, checkpoint: dict) -> str:
    step = checkpoint.get("step")
    if isinstance(step, int):
        return f"ckpt_step:{step}.pt"

    basename = os.path.basename(source_filename)
    if basename.endswith((".pt", ".pth", ".bin")):
        return basename
    return "ckpt_from_hf.pt"


def load_and_save_checkpoint(args):
    checkpoint_path, source_filename = resolve_checkpoint_path(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=args.token,
        local_files_only=args.local_files_only,
    )
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise TypeError(
            "Expected a repo checkpoint dictionary with a 'model' state dict. "
            "This script saves LR-AttnRes training checkpoints, not generic HF models."
        )

    model_args = normalize_model_args(checkpoint.get("model_args", {}))
    model_config = ModelConfig(**model_args)
    state_dict = strip_compiled_prefix(checkpoint["model"])

    print("Instantiating OBPM and loading state dict.")
    model = OBPM(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    output_name = args.output_name or default_output_name(source_filename, checkpoint)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    payload = dict(checkpoint)
    payload["model"] = model.state_dict()
    payload["model_args"] = asdict(model_config)

    if args.copy_only:
        print("Checkpoint loaded successfully; copying original file without rewriting tensors.")
        shutil.copy2(checkpoint_path, output_path)
    else:
        print(f"Saving loaded checkpoint to: {output_path}")
        torch.save(payload, output_path)

    print(f"Done: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download an LR-AttnRes checkpoint from a Hugging Face model repo, "
            "load it with OBPM to validate it, and save it into out/."
        )
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repo id or local checkpoint path. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument("--filename", default=None, help="Checkpoint filename inside the repo.")
    parser.add_argument("--revision", default=None, help="Optional HF revision, branch, or commit.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--token", default=None, help="HF token for private repos, if needed.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only files already present in the Hugging Face cache.",
    )
    parser.add_argument("--out-dir", default="out", help="Directory to save into. Default: out")
    parser.add_argument("--output-name", default=None, help="Output checkpoint filename.")
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Validate by loading, then copy the original checkpoint instead of rewriting it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_and_save_checkpoint(parse_args())
