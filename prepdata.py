# prepdata.py
#
# Download GPT-4-tokenized Ultra-FineWeb-en shards produced by
# prepare_ultrafineweb.py.

import argparse
import json
import os
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "Jonnester/Ultra-FineWeb-en-20B-gpt4"
DEFAULT_LOCAL_DIR = "ultrafineweb20B_gpt4"
DEFAULT_NUM_TRAIN_SHARDS = 10
DEFAULT_NUM_VAL_SHARDS = 1


def download_file(repo_id: str, filename: str, local_dir: str, revision: Optional[str]):
    path = os.path.join(local_dir, filename)
    if os.path.exists(path):
        return path

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )


def try_download_metadata(repo_id: str, local_dir: str, revision: Optional[str]) -> Optional[Dict]:
    try:
        path = download_file(repo_id, "metadata.json", local_dir, revision)
    except Exception as exc:
        print(f"metadata.json unavailable; using shard-count defaults ({type(exc).__name__}: {exc})")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filenames_from_metadata(metadata: Dict) -> List[str]:
    files = []
    for split in ("val", "train"):
        for item in metadata.get("files", {}).get(split, []):
            filename = item.get("filename")
            if filename:
                files.append(filename)
    return files


def default_filenames(num_train_shards: int, num_val_shards: int) -> List[str]:
    files = [
        f"finewebedu_val_{idx:06d}.bin"
        for idx in range(num_val_shards)
    ]
    files.extend(
        f"finewebedu_train_{idx:06d}.bin"
        for idx in range(1, num_train_shards + 1)
    )
    return files


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Download Ultra-FineWeb-en GPT-4 token shards for LR-AttnRes."
    )
    parser.add_argument(
        "legacy_num_train_shards",
        nargs="?",
        type=int,
        help="Optional backwards-compatible positional train shard count.",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("LR_DATA_REPO_ID", DEFAULT_REPO_ID),
        help="Hugging Face dataset repo containing prepared .bin shards.",
    )
    parser.add_argument(
        "--local-dir",
        default=os.environ.get("LR_DATA_DIR", DEFAULT_LOCAL_DIR),
        help="Where to store downloaded shards.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--num-train-shards", type=int, default=None)
    parser.add_argument("--num-val-shards", type=int, default=DEFAULT_NUM_VAL_SHARDS)
    parser.add_argument(
        "--ignore-metadata",
        action="store_true",
        help="Download files from the explicit shard counts instead of metadata.json.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.makedirs(args.local_dir, exist_ok=True)

    metadata = None
    if not args.ignore_metadata:
        metadata = try_download_metadata(args.repo_id, args.local_dir, args.revision)

    filenames = filenames_from_metadata(metadata) if metadata else []
    if not filenames:
        num_train_shards = (
            args.legacy_num_train_shards
            if args.legacy_num_train_shards is not None
            else args.num_train_shards
        )
        if num_train_shards is None:
            num_train_shards = DEFAULT_NUM_TRAIN_SHARDS
        filenames = default_filenames(num_train_shards, args.num_val_shards)

    print(f"Repo: {args.repo_id}")
    print(f"Local dir: {args.local_dir}")
    print(f"Files: {len(filenames)}")

    for i, filename in enumerate(filenames, 1):
        print(f"[{i}/{len(filenames)}] {filename}")
        download_file(args.repo_id, filename, args.local_dir, args.revision)

    if metadata:
        vocab_size = metadata.get("vocab_size")
        eot_token = metadata.get("eot_token")
        dtype = metadata.get("dtype")
        print(
            f"Metadata: tokenizer={metadata.get('tokenizer_model')} "
            f"encoding={metadata.get('encoding_name')} "
            f"vocab_size={vocab_size} eot={eot_token} dtype={dtype}"
        )


if __name__ == "__main__":
    main()
