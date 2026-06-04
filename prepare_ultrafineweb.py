# prepare_ultrafineweb.py
#
# Stream Ultra-FineWeb-en, tokenize it with the GPT-4 tokenizer, and write
# binary token shards compatible with DocumentPackingDataset.

import argparse
import json
import os
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from tokenizer_utils import GPT4_TOKENIZER_MODEL, tokenizer_metadata


DEFAULT_SOURCE_REPO = "openbmb/Ultra-FineWeb"
DEFAULT_SOURCE_SPLIT = "en"
DEFAULT_TEXT_COLUMN = "content"
DEFAULT_OUTPUT_DIR = "ultrafineweb20B_gpt4"
DEFAULT_TOTAL_TOKENS = 20_000_000_000
DEFAULT_VAL_TOKENS = 100_000_000
DEFAULT_SHARD_TOKENS = 100_000_000
DEFAULT_DTYPE = "uint32"
DEFAULT_HF_REPO_ID = "jon123boss/Ultra-FineWeb-en-20B-gpt4"


def parse_int(value: str) -> int:
    multipliers = {
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
    }
    value = value.strip().lower().replace("_", "")
    if value[-1:] in multipliers:
        return int(float(value[:-1]) * multipliers[value[-1]])
    return int(value)


class TokenShardWriter:
    def __init__(
        self,
        output_dir: str,
        split: str,
        shard_tokens: int,
        dtype: np.dtype,
        start_index: int,
    ):
        self.output_dir = output_dir
        self.split = split
        self.shard_tokens = int(shard_tokens)
        self.dtype = np.dtype(dtype)
        self.next_index = int(start_index)
        self.current_path = None
        self.current_file = None
        self.current_tokens = 0
        self.total_tokens = 0
        self.files: List[Dict[str, int]] = []

    def _open_next(self):
        os.makedirs(self.output_dir, exist_ok=True)
        name = f"finewebedu_{self.split}_{self.next_index:06d}.bin"
        self.current_path = os.path.join(self.output_dir, name)
        self.current_file = open(self.current_path, "wb")
        self.current_tokens = 0
        self.next_index += 1

    def _close_current(self):
        if self.current_file is None:
            return
        self.current_file.close()
        self.files.append(
            {
                "filename": os.path.basename(self.current_path),
                "tokens": int(self.current_tokens),
                "bytes": int(os.path.getsize(self.current_path)),
            }
        )
        self.current_file = None
        self.current_path = None
        self.current_tokens = 0

    def write(self, token_ids: List[int]):
        pos = 0
        n_tokens = len(token_ids)

        while pos < n_tokens:
            if self.current_file is None:
                self._open_next()

            room = self.shard_tokens - self.current_tokens
            take = min(room, n_tokens - pos)
            chunk = np.asarray(token_ids[pos:pos + take], dtype=self.dtype)
            self.current_file.write(chunk.tobytes(order="C"))
            self.current_tokens += int(take)
            self.total_tokens += int(take)
            pos += int(take)

            if self.current_tokens == self.shard_tokens:
                self._close_current()

    def close(self):
        self._close_current()


def iter_texts(args) -> Iterable[str]:
    from datasets import load_dataset

    dataset = load_dataset(
        args.source_repo,
        split=args.source_split,
        streaming=True,
        trust_remote_code=args.trust_remote_code,
    )

    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    for row in dataset:
        text = row.get(args.text_column)
        if isinstance(text, str) and text:
            yield text


def encode_text(enc, text: str, eot_token: int) -> List[int]:
    if hasattr(enc, "encode_ordinary"):
        ids = enc.encode_ordinary(text)
    else:
        ids = enc.encode(text, disallowed_special=())
    ids.append(eot_token)
    return ids


def write_readme(output_dir: str, metadata: Dict):
    readme = f"""# Ultra-FineWeb-en 20B GPT-4 Token Shards

Binary token shards for `LR-AttnRes`.

- Source dataset: `{metadata["source_repo"]}`, split `{metadata["source_split"]}`
- Text column: `{metadata["text_column"]}`
- Tokenizer model: `{metadata["tokenizer_model"]}`
- Encoding: `{metadata["encoding_name"]}`
- Vocab size: `{metadata["vocab_size"]}`
- Document separator / EOT token: `{metadata["eot_token"]}`
- Dtype: `{metadata["dtype"]}`

The shard filenames intentionally use the existing dataloader convention:

- `finewebedu_val_*.bin`
- `finewebedu_train_*.bin`

Download with:

```bash
python prepdata.py --repo-id <your-hf-username>/Ultra-FineWeb-en-20B-gpt4
```
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def save_metadata(output_dir: str, metadata: Dict):
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")


def load_metadata(output_dir: str) -> Dict:
    path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist. Run tokenization first, or point --output-dir "
            "at a completed prepared shard directory."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_prepared_output(output_dir: str, metadata: Dict):
    missing = []
    for split_files in metadata.get("files", {}).values():
        for item in split_files:
            filename = item.get("filename")
            if filename and not os.path.exists(os.path.join(output_dir, filename)):
                missing.append(filename)

    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview += f", ... ({len(missing)} missing total)"
        raise FileNotFoundError(f"Prepared shard files are missing: {preview}")


def upload_folder(args):
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.errors import HfHubHTTPError

    try:
        create_repo(
            args.hf_repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        api = HfApi()
        if hasattr(api, "upload_large_folder"):
            api.upload_large_folder(
                folder_path=args.output_dir,
                repo_id=args.hf_repo_id,
                repo_type="dataset",
            )
        else:
            print(
                "Your huggingface_hub version does not have upload_large_folder; "
                "falling back to upload_folder. Upgrade with "
                "`python -m pip install -U huggingface_hub` for better resume behavior."
            )
            api.upload_folder(
                folder_path=args.output_dir,
                repo_id=args.hf_repo_id,
                repo_type="dataset",
                commit_message=args.commit_message,
            )
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Could not create or upload to dataset repo {args.hf_repo_id!r}. "
            "Check that you ran `huggingface-cli login`, that your token has "
            "write permission, and that the namespace before `/` is your HF "
            "username or an organization you can write to."
        ) from exc


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize Ultra-FineWeb-en into GPT-4/cl100k_base uint32 shards "
            "compatible with dataloader.py."
        )
    )
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--source-split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer-model", default=GPT4_TOKENIZER_MODEL)
    parser.add_argument("--total-tokens", type=parse_int, default=DEFAULT_TOTAL_TOKENS)
    parser.add_argument("--val-tokens", type=parse_int, default=DEFAULT_VAL_TOKENS)
    parser.add_argument("--train-tokens", type=parse_int, default=None)
    parser.add_argument("--shard-tokens", type=parse_int, default=DEFAULT_SHARD_TOKENS)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=("uint32", "int64"))
    parser.add_argument("--shuffle-buffer", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Upload an already prepared output directory without tokenizing again.",
    )
    parser.add_argument("--hf-repo-id", default=os.environ.get("LR_DATA_REPO_ID", DEFAULT_HF_REPO_ID))
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="Add GPT-4 tokenized Ultra-FineWeb-en shards",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.upload_only:
        metadata = load_metadata(args.output_dir)
        validate_prepared_output(args.output_dir, metadata)
        upload_folder(args)
        print(f"Uploaded to https://huggingface.co/datasets/{args.hf_repo_id}")
        return

    if args.val_tokens < 0:
        raise ValueError("--val-tokens must be >= 0")
    if args.train_tokens is None:
        args.train_tokens = args.total_tokens - args.val_tokens
    if args.train_tokens < 1:
        raise ValueError("--train-tokens must be >= 1")
    if args.shard_tokens < 1:
        raise ValueError("--shard-tokens must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)

    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(args.tokenizer_model)
    except KeyError:
        if args.tokenizer_model == GPT4_TOKENIZER_MODEL:
            enc = tiktoken.get_encoding("cl100k_base")
        else:
            raise

    dtype = np.dtype(args.dtype)
    if np.iinfo(dtype).max < enc.n_vocab - 1:
        raise ValueError(
            f"dtype={dtype} cannot represent tokenizer vocab size {enc.n_vocab}"
        )

    eot_token = int(enc.eot_token)
    token_meta = tokenizer_metadata(args.tokenizer_model)
    if token_meta["encoding_name"] != enc.name:
        raise RuntimeError("tokenizer helper and preprocessing tokenizer disagree")

    targets = [
        ("val", args.val_tokens, 0),
        ("train", args.train_tokens, 1),
    ]
    targets = [target for target in targets if target[1] > 0]
    writers = {
        split: TokenShardWriter(args.output_dir, split, args.shard_tokens, dtype, start_idx)
        for split, _, start_idx in targets
    }

    target_idx = 0
    target_written = 0
    docs_seen = 0
    skipped_empty = 0
    start_time = time.time()

    try:
        for text in iter_texts(args):
            if target_idx >= len(targets):
                break

            ids = encode_text(enc, text, eot_token)
            if not ids:
                skipped_empty += 1
                continue

            docs_seen += 1
            pos = 0
            while pos < len(ids) and target_idx < len(targets):
                split, target_tokens, _ = targets[target_idx]
                remaining = target_tokens - target_written
                take = min(remaining, len(ids) - pos)
                writers[split].write(ids[pos:pos + take])
                pos += take
                target_written += take

                if target_written == target_tokens:
                    writers[split].close()
                    print(f"Completed {split}: {target_written:,} tokens")
                    target_idx += 1
                    target_written = 0

            if docs_seen % 100_000 == 0:
                elapsed = max(1e-9, time.time() - start_time)
                written = sum(writer.total_tokens for writer in writers.values())
                print(
                    f"docs={docs_seen:,} tokens={written:,} "
                    f"tok/s={written / elapsed:,.0f}"
                )
    finally:
        for writer in writers.values():
            writer.close()

    total_written = sum(writer.total_tokens for writer in writers.values())
    requested = sum(target[1] for target in targets)
    if total_written < requested:
        raise RuntimeError(
            f"Dataset stream ended after {total_written:,} tokens; requested {requested:,}"
        )

    metadata = {
        **token_meta,
        "source_repo": args.source_repo,
        "source_split": args.source_split,
        "text_column": args.text_column,
        "total_tokens": int(total_written),
        "requested_total_tokens": int(args.total_tokens),
        "train_tokens": int(writers["train"].total_tokens),
        "val_tokens": int(writers["val"].total_tokens) if "val" in writers else 0,
        "shard_tokens": int(args.shard_tokens),
        "dtype": str(dtype),
        "doc_separator_token": int(eot_token),
        "docs_seen": int(docs_seen),
        "skipped_empty": int(skipped_empty),
        "shuffle_buffer": int(args.shuffle_buffer),
        "seed": int(args.seed),
        "files": {
            split: writer.files
            for split, writer in writers.items()
        },
    }
    save_metadata(args.output_dir, metadata)
    write_readme(args.output_dir, metadata)

    print(f"Wrote {total_written:,} tokens to {args.output_dir}")
    print(f"Train shards: {len(writers['train'].files)}")
    if "val" in writers:
        print(f"Val shards: {len(writers['val'].files)}")
    print(f"Tokenizer: {metadata['tokenizer_model']} / {metadata['encoding_name']}")
    print(f"Vocab size: {metadata['vocab_size']}")
    print(f"EOT/doc separator: {metadata['eot_token']}")

    if args.upload:
        upload_folder(args)
        print(f"Uploaded to https://huggingface.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
