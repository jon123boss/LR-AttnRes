#!/usr/bin/env python3
"""Download the exact 131+1 shard dataset used by the 0.5B sweep."""

from huggingface_hub import snapshot_download


REPO_ID = "Jonnester/Ultra-FineWeb-en-20B-gpt4"
REVISION = "2d102ffbc415103c82705a227afba5dad5b9d217"
LOCAL_DIR = "/dev/shm/ultrafineweb20B_gpt4"


def main() -> None:
    files = ["finewebedu_val_000000.bin"]
    files.extend(f"finewebedu_train_{index:06d}.bin" for index in range(1, 132))
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=LOCAL_DIR,
        allow_patterns=files,
        max_workers=8,
    )


if __name__ == "__main__":
    main()
