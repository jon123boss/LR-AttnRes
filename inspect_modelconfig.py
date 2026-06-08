from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import pprint
import sys


def require_package(import_name, package_name=None):
    try:
        return __import__(import_name)
    except ModuleNotFoundError as exc:
        if exc.name != import_name:
            raise
        package_name = package_name or import_name
        print(
            f"Missing dependency: {package_name}\n\n"
            f"Install it for this Python with:\n"
            f"  {sys.executable} -m pip install {package_name}\n",
            file=sys.stderr,
        )
        if sys.version_info >= (3, 14) and package_name == "torch":
            print(
                "Note: you are using Python 3.14. If pip cannot find a torch wheel, "
                "run this script with a Python version supported by PyTorch, such as "
                "Python 3.12 or 3.13.",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc


torch = require_package("torch")
snapshot_download = require_package("huggingface_hub").snapshot_download

from model import ModelConfig


REPO_ID = "Jonnester/LR-AttnRes-n16"


def main():
    model_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=Path("hf_models") / REPO_ID.split("/")[-1],
        )
    )

    print(f"Downloaded to: {model_dir}")

    config_json = model_dir / "config.json"
    if config_json.exists():
        print("\n=== config.json ===")
        print(json.dumps(json.loads(config_json.read_text()), indent=2, sort_keys=True))

    candidates = (
        list(model_dir.glob("*.pt"))
        + list(model_dir.glob("*.pth"))
        + list(model_dir.glob("*.bin"))
    )
    if not candidates:
        raise FileNotFoundError(f"No .pt/.pth/.bin checkpoint found in {model_dir}")

    ckpt_path = candidates[0]
    print(f"\nLoading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("\n=== raw checkpoint model_args ===")
    model_args = checkpoint.get("model_args", {})
    pprint.pp(model_args)

    print("\n=== reconstructed ModelConfig ===")
    model_config = ModelConfig(**model_args) if isinstance(model_args, dict) else model_args
    pprint.pp(asdict(model_config) if is_dataclass(model_config) else model_config)

    print("\n=== raw training config, if present ===")
    pprint.pp(checkpoint.get("config", {}))


if __name__ == "__main__":
    main()
