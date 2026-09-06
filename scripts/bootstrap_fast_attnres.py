#!/usr/bin/env python3
"""Create the pinned CUDA 13 Fast-AttnRes environment runtime once."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements-fast-attnres-cu13.txt"
STAMP = ROOT / ".runtime" / "fast-attnres-bootstrap.sha256"


def run(*args: str) -> None:
    environment = os.environ.copy()
    environment["PATH"] = f"{VENV / 'bin'}:{environment.get('PATH', '')}"
    cuda_home = Path(environment.get("CUDA_HOME", "/usr/local/cuda"))
    if cuda_home.is_dir():
        environment["CUDA_HOME"] = str(cuda_home.resolve())
        environment["PATH"] = f"{cuda_home.resolve() / 'bin'}:{environment['PATH']}"
    subprocess.run(args, cwd=ROOT, check=True, env=environment)


def main() -> None:
    digest = hashlib.sha256(REQUIREMENTS.read_bytes() + Path(__file__).read_bytes()).hexdigest()
    python = VENV / "bin" / "python"
    if python.is_file() and STAMP.is_file() and STAMP.read_text().strip() == digest:
        return

    if python.is_file():
        probe = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import pathlib, sys, sysconfig; "
                    "assert (pathlib.Path(sysconfig.get_paths()['include']) / 'Python.h').is_file(); "
                    "print(f'{sys.version_info.major}.{sys.version_info.minor}')"
                ),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if probe.returncode or probe.stdout.strip() != "3.12":
            shutil.rmtree(VENV)
    if not python.is_file():
        uv = shutil.which("uv")
        if uv is not None:
            # uv-managed CPython includes Python.h, which Triton's launcher
            # needs and minimal system-Python images frequently omit.
            run(uv, "venv", "--managed-python", "--python", "3.12", "--seed", str(VENV))
        else:
            if sys.version_info[:2] != (3, 12):
                raise RuntimeError(
                    "install uv, or run this script with a complete Python 3.12 installation"
                )
            try:
                venv.EnvBuilder(with_pip=True).create(VENV)
            except Exception as exc:
                shutil.rmtree(VENV, ignore_errors=True)
                raise RuntimeError(
                    "could not create .venv; install uv or python3.12-venv"
                ) from exc

    run(str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS))
    run(
        str(python),
        "-c",
        (
            "import flash_attn, torch, triton; "
            "from importlib.metadata import version; "
            "assert torch.__version__ == '2.10.0+cu130', torch.__version__; "
            "assert triton.__version__ == '3.6.0', triton.__version__; "
            "assert flash_attn.__version__ == '2.8.3', flash_attn.__version__; "
            "assert version('fast-attnres') == '2.0.1'"
        ),
    )
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(digest + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
