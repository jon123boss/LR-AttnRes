"""Optional adapters for Liger Kernel.

The repo should remain importable without liger-kernel installed. Keep imports
centralized here so model and criterion code can ask for a kernel by name and
fall back to PyTorch when it is unavailable.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


ACCELERATOR_DEVICE_TYPES = {"cuda", "xpu", "npu"}
_KERNELS: Dict[str, object] = {}
_ERRORS: Dict[str, str] = {}


def _register(name, import_path, attr):
    try:
        module = __import__(import_path, fromlist=[attr])
        _KERNELS[name] = getattr(module, attr)
    except Exception as exc:
        _KERNELS[name] = None
        _ERRORS[name] = f"{type(exc).__name__}: {exc}"


_register("rms_norm", "liger_kernel.transformers.functional", "liger_rms_norm")
_register("rope", "liger_kernel.transformers", "liger_rotary_pos_emb")
_register("swiglu", "liger_kernel.transformers.functional", "liger_swiglu")
_register("cross_entropy", "liger_kernel.transformers", "LigerCrossEntropyLoss")
_register(
    "fused_linear_cross_entropy",
    "liger_kernel.transformers",
    "LigerFusedLinearCrossEntropyLoss",
)
_register("attnres", "liger_kernel.transformers.functional", "liger_attn_res")
_register(
    "embedding",
    "liger_kernel.ops.experimental.embedding",
    "LigerEmbeddingFunction",
)


def get_liger_kernel(name: str):
    return _KERNELS.get(name)


def liger_kernel_available(name: str) -> bool:
    return _KERNELS.get(name) is not None


def liger_import_error(name: str) -> Optional[str]:
    return _ERRORS.get(name)


def tensor_supports_liger(tensor: torch.Tensor) -> bool:
    return tensor.device.type in ACCELERATOR_DEVICE_TYPES


def rms_norm_eps(x: torch.Tensor, eps: Optional[float] = None) -> float:
    if eps is not None:
        return eps
    return torch.finfo(x.dtype).eps


def enabled_liger_kernels(config) -> Dict[str, bool]:
    return {
        "rms_norm": bool(getattr(config, "liger_rms_norm", False)),
        "rope": bool(getattr(config, "liger_rope", False)),
        "swiglu": bool(getattr(config, "liger_swiglu", False)),
        "embedding": bool(getattr(config, "liger_embedding", False)),
        "attnres": bool(getattr(config, "liger_attnres", False)),
    }


def report_liger_status(
    enabled: Dict[str, bool],
    strict: bool = False,
    extra_names: Iterable[str] = (),
) -> None:
    names = list(enabled)
    for name in extra_names:
        if name not in enabled:
            names.append(name)

    active = [name for name in names if enabled.get(name, False)]
    if not active:
        print("Liger kernels: disabled")
        return

    unavailable = [name for name in active if not liger_kernel_available(name)]
    available = [name for name in active if liger_kernel_available(name)]
    if available:
        print("Liger kernels enabled: " + ", ".join(available))
    if unavailable:
        details = ", ".join(
            f"{name} ({liger_import_error(name) or 'unavailable'})"
            for name in unavailable
        )
        message = f"Liger kernels unavailable, falling back for: {details}"
        if strict:
            raise RuntimeError(message)
        print(message)
