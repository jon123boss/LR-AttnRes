"""Optional adapter for the public Fast-AttnRes v1.0.0 API.

The repository's model supports more AttnRes variants than the small public
``fast-attnres`` package.  This module deliberately keeps the supported
surface narrow: callers pass the full-width source values and a *prepared*
one-dimensional query to ``attnres.attnres``.  Unsupported model semantics or
the package's documented shape/dtype envelope return a structured legacy
fallback decision.  A missing/wrong package and an error raised by an actual
runtime kernel are hard failures.

Only the public import ``from attnres import attnres`` is used.  In
particular, this adapter does not reach into the package's reference or kernel
modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import Tensor


FAST_ATTNRES_VERSION = "1.0.0"
FAST_ATTNRES_DISTRIBUTION = "fast-attnres"
FAST_ATTNRES_MAX_SOURCES = 129
FAST_ATTNRES_MAX_WIDTH = 8192
FAST_ATTNRES_DTYPES = (torch.bfloat16, torch.float32)


class FastAttnResPackageError(RuntimeError):
    """The requested Fast-AttnRes package is absent or has the wrong API."""


class FastAttnResRuntimeError(RuntimeError):
    """The selected Fast-AttnRes runtime/kernel failed during execution."""


@dataclass(frozen=True)
class FastAttnResDecision:
    """A route decision suitable for accounting and diagnostics.

    ``path`` is either ``"fast"`` or ``"legacy"``.  ``reason`` is a stable,
    machine-readable code when the legacy path is selected; ``detail`` keeps a
    short human-readable explanation without making callers parse it.
    """

    path: str
    reason: Optional[str] = None
    detail: Optional[str] = None

    @property
    def eligible(self) -> bool:
        return self.path == "fast"

    @property
    def will_execute(self) -> bool:
        return self.eligible

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "eligible": self.eligible,
            "will_execute": self.will_execute,
            "reason": self.reason,
            "detail": self.detail,
        }


def _legacy(reason: str, detail: str | None = None) -> FastAttnResDecision:
    return FastAttnResDecision("legacy", reason, detail)


def _fast() -> FastAttnResDecision:
    return FastAttnResDecision("fast")


def _as_source_tuple(values: Tensor | Iterable[Tensor]) -> tuple[Tensor, ...] | None:
    if isinstance(values, Tensor):
        if values.ndim < 2:
            return None
        # ``unbind`` preserves aliases and strides of each source.  It also
        # mirrors the public package's accepted packed representation without
        # materializing an extra stack in the adapter.
        return tuple(values.unbind(0))
    if isinstance(values, (list, tuple)):
        return tuple(values)
    return None


def assess_fast_attnres_inputs(
    values: Tensor | list[Tensor] | tuple[Tensor, ...],
    query: Tensor,
    *,
    key_norm: bool = True,
    source_counts: Any = None,
    source_logit_biases: Any = None,
    device_type: str | None = None,
) -> FastAttnResDecision:
    """Check the public operator's input envelope.

    This is intentionally independent of package discovery, so an
    incompatible model can cleanly use its legacy implementation even on a
    machine without the optional dependency.  ``source_counts`` and
    ``source_logit_biases`` are accepted only to make the incompatibility
    explicit: v1.0.0 has no bias/prior arguments.
    """

    if not key_norm:
        return _legacy(
            "key_normalization_disabled",
            "Fast-AttnRes always RMS-normalizes its implicit tail key",
        )
    if source_counts is not None or source_logit_biases is not None:
        return _legacy(
            "source_counts_or_logit_biases",
            "Fast-AttnRes v1.0.0 has no source-count or source-logit-bias API",
        )

    if not isinstance(query, Tensor):
        return _legacy("unsupported_query", "query must be a tensor")
    if query.ndim != 1:
        return _legacy("unsupported_query_shape", "Fast-AttnRes requires a one-dimensional query")

    source_tuple = _as_source_tuple(values)
    if source_tuple is None:
        return _legacy("unsupported_values_container", "values must be packed or an ordered list/tuple")
    source_count = len(source_tuple)
    if source_count < 2:
        # A one-source read is an exact identity/no-op in the legacy model and
        # should not count as an actual Fast execution.
        return _legacy("single_source_noop", "Fast-AttnRes is unnecessary for one source")
    if source_count > FAST_ATTNRES_MAX_SOURCES:
        return _legacy(
            "unsupported_source_count",
            f"Fast-AttnRes supports at most {FAST_ATTNRES_MAX_SOURCES} sources",
        )

    if any(not isinstance(source, Tensor) for source in source_tuple):
        return _legacy("unsupported_source", "all sources must be tensors")
    first = source_tuple[0]
    if first.ndim < 1:
        return _legacy("unsupported_source_shape", "sources must have shape [..., D]")
    if any(int(size) < 1 for size in first.shape):
        return _legacy("unsupported_source_shape", "source dimensions must be positive")
    width = int(first.shape[-1])
    if width < 1 or width > FAST_ATTNRES_MAX_WIDTH:
        return _legacy(
            "unsupported_value_width",
            f"Fast-AttnRes supports 1 <= D <= {FAST_ATTNRES_MAX_WIDTH}",
        )
    if first.dtype not in FAST_ATTNRES_DTYPES:
        return _legacy(
            "unsupported_dtype",
            "Fast-AttnRes values must use BF16 or FP32 storage",
        )
    if device_type is None:
        device_type = first.device.type
    if device_type not in {"cpu", "cuda"}:
        return _legacy(
            "unsupported_device",
            "Fast-AttnRes v1.0.0 supports CPU reference and CUDA kernels",
        )
    if query.dtype not in FAST_ATTNRES_DTYPES:
        return _legacy(
            "unsupported_query_dtype",
            "Fast-AttnRes query must use BF16 or FP32 storage",
        )
    if query.device != first.device:
        return _legacy("query_device_mismatch", "query and values must share a device")
    if query.numel() < 1 or query.numel() > width:
        return _legacy("unsupported_query_rank", "query rank must satisfy 1 <= R <= D")

    first_shape = tuple(first.shape)
    for index, source in enumerate(source_tuple):
        if source.ndim < 1 or tuple(source.shape) != first_shape:
            return _legacy(
                "source_shape_mismatch",
                f"source {index} does not match the first source shape",
            )
        if source.dtype != first.dtype or source.device != first.device:
            return _legacy(
                "source_dtype_or_device_mismatch",
                f"source {index} does not match the first source dtype/device",
            )

    return _fast()


def _validate_public_api(function: Any) -> None:
    """Reject a same-named but incompatible ``attnres`` package/API."""

    if not callable(function):
        raise FastAttnResPackageError(
            "fast-attnres v1.0.0 is required, but attnres.attnres is not callable"
        )
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise FastAttnResPackageError(
            "fast-attnres v1.0.0 is required, but its public attnres signature is unavailable"
        ) from exc
    parameters = signature.parameters
    required = {"values", "query", "eps", "scale"}
    if not required.issubset(parameters):
        raise FastAttnResPackageError(
            "fast-attnres v1.0.0 is required; expected public "
            "attnres(values, query, *, eps=..., scale=...)"
        )
    # ``eps`` and ``scale`` must be accepted by keyword.  A positional-only
    # function with these names is not the v1.0.0 public API.
    for name in ("eps", "scale"):
        if parameters[name].kind is inspect.Parameter.POSITIONAL_ONLY:
            raise FastAttnResPackageError(
                "fast-attnres v1.0.0 is required; eps and scale must be keyword arguments"
            )


@lru_cache(maxsize=1)
def load_fast_attnres() -> Any:
    """Load and validate exactly Fast-AttnRes v1.0.0's public function."""

    try:
        from attnres import attnres
    except Exception as exc:
        raise FastAttnResPackageError(
            "attnres_backend='fast' requires fast-attnres==1.0.0; "
            "install it with `python -m pip install fast-attnres==1.0.0`"
        ) from exc

    try:
        installed_version = importlib.metadata.version(FAST_ATTNRES_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise FastAttnResPackageError(
            "attnres_backend='fast' requires the fast-attnres==1.0.0 distribution; "
            "the imported attnres module has no matching distribution metadata"
        ) from exc
    if installed_version != FAST_ATTNRES_VERSION:
        raise FastAttnResPackageError(
            "attnres_backend='fast' requires fast-attnres==1.0.0, "
            f"but found {installed_version!r}"
        )
    _validate_public_api(attnres)
    return attnres


def fast_attnres_package_provenance() -> dict[str, Any]:
    function = load_fast_attnres()
    distribution = importlib.metadata.distribution(FAST_ATTNRES_DISTRIBUTION)
    hashes = {}
    installed_sources = set()
    combined = hashlib.sha256()
    for relative in distribution.files or ():
        relative_path = Path(str(relative))
        if not relative_path.parts or relative_path.parts[0] != "attnres":
            continue
        if relative_path.suffix not in {".py", ".pyi"}:
            continue
        installed_path = Path(distribution.locate_file(relative)).resolve()
        if not installed_path.is_file():
            continue
        contents = installed_path.read_bytes()
        installed_sources.add(installed_path)
        name = relative_path.as_posix()
        hashes[name] = hashlib.sha256(contents).hexdigest()
        combined.update(name.encode())
        combined.update(b"\0")
        combined.update(contents)
        combined.update(b"\0")

    public_api_path = inspect.getsourcefile(function)
    if not public_api_path or Path(public_api_path).resolve() not in installed_sources:
        raise FastAttnResPackageError(
            "The imported Fast-AttnRes public API does not belong to the installed distribution"
        )
    return {
        "version": distribution.version,
        "distribution_source_sha256": combined.hexdigest(),
        "source_hashes": hashes,
    }


def format_fast_attnres_banner(report: dict[str, Any]) -> str | None:
    """Format the startup line only for a route that will execute Fast."""
    if not int(report.get("active_reads", 0)):
        return None
    return (
        "[Fast-AttnRes] backend=fast-attnres "
        f"version={report.get('version')} "
        f"active_reads={report['active_reads']}/{report['total_reads']} "
        f"legacy_fallback_reads={report['legacy_fallback_reads']}"
    )


def print_fast_attnres_banner(
    report: dict[str, Any], *, is_rank_zero: bool, print_fn=print
) -> bool:
    """Emit the Fast line on rank zero iff at least one read is active."""
    line = format_fast_attnres_banner(report)
    if not is_rank_zero or line is None:
        return False
    print_fn(line, flush=True)
    return True


def fast_attnres_read(
    values: Tensor | list[Tensor] | tuple[Tensor, ...],
    query: Tensor,
    *,
    eps: float,
    scale: float = 1.0,
    key_norm: bool = True,
    source_counts: Any = None,
    source_logit_biases: Any = None,
    device_type: str | None = None,
) -> tuple[Tensor | None, FastAttnResDecision]:
    """Execute one eligible Fast-AttnRes read or return a legacy decision.

    Package/runtime failures intentionally do not silently select the legacy
    route.  The only fallback decisions returned here are the semantic and
    documented input-domain checks performed before the package call.
    """

    decision = assess_fast_attnres_inputs(
        values,
        query,
        key_norm=key_norm,
        source_counts=source_counts,
        source_logit_biases=source_logit_biases,
        device_type=device_type,
    )
    if not decision.eligible:
        return None, decision

    function = load_fast_attnres()
    try:
        output = function(values, query, eps=float(eps), scale=float(scale))
    except Exception as exc:
        raise FastAttnResRuntimeError(
            "Fast-AttnRes runtime/kernel execution failed"
        ) from exc

    if not isinstance(output, Tensor):
        raise FastAttnResRuntimeError(
            "Fast-AttnRes public attnres returned a non-tensor result"
        )
    return output, decision


def fast_attnres_config_decision(
    *,
    use_attnres: bool,
    use_lrid: bool,
    attnres_type: str,
    key_norm: bool,
    lrid_rank: int,
    n_embd: int,
    lrid_num_heads: int = 1,
    lrid_input_dependent_query: bool = False,
    lrid_key_from_output_tail: bool = False,
    lrid_key_from_value: bool = False,
    lrid_key_from_value_shared: bool = False,
    lrid_query_from_value: bool = False,
    lrid_query_from_value_shared: bool = False,
    lrid_static_embedding_key: bool = False,
    lrid_add_static_embedding_key: bool = False,
    lrid_add_static_source_key: bool = False,
    n_layer: int = 1,
) -> FastAttnResDecision:
    """Check model-level semantics independent of runtime input shapes.

    The public operator has implicit tail keys and one static query.  All
    explicit/projected keys, dynamic queries, multi-head LRID, and source
    priors therefore remain on the legacy path.  Block value summaries are
    allowed because the caller has already materialized the exact summary;
    nonzero count/logit priors are not representable by v1.0.0.
    """

    if not use_attnres:
        return _legacy("attnres_disabled", "the model does not use AttnRes")
    if str(attnres_type).lower() not in {"full", "block"}:
        return _legacy("unsupported_attnres_type", "expected Full or Block")
    if not key_norm:
        return _legacy("key_normalization_disabled", "Fast-AttnRes always normalizes tail keys")
    if int(n_layer) < 1:
        return _legacy("no_multi_source_read", "the model has no read with multiple sources")

    if not use_lrid:
        if int(lrid_rank) != int(n_embd):
            # ``lrid_rank`` is irrelevant to standard AttnRes, so this branch
            # is intentionally not a rejection.  The caller passes D as the
            # effective rank for its standard query.
            pass
    else:
        if not lrid_key_from_output_tail:
            return _legacy(
                "projected_or_non_tail_lrid_key",
                "Fast-AttnRes v1.0.0 only has implicit output-tail keys",
            )
        if int(lrid_rank) >= int(n_embd):
            return _legacy(
                "lrid_rank_not_less_than_width",
                "the supported LR adapter requires R < D",
            )
        if int(lrid_num_heads) != 1:
            return _legacy("multi_head_lrid", "the Fast LR adapter supports one routing head")
        if lrid_input_dependent_query:
            return _legacy("dynamic_lrid_query", "the Fast LR adapter requires a static query")
        if (
            lrid_key_from_value
            or lrid_key_from_value_shared
            or lrid_query_from_value
            or lrid_query_from_value_shared
            or lrid_static_embedding_key
            or lrid_add_static_embedding_key
            or lrid_add_static_source_key
        ):
            return _legacy(
                "explicit_lrid_projection_or_static_key",
                "only implicit output-tail LR keys are representable",
            )

    return _fast()


__all__ = [
    "FAST_ATTNRES_VERSION",
    "FAST_ATTNRES_DISTRIBUTION",
    "FAST_ATTNRES_MAX_SOURCES",
    "FAST_ATTNRES_MAX_WIDTH",
    "FastAttnResDecision",
    "FastAttnResPackageError",
    "FastAttnResRuntimeError",
    "assess_fast_attnres_inputs",
    "fast_attnres_config_decision",
    "fast_attnres_read",
    "fast_attnres_package_provenance",
    "format_fast_attnres_banner",
    "print_fast_attnres_banner",
    "load_fast_attnres",
]
