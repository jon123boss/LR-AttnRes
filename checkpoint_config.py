"""Reconstruct saved model settings without applying new-training defaults."""

from collections.abc import Mapping
from dataclasses import is_dataclass
import warnings

from model import ModelConfig


def model_config_from_checkpoint(model_args, training_config=None):
    """Preserve saved settings and migrate the pre-count-prior checkpoint format.

    Explicit model arguments take precedence over the saved training config.
    A missing count-prior flag in both means the legacy no-prior behavior, not
    ModelConfig's default for a newly constructed model. Unknown model arguments
    remain errors rather than silently discarding potentially architectural data.
    """
    if isinstance(model_args, Mapping):
        saved_args = dict(model_args)
    elif is_dataclass(model_args) and not isinstance(model_args, type):
        # vars is intentional: asdict/getattr can pick up today's class default
        # for a field absent from a historical pickled dataclass instance.
        saved_args = dict(vars(model_args))
        # ModelConfig caches this derived diagnostic outside its dataclass
        # fields; reconstruction recomputes it from the saved backend.
        saved_args.pop("_attnres_backend_requested", None)
    else:
        raise TypeError(
            "checkpoint['model_args'] must be a mapping or dataclass instance, "
            f"got {type(model_args)!r}"
        )

    prior_field = "attnres_block_count_prior"
    prior_source = None
    if prior_field not in saved_args:
        if isinstance(training_config, Mapping) and prior_field in training_config:
            saved_args[prior_field] = training_config[prior_field]
            prior_source = "saved training config"
        else:
            saved_args[prior_field] = False
            prior_source = "legacy checkpoint default (absent from both saved configs)"

    config = ModelConfig(**saved_args)
    if prior_source is not None and config.use_attnres and config.attnres_type == "block":
        warnings.warn(
            f"Checkpoint model_args has no {prior_field}; restoring "
            f"{prior_field}={getattr(config, prior_field)!r} from {prior_source}.",
            UserWarning,
            stacklevel=2,
        )
    return config
