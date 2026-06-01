# schedulers.py
import math
from typing import Literal
import torch
class WarmupStableDecay:
    """
    Warmup -> constant -> decay LR schedule.
    """

    def __init__(
        self,
        warmup_steps: int,
        max_steps: int,
        warmdown_steps: int,
        mode: Literal["linear", "sqrt"] = "sqrt",
        max_lr: float = 1.0,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmdown_steps = warmdown_steps
        self.max_lr = max_lr

        mode = mode.lower()
        if mode not in ("linear", "sqrt"):
            raise ValueError("mode must be 'linear' or 'sqrt'")
        self.mode = mode

    def __call__(self, step: int) -> float:
        """Callable interface for LambdaLR."""
        if self.mode == "linear":
            return self._wsd_linear(step)
        else:
            return self._wsd_sqrt(step)

    def _wsd_linear(self, step: int) -> float:
        """Linear warmup -> constant -> linear warmdown."""
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps if self.warmup_steps > 0 else 1.0
        elif step < self.max_steps - self.warmdown_steps:
            return 1.0
        else:
            decay_ratio = (self.max_steps - step) / self.warmdown_steps
            return 0.1 + 0.9 * decay_ratio

    def _wsd_sqrt(self, step: int) -> float:
        """Linear warmup -> constant -> 1 - sqrt warmdown."""
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps if self.warmup_steps > 0 else 1.0
        elif step < self.max_steps - self.warmdown_steps:
            return 1.0
        else:
            progress_into_cooldown = (step - (self.max_steps - self.warmdown_steps) + 1) / self.warmdown_steps
            return 0.1 + 0.9 * (1 - math.sqrt(progress_into_cooldown))

def get_schedulers(config, muon_optimizer, adamw_optimizer):
    lr_lambda = WarmupStableDecay(
        warmup_steps=config["warmup_steps"],
        max_steps=config["max_steps"],
        warmdown_steps=config["warmdown_steps"],
        mode=config["sched_mode"],
    )

    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(muon_optimizer, lr_lambda=lr_lambda),
        torch.optim.lr_scheduler.LambdaLR(adamw_optimizer, lr_lambda=lr_lambda),
    ]
    return schedulers
