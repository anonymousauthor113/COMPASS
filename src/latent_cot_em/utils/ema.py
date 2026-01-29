from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class EMAConfig:
    """Configuration for exponential moving average tracking.

    Notes:
    - The EMA is tracked over **trainable** parameters only (requires_grad=True). This is
      important for memory efficiency when training adapters (e.g., LoRA).
    """

    decay: float = 0.999
    update_every: int = 1
    warmup_steps: int = 0


class ModelEMA:
    """EMA tracker for trainable parameters.

    This implementation is intentionally lightweight:
    - Stores shadow copies for parameters with requires_grad=True.
    - Supports a context manager that temporarily applies EMA weights for reference-policy
      evaluations (e.g., PPO-style ratios, DPO reference).

    If you train all model parameters (not adapter-style), EMA will mirror the full model
    and thus consume comparable memory.
    """

    def __init__(self, model: nn.Module, cfg: EMAConfig) -> None:
        self.cfg = cfg
        self._shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._shadow[name] = p.detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self._shadow = {k: v.clone() for k, v in state.items()}

    def update(self, model: nn.Module, step: int) -> None:
        if step < self.cfg.warmup_steps:
            # Initialize from current trainable parameters.
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self._shadow[name] = p.detach().clone()
            return

        if self.cfg.update_every <= 0:
            return
        if step % self.cfg.update_every != 0:
            return

        d = float(self.cfg.decay)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self._shadow:
                self._shadow[name] = p.detach().clone()
                continue
            self._shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def _apply_shadow(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self._shadow:
                p.data.copy_(self._shadow[name].to(device=p.device, dtype=p.dtype))

    @contextlib.contextmanager
    def use_ema(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap trainable parameters to EMA values.

        This is intended for computing reference-policy log-probabilities without maintaining
        a full second model instance.
        """

        backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self._shadow:
                backup[name] = p.detach().clone()
        try:
            self._apply_shadow(model)
            yield
        finally:
            # Restore
            for name, p in model.named_parameters():
                if p.requires_grad and name in backup:
                    p.data.copy_(backup[name].to(device=p.device, dtype=p.dtype))
