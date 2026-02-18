# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Learning rate schedulers."""
from __future__ import annotations

from typing import Callable


class _LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._last_lr: list[float] = [g['lr'] for g in optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        new_lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group['lr'] = lr
        self._last_lr = new_lrs

    def get_lr(self) -> list[float]:
        raise NotImplementedError

    def get_last_lr(self) -> list[float]:
        return self._last_lr


class LambdaLR(_LRScheduler):
    """Multiply learning rate by a lambda function each step."""

    def __init__(self, optimizer, lr_lambda: Callable[[int], float],
                 last_epoch: int = -1):
        self.lr_lambda = lr_lambda
        self.base_lrs: list[float] = [g['lr'] for g in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        factor = self.lr_lambda(self.last_epoch)
        return [base * factor for base in self.base_lrs]


class StepLR(_LRScheduler):
    """Decay LR by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1,
                 last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        n = self.last_epoch // self.step_size
        factor = self.gamma ** n
        return [base * factor for base in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    """Cosine annealing schedule."""

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0,
                 last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        import math
        t = self.last_epoch % self.T_max
        cos_val = 0.5 * (1 + math.cos(math.pi * t / self.T_max))
        return [self.eta_min + (base - self.eta_min) * cos_val
                for base in self.base_lrs]
