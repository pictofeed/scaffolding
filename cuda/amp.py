# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.cuda.amp — Automatic Mixed Precision stubs.

GradScaler and autocast are no-ops in Scaffolding (CPU-only).
"""
from __future__ import annotations

from contextlib import contextmanager


class GradScaler:
    """No-op GradScaler for AMP compatibility."""

    def __init__(self, enabled: bool = True, **kwargs):
        self.enabled = enabled
        self._scale = 1.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def get_scale(self) -> float:
        return self._scale

    def state_dict(self) -> dict:
        return {'scale': self._scale}

    def load_state_dict(self, state_dict: dict):
        self._scale = state_dict.get('scale', 1.0)


class autocast:
    """No-op autocast context manager / decorator."""

    def __init__(self, enabled: bool = True, dtype=None, **kwargs):
        self.enabled = enabled

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with self:
                return fn(*a, **kw)
        return wrapper
