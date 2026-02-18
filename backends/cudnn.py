# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.backends.cudnn — cuDNN configuration.

Exposes the three standard cuDNN knobs that PyTorch users expect:
``benchmark``, ``deterministic``, and ``enabled``.

When CUDA is available these flags influence kernel selection in
the CUDA backend.  Otherwise they are harmless no-ops.
"""
from __future__ import annotations


class _CuDNNConfig:
    """Module-level cuDNN configuration singleton."""
    __slots__ = ('_benchmark', '_deterministic', '_enabled', '_allow_tf32',
                 '_version')

    def __init__(self):
        self._benchmark = False
        self._deterministic = False
        self._enabled = True   # cuDNN is on by default
        self._allow_tf32 = True
        self._version = self._detect_version()

    @staticmethod
    def _detect_version() -> int | None:
        """Try to read the cuDNN version from the runtime."""
        try:
            from .. import _cuda_ops as _cops  # type: ignore[attr-defined]
            v = getattr(_cops, 'cudnn_version', None)
            if callable(v):
                return v()
            return v
        except Exception:
            return None

    # ── benchmark ──
    @property
    def benchmark(self) -> bool:
        return self._benchmark

    @benchmark.setter
    def benchmark(self, value: bool):
        self._benchmark = value

    # ── deterministic ──
    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @deterministic.setter
    def deterministic(self, value: bool):
        self._deterministic = value

    # ── enabled ──
    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    # ── allow_tf32 ──
    @property
    def allow_tf32(self) -> bool:
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value: bool):
        self._allow_tf32 = value

    # ── version ──
    def version(self) -> int | None:
        """Return cuDNN version as an integer (e.g. 8904) or None."""
        return self._version

    def is_available(self) -> bool:
        """Return True if cuDNN is usable."""
        return self._enabled and self._version is not None


# Expose as module-level attributes so ``backends.cudnn.benchmark = True``
# works the same way as PyTorch.
_config = _CuDNNConfig()

benchmark    = _config.benchmark
deterministic = _config.deterministic
enabled       = _config.enabled
allow_tf32    = _config.allow_tf32


def version() -> int | None:
    return _config.version()


def is_available() -> bool:
    return _config.is_available()


__all__ = ['benchmark', 'deterministic', 'enabled', 'allow_tf32',
           'version', 'is_available']
