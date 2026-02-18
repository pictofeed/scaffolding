# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.backends.cuda — CUDA backend configuration.

Provides matmul-level TF32 control and flash-sdp / cudnn-sdp toggles,
mirroring ``torch.backends.cuda``.
"""
from __future__ import annotations

try:
    from .. import _cuda_ops as _cops
    _CUDA = True
except ImportError:
    _cops = None
    _CUDA = False


class _MatmulConfig:
    """Control tensor-core math for matmul operations."""
    _allow_tf32: bool = False

    @property
    def allow_tf32(self) -> bool:
        return self._allow_tf32

    @allow_tf32.setter
    def allow_tf32(self, value: bool):
        self._allow_tf32 = value
        if _CUDA and _cops is not None:
            _cops.set_tf32_enabled(value)


matmul = _MatmulConfig()


# ── Flash / math SDP toggles (compatibility stubs) ──
_flash_sdp_enabled: bool = True
_math_sdp_enabled: bool = True
_cudnn_sdp_enabled: bool = True


def enable_flash_sdp(enabled: bool = True) -> None:
    global _flash_sdp_enabled
    _flash_sdp_enabled = enabled


def flash_sdp_enabled() -> bool:
    return _flash_sdp_enabled


def enable_math_sdp(enabled: bool = True) -> None:
    global _math_sdp_enabled
    _math_sdp_enabled = enabled


def math_sdp_enabled() -> bool:
    return _math_sdp_enabled


def enable_cudnn_sdp(enabled: bool = True) -> None:
    global _cudnn_sdp_enabled
    _cudnn_sdp_enabled = enabled


def cudnn_sdp_enabled() -> bool:
    return _cudnn_sdp_enabled


def is_built() -> bool:
    """Return True if scaffolding was compiled with CUDA support."""
    return _CUDA


__all__ = [
    'matmul', 'is_built',
    'enable_flash_sdp', 'flash_sdp_enabled',
    'enable_math_sdp', 'math_sdp_enabled',
    'enable_cudnn_sdp', 'cudnn_sdp_enabled',
]
