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


# ── Delegated runtime APIs (mirror scaffolding.cuda for convenience) ──

def is_available() -> bool:
    """Return True if CUDA is available (delegates to scaffolding.cuda)."""
    from ..cuda import is_available as _is_avail
    return _is_avail()


def get_device_properties(device: int = 0):
    """Return device properties (delegates to scaffolding.cuda)."""
    from ..cuda import get_device_properties as _get_props
    return _get_props(device)


def empty_cache() -> None:
    """Release cached CUDA memory (delegates to scaffolding.cuda)."""
    from ..cuda import empty_cache as _empty
    _empty()


def device_count() -> int:
    """Number of CUDA devices (delegates to scaffolding.cuda)."""
    from ..cuda import device_count as _dc
    return _dc()


def total_memory_all_devices() -> int:
    """Combined VRAM across all CUDA devices (bytes).

    On a K80, sums both GK210 dies (~24 GB).
    """
    from ..cuda import total_memory_all_devices as _tma
    return _tma()


__all__ = [
    'matmul', 'is_built',
    'enable_flash_sdp', 'flash_sdp_enabled',
    'enable_math_sdp', 'math_sdp_enabled',
    'enable_cudnn_sdp', 'cudnn_sdp_enabled',
    'is_available', 'get_device_properties', 'empty_cache', 'device_count',
]
