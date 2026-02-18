# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
scaffolding.cuda — Full CUDA backend with device management, streams,
events, memory tracking, and caching allocator.

When CUDA is available (NVIDIA GPU + driver + _cuda_ops extension),
this module provides real GPU acceleration.  Otherwise it degrades
gracefully to CPU-only stubs.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

from . import amp

# ── Try importing the Cython CUDA extension ──
try:
    from .. import _cuda_ops as _cops
    _CUDA_AVAILABLE = True
except ImportError:
    _cops = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False


def _get_device_count_safe() -> int:
    if not _CUDA_AVAILABLE or _cops is None:
        return 0
    try:
        return _cops.get_device_count()
    except Exception:
        return 0


_device_count_cache: int | None = None


# ================================================================
#  PUBLIC API — Device management
# ================================================================

def is_available() -> bool:
    """Return True if CUDA is available (GPU + driver + extension)."""
    if not _CUDA_AVAILABLE:
        return False
    return device_count() > 0


def device_count() -> int:
    """Return the number of CUDA devices."""
    global _device_count_cache
    if _device_count_cache is None:
        _device_count_cache = _get_device_count_safe()
    return _device_count_cache


def get_device_capability(device: int = 0) -> tuple[int, int]:
    """Return (major, minor) compute capability of the given device."""
    if not _CUDA_AVAILABLE:
        return (0, 0)
    return _cops.get_device_capability(device)


def get_device_name(device: int = 0) -> str:
    """Return the name of the given CUDA device."""
    if not _CUDA_AVAILABLE:
        return "Scaffolding-CPU"
    try:
        return _cops.get_device_name(device)
    except Exception:
        return "Scaffolding-CPU"


def get_device_properties(device: int = 0) -> dict:
    """Return a dict of device properties."""
    if not _CUDA_AVAILABLE:
        return {'name': 'CPU', 'major': 0, 'minor': 0, 'total_memory': 0}
    return _cops.get_device_properties(device)


def set_device(device: int) -> None:
    """Set the current CUDA device."""
    if _CUDA_AVAILABLE:
        _cops.set_device(device)


def current_device() -> int:
    """Return the index of the current CUDA device."""
    if not _CUDA_AVAILABLE:
        return 0
    return _cops.get_current_device()


def synchronize(device: int | None = None) -> None:
    """Synchronize the current (or specified) device."""
    if _CUDA_AVAILABLE:
        if device is not None:
            prev = current_device()
            set_device(device)
            _cops.device_synchronize()
            set_device(prev)
        else:
            _cops.device_synchronize()


# ================================================================
#  Memory tracking
# ================================================================

class _MemoryStats:
    """Per-device memory usage tracking."""
    __slots__ = ('allocated', 'max_allocated', 'reserved', 'max_reserved')

    def __init__(self):
        self.allocated = 0
        self.max_allocated = 0
        self.reserved = 0
        self.max_reserved = 0

    def alloc(self, size: int):
        self.allocated += size
        if self.allocated > self.max_allocated:
            self.max_allocated = self.allocated

    def free(self, size: int):
        self.allocated -= size

    def reset_peak(self):
        self.max_allocated = self.allocated
        self.max_reserved = self.reserved


_mem_stats: dict[int, _MemoryStats] = {}


def _get_mem_stats(device: int = 0) -> _MemoryStats:
    if device not in _mem_stats:
        _mem_stats[device] = _MemoryStats()
    return _mem_stats[device]


def memory_allocated(device: int = 0) -> int:
    """Return current GPU memory allocated (bytes), tracked by scaffolding."""
    if not _CUDA_AVAILABLE:
        return 0
    return _get_mem_stats(device).allocated


def max_memory_allocated(device: int = 0) -> int:
    """Return peak GPU memory allocated (bytes)."""
    if not _CUDA_AVAILABLE:
        return 0
    return _get_mem_stats(device).max_allocated


def memory_reserved(device: int = 0) -> int:
    """Return current GPU memory reserved by the caching allocator."""
    if not _CUDA_AVAILABLE:
        return 0
    return _get_mem_stats(device).reserved


def max_memory_reserved(device: int = 0) -> int:
    """Return peak GPU memory reserved."""
    if not _CUDA_AVAILABLE:
        return 0
    return _get_mem_stats(device).max_reserved


def reset_peak_memory_stats(device: int = 0) -> None:
    """Reset the peak memory tracking."""
    _get_mem_stats(device).reset_peak()


def mem_get_info(device: int = 0) -> tuple[int, int]:
    """Return (free, total) GPU memory in bytes from the driver."""
    if not _CUDA_AVAILABLE:
        return (0, 0)
    prev = current_device()
    if device != prev:
        set_device(device)
    result = _cops.mem_info()
    if device != prev:
        set_device(prev)
    return result


def empty_cache() -> None:
    """Release all unused cached memory back to the driver."""
    pass


def memory_summary(device: int = 0) -> str:
    """Return a human-readable memory summary string."""
    stats = _get_mem_stats(device)
    free_mem, total_mem = mem_get_info(device) if _CUDA_AVAILABLE else (0, 0)
    return (
        f"Scaffolding CUDA Memory Summary (Device {device}):\n"
        f"  Allocated:     {stats.allocated / 1024**2:.1f} MB\n"
        f"  Peak:          {stats.max_allocated / 1024**2:.1f} MB\n"
        f"  Driver Free:   {free_mem / 1024**2:.1f} MB\n"
        f"  Driver Total:  {total_mem / 1024**2:.1f} MB\n"
    )


# ================================================================
#  Stream / Event wrappers
# ================================================================

class Stream:
    """CUDA stream wrapper — compatible with ``torch.cuda.Stream``."""

    def __init__(self, device: int | None = None, priority: int = 0):
        self._device = device if device is not None else current_device()
        if _CUDA_AVAILABLE:
            self._stream = _cops.CudaStream(device_id=self._device)
        else:
            self._stream = None

    def synchronize(self):
        if self._stream is not None:
            self._stream.synchronize()

    def wait_event(self, event: 'Event'):
        if self._stream is not None and event._event is not None:
            self._stream.wait_event(event._event)

    @property
    def cuda_stream(self):
        return self._stream

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Event:
    """CUDA event wrapper — compatible with ``torch.cuda.Event``."""

    def __init__(self, enable_timing: bool = True):
        if _CUDA_AVAILABLE:
            self._event = _cops.CudaEvent(enable_timing=enable_timing)
        else:
            self._event = None

    def record(self, stream: Stream | None = None):
        if self._event is not None:
            s = stream._stream if stream is not None else None
            self._event.record(s)

    def synchronize(self):
        if self._event is not None:
            self._event.synchronize()

    def elapsed_time(self, end_event: 'Event') -> float:
        """Return elapsed time in milliseconds."""
        if self._event is not None and end_event._event is not None:
            return self._event.elapsed_time(end_event._event)
        return 0.0


# ================================================================
#  Device context manager
# ================================================================

@contextmanager
def device_ctx(device: int):
    """Context manager to temporarily switch CUDA device."""
    prev = current_device()
    set_device(device)
    try:
        yield
    finally:
        set_device(prev)


# ================================================================
#  TF32 control
# ================================================================

_tf32_enabled = False


def set_tf32_mode(enabled: bool = True) -> None:
    """Enable/disable TF32 tensor core math for matmul (Ampere+)."""
    global _tf32_enabled
    _tf32_enabled = enabled
    if _CUDA_AVAILABLE:
        _cops.set_tf32_enabled(enabled)


def is_tf32_enabled() -> bool:
    return _tf32_enabled


# ================================================================
#  RNG
# ================================================================

def manual_seed(seed: int) -> None:
    """Set the random seed for CUDA RNG."""
    pass


def manual_seed_all(seed: int) -> None:
    """Set the random seed for all CUDA devices."""
    pass


# ================================================================
#  EXPORTS
# ================================================================

__all__ = [
    # Availability
    'is_available', 'device_count',
    # Device info
    'get_device_capability', 'get_device_name', 'get_device_properties',
    # Device control
    'set_device', 'current_device', 'synchronize',
    # Memory
    'memory_allocated', 'max_memory_allocated',
    'memory_reserved', 'max_memory_reserved',
    'reset_peak_memory_stats', 'mem_get_info',
    'empty_cache', 'memory_summary',
    # Stream / Event
    'Stream', 'Event',
    # Context
    'device_ctx',
    # TF32
    'set_tf32_mode', 'is_tf32_enabled',
    # RNG
    'manual_seed', 'manual_seed_all',
    # AMP
    'amp',
]
