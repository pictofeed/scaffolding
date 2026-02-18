# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.cuda — CUDA compatibility stubs.

Scaffolding runs on CPU via NumPy.  These stubs allow code that checks
for CUDA availability to degrade gracefully.
"""
from __future__ import annotations

from . import amp


def is_available() -> bool:
    """Scaffolding has no CUDA backend — always returns False."""
    return False


def device_count() -> int:
    return 0


def get_device_capability(device: int = 0) -> tuple[int, int]:
    return (0, 0)


def get_device_name(device: int = 0) -> str:
    return "Scaffolding-CPU"


def set_device(device: int) -> None:
    pass


def current_device() -> int:
    return 0


def synchronize() -> None:
    pass


def memory_allocated(device: int = 0) -> int:
    return 0


def max_memory_allocated(device: int = 0) -> int:
    return 0


def empty_cache() -> None:
    pass


__all__ = [
    'is_available', 'device_count', 'get_device_capability',
    'get_device_name', 'set_device', 'current_device', 'synchronize',
    'memory_allocated', 'max_memory_allocated', 'empty_cache', 'amp',
]
