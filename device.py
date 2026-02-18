# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Device abstraction mirroring PyTorch's torch.device."""
from __future__ import annotations


class device:
    """Represents a compute device (cpu, cuda, mps)."""

    __slots__ = ('_type', '_index')

    def __init__(self, type_or_str: str = 'cpu', index: int | None = None):
        if isinstance(type_or_str, device):
            self._type = type_or_str._type
            self._index = type_or_str._index
            return
        s = str(type_or_str)
        if ':' in s:
            parts = s.split(':')
            self._type = parts[0]
            self._index = int(parts[1])
        else:
            self._type = s
            self._index = index

    @property
    def type(self) -> str:
        return self._type

    @property
    def index(self) -> int | None:
        return self._index

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            other = device(other)
        if not isinstance(other, device):
            return NotImplemented
        return self._type == other._type and self._index == other._index

    def __hash__(self) -> int:
        return hash((self._type, self._index))

    def __repr__(self) -> str:
        if self._index is not None:
            return f"device(type='{self._type}', index={self._index})"
        return f"device(type='{self._type}')"

    def __str__(self) -> str:
        if self._index is not None:
            return f"{self._type}:{self._index}"
        return self._type
