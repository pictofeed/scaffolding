# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.distributed — Distributed training stubs.

Provides the same API as torch.distributed so code that checks
``dist.is_initialized()`` or calls ``dist.barrier()`` degrades
gracefully in single-process mode.
"""
from __future__ import annotations

_initialized: bool = False
_rank: int = 0
_world_size: int = 1
_backend: str = 'gloo'


def init_process_group(backend: str = 'gloo',
                       rank: int = 0,
                       world_size: int = 1,
                       **kwargs) -> None:
    global _initialized, _rank, _world_size, _backend
    _initialized = True
    _rank = rank
    _world_size = world_size
    _backend = backend


def is_initialized() -> bool:
    return _initialized


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def barrier() -> None:
    """No-op barrier in single-process mode."""
    pass


def all_reduce(tensor, op=None):
    """No-op all_reduce."""
    return tensor


def broadcast(tensor, src: int = 0):
    return tensor


def all_gather(tensor_list, tensor):
    if tensor_list:
        tensor_list[0] = tensor


def destroy_process_group() -> None:
    global _initialized, _rank, _world_size
    _initialized = False
    _rank = 0
    _world_size = 1


class ReduceOp:
    SUM = 'SUM'
    PRODUCT = 'PRODUCT'
    MIN = 'MIN'
    MAX = 'MAX'
