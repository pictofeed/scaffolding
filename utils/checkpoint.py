# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.utils.checkpoint — Gradient checkpointing.

Saves memory by recomputing intermediate activations during backward
instead of storing them.  The Scaffolding implementation simply calls
the function normally (no recomputation) since the autograd graph is
lightweight compared to GPU-backed frameworks.
"""
from __future__ import annotations


def checkpoint(function, *args, use_reentrant=True, **kwargs):
    """Run *function* with gradient checkpointing.

    In Scaffolding this is a pass-through — the function is called
    normally.  Future versions may add recomputation-based memory
    savings.
    """
    return function(*args, **kwargs)
