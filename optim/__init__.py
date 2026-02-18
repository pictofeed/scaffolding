# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.optim — Optimizers and LR schedulers."""
from __future__ import annotations

from .optimizer import Optimizer, AdamW
from . import lr_scheduler

__all__ = ['Optimizer', 'AdamW', 'lr_scheduler']
