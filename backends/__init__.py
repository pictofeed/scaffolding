# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.backends — Backend configuration stubs."""
from __future__ import annotations

from . import cudnn
from . import cuda
from . import mps

__all__ = ['cudnn', 'cuda', 'mps']
