# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                             ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Data type definitions mirroring PyTorch's dtype system."""
from __future__ import annotations

import enum
import numpy as np


class dtype(enum.Enum):
    """Scaffolding data types — mirrors torch dtype constants."""
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    bfloat16 = "bfloat16"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    bool = "bool"

    def to_numpy(self) -> np.dtype:
        """Convert to numpy dtype."""
        _map = {
            dtype.float16: np.float16,
            dtype.float32: np.float32,
            dtype.float64: np.float64,
            dtype.bfloat16: np.float32,  # numpy has no bfloat16; use float32
            dtype.int8: np.int8,
            dtype.int16: np.int16,
            dtype.int32: np.int32,
            dtype.int64: np.int64,
            dtype.uint8: np.uint8,
            dtype.bool: np.bool_,
        }
        return np.dtype(_map[self])

    @staticmethod
    def from_numpy(np_dtype: np.dtype) -> 'dtype':
        """Convert numpy dtype to scaffolding dtype."""
        _map = {
            np.dtype(np.float16): dtype.float16,
            np.dtype(np.float32): dtype.float32,
            np.dtype(np.float64): dtype.float64,
            np.dtype(np.int8): dtype.int8,
            np.dtype(np.int16): dtype.int16,
            np.dtype(np.int32): dtype.int32,
            np.dtype(np.int64): dtype.int64,
            np.dtype(np.uint8): dtype.uint8,
            np.dtype(np.bool_): dtype.bool,
        }
        return _map.get(np.dtype(np_dtype), dtype.float32)

    def __repr__(self) -> str:
        return f"scaffolding.{self.name}"


# Convenience aliases (top-level torch.float32, torch.long, etc.)
float16 = dtype.float16
float32 = dtype.float32
float64 = dtype.float64
bfloat16 = dtype.bfloat16
half = dtype.float16
int8 = dtype.int8
int16 = dtype.int16
int32 = dtype.int32
int64 = dtype.int64
long = dtype.int64
uint8 = dtype.uint8
bool = dtype.bool
