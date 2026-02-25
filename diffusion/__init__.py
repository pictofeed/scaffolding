# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""scaffolding.diffusion — Diffusion models, schedulers, and pipelines.

A comprehensive diffusion modelling package providing noise schedulers
(DDPM, DDIM, DPM-Solver++, CogVideoX-DPM, Euler, PNDM, Flow Matching),
model architectures (UNet2D, DiT, AutoencoderKL), and ready-to-use
generation pipelines.

Usage::

    import scaffolding as torch
    from scaffolding.diffusion import (
        DDPMScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        CogVideoXDPMScheduler,
        EulerDiscreteScheduler,
        PNDMScheduler,
        FlowMatchEulerDiscreteScheduler,
        DiffusionPipeline,
        StableDiffusionPipeline,
        CogVideoXPipeline,
        UNet2DConditionModel,
        DiTModel,
        AutoencoderKL,
    )
"""
from __future__ import annotations

# ── Schedulers ──
from .schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    CogVideoXDPMScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    FlowMatchEulerDiscreteScheduler,
)

# ── Models ──
from .models import (
    UNet2DConditionModel,
    DiTModel,
    AutoencoderKL,
)

# ── Pipelines ──
from .pipelines import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    CogVideoXPipeline,
)

# ── Utilities ──
from .utils import (
    classifier_free_guidance,
    rescale_noise_cfg,
    randn_tensor,
    get_beta_schedule,
)

__all__ = [
    # Schedulers
    'DDPMScheduler',
    'DDIMScheduler',
    'DPMSolverMultistepScheduler',
    'CogVideoXDPMScheduler',
    'EulerDiscreteScheduler',
    'PNDMScheduler',
    'FlowMatchEulerDiscreteScheduler',
    # Models
    'UNet2DConditionModel',
    'DiTModel',
    'AutoencoderKL',
    # Pipelines
    'DiffusionPipeline',
    'StableDiffusionPipeline',
    'CogVideoXPipeline',
    # Utilities
    'classifier_free_guidance',
    'rescale_noise_cfg',
    'randn_tensor',
    'get_beta_schedule',
]
