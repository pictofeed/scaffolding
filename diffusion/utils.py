# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝

"""Diffusion utilities — noise helpers, guidance, and schedule builders.

Shared helpers used across schedulers, models, and pipelines:

- ``randn_tensor``              — generate random noise as a Tensor.
- ``classifier_free_guidance``  — run a model with CFG in one call.
- ``rescale_noise_cfg``         — rescale guided noise (Imagen-style).
- ``get_beta_schedule``         — public API for building β schedules.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple, Union, Sequence

from scaffolding.tensor import Tensor


# ═════════════════════════════════════════════════════════════════════
#  Noise generation
# ═════════════════════════════════════════════════════════════════════

def randn_tensor(
    shape: Union[Tuple[int, ...], Sequence[int]],
    seed: Optional[int] = None,
    dtype: np.dtype = np.float32,
    device=None,
) -> Tensor:
    """Generate a Tensor filled with standard normal noise.

    Args:
        shape:  Shape of the output tensor.
        seed:   Optional seed for reproducibility.
        dtype:  NumPy dtype (default ``float32``).
        device: Target device (currently informational).

    Returns:
        A Tensor with i.i.d. N(0, 1) entries.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(dtype)
    return Tensor._wrap(data, False, None, device)


# ═════════════════════════════════════════════════════════════════════
#  Classifier-Free Guidance
# ═════════════════════════════════════════════════════════════════════

def classifier_free_guidance(
    model,
    latents: Tensor,
    timesteps: Tensor,
    prompt_embeds: Tensor,
    negative_prompt_embeds: Tensor,
    guidance_scale: float = 7.5,
) -> Tensor:
    """Run a denoising model with classifier-free guidance.

    Performs two forward passes (conditional + unconditional) and
    combines the predictions::

        guided = uncond + guidance_scale * (cond - uncond)

    Args:
        model:                  The denoising network (UNet or DiT).
        latents:                (B, …) current noisy sample.
        timesteps:              (B,) current diffusion timestep.
        prompt_embeds:          (B, S, D) conditional text embeddings.
        negative_prompt_embeds: (B, S, D) unconditional text embeddings.
        guidance_scale:         CFG weight (1.0 = no guidance).

    Returns:
        (B, …) guided noise prediction.
    """
    # Conditional
    noise_cond = model(latents, timesteps, prompt_embeds)
    # Unconditional
    noise_uncond = model(latents, timesteps, negative_prompt_embeds)

    noise_cond._ensure_cpu()
    noise_uncond._ensure_cpu()

    guided = noise_uncond._data + guidance_scale * (
        noise_cond._data - noise_uncond._data
    )
    return Tensor._wrap(guided.astype(np.float32), False, None,
                        latents._device)


def rescale_noise_cfg(
    noise_pred: Tensor,
    noise_pred_text: Tensor,
    guidance_rescale: float = 0.7,
) -> Tensor:
    """Rescale CFG noise prediction (Imagen-style guidance rescaling).

    Adjusts the magnitude of the guided prediction to be closer to the
    text-only prediction's magnitude, reducing over-saturation artifacts.

    ``noise_out = noise_pred * (std_text / std_pred) * rescale + noise_pred * (1 - rescale)``

    Args:
        noise_pred:       The CFG-guided noise prediction.
        noise_pred_text:  The text-only (no guidance) noise prediction.
        guidance_rescale: Blending factor (0 = no rescaling, 1 = full).

    Returns:
        Rescaled noise prediction tensor.
    """
    noise_pred._ensure_cpu()
    noise_pred_text._ensure_cpu()

    # Per-sample standard deviations
    axes = tuple(range(1, noise_pred._data.ndim))
    std_pred = np.std(noise_pred._data, axis=axes, keepdims=True) + 1e-8
    std_text = np.std(noise_pred_text._data, axis=axes, keepdims=True) + 1e-8

    # Rescaled = pred * (std_text / std_pred)
    rescaled = noise_pred._data * (std_text / std_pred)

    # Blend
    out = (guidance_rescale * rescaled
           + (1.0 - guidance_rescale) * noise_pred._data)

    return Tensor._wrap(out.astype(np.float32), False, None,
                        noise_pred._device)


# ═════════════════════════════════════════════════════════════════════
#  Beta-schedule builder (public API)
# ═════════════════════════════════════════════════════════════════════

def get_beta_schedule(
    schedule: str,
    num_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> np.ndarray:
    """Construct a beta noise schedule.

    Args:
        schedule:       One of ``'linear'``, ``'cosine'``,
                        ``'scaled_linear'``, ``'squaredcos_cap_v2'``.
        num_timesteps:  Number of diffusion timesteps.
        beta_start:     Starting beta value (for linear / scaled_linear).
        beta_end:       Ending beta value.

    Returns:
        1-D float32 numpy array of length ``num_timesteps``.
    """
    if schedule == 'linear':
        return np.linspace(beta_start, beta_end, num_timesteps,
                           dtype=np.float32)
    elif schedule == 'cosine':
        steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
        alpha_bar = np.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return np.clip(betas, 0.0001, 0.9999).astype(np.float32)
    elif schedule == 'scaled_linear':
        return (np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                            num_timesteps, dtype=np.float32) ** 2)
    elif schedule == 'squaredcos_cap_v2':
        steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
        alpha_bar = np.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return np.clip(betas, 0.0, 0.999).astype(np.float32)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule!r}")


# ═════════════════════════════════════════════════════════════════════
#  Exports
# ═════════════════════════════════════════════════════════════════════

__all__ = [
    'randn_tensor',
    'classifier_free_guidance',
    'rescale_noise_cfg',
    'get_beta_schedule',
]
