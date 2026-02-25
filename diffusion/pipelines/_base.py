# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝

"""Diffusion pipelines — end-to-end generation abstractions.

Provides ready-to-use pipeline classes that orchestrate a VAE,
text encoder, denoising model, and noise scheduler into a coherent
generation workflow.

- **DiffusionPipeline** — base class with shared logic (noise init,
  denoising loop, guidance).
- **StableDiffusionPipeline** — SD-style latent-diffusion pipeline
  with UNet2DConditionModel.
- **CogVideoXPipeline** — CogVideoX video generation pipeline using
  CogVideoXDPMScheduler and a 3-D transformer / UNet backbone.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional, Callable, Union

from scaffolding.tensor import Tensor
from scaffolding import autograd as _ag
from scaffolding.nn.module import Module

from scaffolding.diffusion.schedulers import (
    DDIMScheduler,
    CogVideoXDPMScheduler,
)
from scaffolding.diffusion.models import (
    UNet2DConditionModel,
    AutoencoderKL,
)
from scaffolding.diffusion.utils import classifier_free_guidance, randn_tensor


# ═════════════════════════════════════════════════════════════════════
#  DiffusionPipeline — base class
# ═════════════════════════════════════════════════════════════════════

class DiffusionPipeline:
    """Base class for diffusion generation pipelines.

    Subclasses should set ``self.scheduler``, ``self.unet`` (or
    ``self.transformer``), and optionally ``self.vae`` / ``self.text_encoder``.

    Provides:
    - ``prepare_latents`` — initialise / scale starting noise.
    - ``decode_latents`` — run VAE decoder on denoised latents.
    - ``__call__`` — the main generation entry point (overridden by subclasses).
    """

    scheduler: object
    unet: Optional[Module] = None
    transformer: Optional[Module] = None
    vae: Optional[AutoencoderKL] = None

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        seed: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> Tensor:
        """Create initial noise latents."""
        if seed is not None:
            np.random.seed(seed)

        if num_frames is not None:
            shape = (batch_size, num_channels, num_frames, height, width)
        else:
            shape = (batch_size, num_channels, height, width)

        latents = np.random.randn(*shape).astype(np.float32)

        # Scale by scheduler init_noise_sigma if available
        init_sigma = getattr(self.scheduler, 'init_noise_sigma', 1.0)
        latents *= init_sigma

        return Tensor._wrap(latents, False, None, None)

    def decode_latents(self, latents: Tensor) -> Tensor:
        """Decode latent-space tensor through the VAE."""
        if self.vae is not None:
            latents._ensure_cpu()
            z = Tensor._wrap(
                (latents._data / self.vae.scaling_factor).astype(np.float32),
                False, None, latents._device)
            return self.vae.decode(z)
        return latents

    def progress_bar(self, iterable, desc: str = ''):
        """Wrap an iterable with an optional progress bar."""
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable, desc=desc)
        except ImportError:
            return iterable

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement __call__")


# ═════════════════════════════════════════════════════════════════════
#  StableDiffusionPipeline
# ═════════════════════════════════════════════════════════════════════

class StableDiffusionPipeline(DiffusionPipeline):
    """Stable Diffusion latent-diffusion pipeline.

    A minimal but functional implementation of the SD inference loop:

    1. Encode prompt → text embeddings.
    2. Initialise latent noise.
    3. Iterative denoising with classifier-free guidance.
    4. Decode latents through VAE.

    Args:
        unet:           UNet2DConditionModel instance.
        scheduler:      Any compatible noise scheduler.
        vae:            AutoencoderKL instance (or None for pixel-space).
        text_encoder:   A Module that maps token-IDs → embeddings, or None
                        to pass pre-computed embeddings directly.
        tokenizer:      Callable that maps str → token-ID Tensor, or None.
        latent_channels: Channels in latent space (4 for SD).
        latent_scale:   Spatial scale factor of the VAE (8 for SD).
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler,
        vae: Optional[AutoencoderKL] = None,
        text_encoder: Optional[Module] = None,
        tokenizer: Optional[Callable] = None,
        latent_channels: int = 4,
        latent_scale: int = 8,
    ):
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.latent_channels = latent_channels
        self.latent_scale = latent_scale

    @_ag.no_grad()
    def __call__(
        self,
        prompt_embeds: Optional[Tensor] = None,
        negative_prompt_embeds: Optional[Tensor] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ) -> Tensor:
        """Run the Stable Diffusion generation loop.

        Pass pre-computed ``prompt_embeds`` (B, S, dim) directly if no
        text_encoder / tokenizer are configured.

        Returns:
            Tensor of shape (B, 3, H, W) with pixel values in [-1, 1].
        """
        if prompt_embeds is None:
            raise ValueError("prompt_embeds is required "
                             "(or set text_encoder + tokenizer)")

        B = prompt_embeds._data.shape[0] if hasattr(prompt_embeds, '_data') else 1

        # Latent dimensions
        lH = height // self.latent_scale
        lW = width // self.latent_scale

        # Initialise latents
        latents = self.prepare_latents(
            B, self.latent_channels, lH, lW, seed=seed)

        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Denoising loop
        for i, t in enumerate(timesteps):
            t_tensor = Tensor._wrap(
                np.full((B,), t, dtype=np.float32), False, None, None)

            # Scale model input if scheduler requires it
            model_input = latents
            if hasattr(self.scheduler, 'scale_model_input'):
                model_input = self.scheduler.scale_model_input(latents, int(t))

            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                noise_pred = classifier_free_guidance(
                    self.unet, model_input, t_tensor,
                    prompt_embeds, negative_prompt_embeds,
                    guidance_scale)
            else:
                noise_pred = self.unet(model_input, t_tensor, prompt_embeds)

            # Scheduler step
            if hasattr(self.scheduler, 'step'):
                import inspect
                sig = inspect.signature(self.scheduler.step)
                if 'eta' in sig.parameters:
                    latents = self.scheduler.step(
                        noise_pred, int(t), latents, eta=eta)
                else:
                    latents = self.scheduler.step(
                        noise_pred, int(t), latents)

            if callback is not None and (i + 1) % callback_steps == 0:
                callback(i + 1, t, latents)

        # Decode
        images = self.decode_latents(latents)
        images._ensure_cpu()
        images = Tensor._wrap(
            np.clip(images._data, -1.0, 1.0).astype(np.float32),
            False, None, None)
        return images


# ═════════════════════════════════════════════════════════════════════
#  CogVideoXPipeline
# ═════════════════════════════════════════════════════════════════════

class CogVideoXPipeline(DiffusionPipeline):
    """CogVideoX video generation pipeline.

    Orchestrates a 3-D denoising model (UNet3D or DiT) with the
    CogVideoXDPMScheduler for temporally-coherent video generation.

    Args:
        transformer:     3-D denoising model (e.g. from ``nn.video``).
        scheduler:       CogVideoXDPMScheduler instance.
        vae:             Video VAE (3-D encoder/decoder), or None.
        text_encoder:    Text encoder module, or None.
        tokenizer:       Text tokenizer callable, or None.
        num_frames:      Number of output video frames.
        latent_channels: Latent space channels.
    """

    def __init__(
        self,
        transformer: Module,
        scheduler: Optional[CogVideoXDPMScheduler] = None,
        vae: Optional[Module] = None,
        text_encoder: Optional[Module] = None,
        tokenizer: Optional[Callable] = None,
        num_frames: int = 49,
        latent_channels: int = 16,
    ):
        self.transformer = transformer
        self.scheduler = scheduler or CogVideoXDPMScheduler()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.latent_channels = latent_channels

    @_ag.no_grad()
    def __call__(
        self,
        prompt_embeds: Optional[Tensor] = None,
        negative_prompt_embeds: Optional[Tensor] = None,
        height: int = 480,
        width: int = 720,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ) -> Tensor:
        """Generate a video from prompt embeddings.

        Args:
            prompt_embeds:           (B, S, D) text embeddings.
            negative_prompt_embeds:  (B, S, D) negative text embeddings.
            height / width:          Output resolution.
            num_frames:              Number of video frames.
            num_inference_steps:     Denoising steps.
            guidance_scale:          Classifier-free guidance strength.
            seed:                    Random seed.

        Returns:
            (B, C, T, H, W) video tensor in [-1, 1].
        """
        if prompt_embeds is None:
            raise ValueError("prompt_embeds is required")

        T = num_frames or self.num_frames
        B = prompt_embeds._data.shape[0]

        # Latent spatial dimensions (assume 8× compression)
        lH = height // 8
        lW = width // 8

        # Initialise latents
        latents = self.prepare_latents(
            B, self.latent_channels, lH, lW, seed=seed, num_frames=T)

        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            t_tensor = Tensor._wrap(
                np.full((B,), t, dtype=np.float32), False, None, None)

            model_input = latents
            if hasattr(self.scheduler, 'scale_model_input'):
                model_input = self.scheduler.scale_model_input(latents, int(t))

            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                noise_pred = classifier_free_guidance(
                    self.transformer, model_input, t_tensor,
                    prompt_embeds, negative_prompt_embeds,
                    guidance_scale)
            else:
                noise_pred = self.transformer(
                    model_input, t_tensor, prompt_embeds)

            latents = self.scheduler.step(noise_pred, int(t), latents)

            if callback is not None and (i + 1) % callback_steps == 0:
                callback(i + 1, t, latents)

        # Decode
        video = self.decode_latents(latents)
        video._ensure_cpu()
        video = Tensor._wrap(
            np.clip(video._data, -1.0, 1.0).astype(np.float32),
            False, None, None)
        return video


# ═════════════════════════════════════════════════════════════════════
#  Exports
# ═════════════════════════════════════════════════════════════════════

__all__ = [
    'DiffusionPipeline',
    'StableDiffusionPipeline',
    'CogVideoXPipeline',
]
