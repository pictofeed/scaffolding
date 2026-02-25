# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝

"""
ToraPipeline — Alibaba's TORA diffusion pipeline (Scaffolding reimplementation).

This is an original implementation inspired by TORA, not a copy of Alibaba's code.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Callable

from scaffolding.tensor import Tensor
from scaffolding import autograd as _ag
from scaffolding.nn.module import Module
from scaffolding.diffusion.schedulers import DDIMScheduler
from scaffolding.diffusion.models import UNet2DConditionModel, AutoencoderKL
from scaffolding.diffusion.utils import classifier_free_guidance, randn_tensor

class ToraPipeline:
    
    @classmethod
    def from_pretrained(cls, model_path, dtype=None, **kwargs):
        """
        Load a ToraPipeline from a pretrained model directory.
        Args:
            model_path: Path to the pretrained model directory.
            dtype: Optional dtype for model weights.
            kwargs: Additional arguments for pipeline construction.
        Returns:
            ToraPipeline instance with loaded weights.
        """
        unet = UNet2DConditionModel.from_pretrained(model_path, dtype=dtype) if hasattr(UNet2DConditionModel, 'from_pretrained') else UNet2DConditionModel()
        vae = AutoencoderKL.from_pretrained(model_path, dtype=dtype) if hasattr(AutoencoderKL, 'from_pretrained') else None
        scheduler = DDIMScheduler.from_pretrained(model_path) if hasattr(DDIMScheduler, 'from_pretrained') else DDIMScheduler()
        text_encoder = None
        tokenizer = None
        return cls(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            **kwargs
        )

    def enable_sequential_cpu_offload(self):
        """No-op for compatibility with memory optimization APIs."""
        pass

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: Optional[DDIMScheduler] = None,
        vae: Optional[AutoencoderKL] = None,
        text_encoder: Optional[Module] = None,
        tokenizer: Optional[Callable] = None,
        latent_channels: int = 4,
        latent_scale: int = 8,
    ):
        self.unet = unet
        self.scheduler = scheduler or DDIMScheduler()
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
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        eta: float = 0.0,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ) -> Tensor:
        if prompt_embeds is None:
            raise ValueError("prompt_embeds is required (or set text_encoder + tokenizer)")
        B = prompt_embeds._data.shape[0] if hasattr(prompt_embeds, '_data') else 1
        lH = height // self.latent_scale
        lW = width // self.latent_scale
        latents = randn_tensor((B, self.latent_channels, lH, lW), seed=seed)
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
                    self.unet, model_input, t_tensor,
                    prompt_embeds, negative_prompt_embeds,
                    guidance_scale)
            else:
                noise_pred = self.unet(model_input, t_tensor, prompt_embeds)
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
        if self.vae is not None:
            latents._ensure_cpu()
            z = Tensor._wrap(
                (latents._data / self.vae.scaling_factor).astype(np.float32),
                False, None, latents._device)
            images = self.vae.decode(z)
        else:
            images = latents
        images._ensure_cpu()
        images = Tensor._wrap(
            np.clip(images._data, -1.0, 1.0).astype(np.float32),
            False, None, None)
        return images

def _load_pipeline(*args, **kwargs):
    return ToraPipeline(*args, **kwargs)
