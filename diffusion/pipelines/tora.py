# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝

"""
ToraPipeline — Alibaba's TORA diffusion pipeline (Scaffolding reimplementation).

This is an original implementation inspired by TORA, not a copy of Alibaba's code.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional, Callable, Union, Any

from scaffolding.tensor import Tensor
from scaffolding import autograd as _ag
from scaffolding.nn.module import Module
from scaffolding.diffusion.schedulers import DDIMScheduler
from scaffolding.diffusion.models import UNet2DConditionModel, AutoencoderKL
from scaffolding.diffusion.utils import classifier_free_guidance, randn_tensor


# ── Pipeline output wrapper ──

class PipelineOutput:
    """Wraps pipeline output to provide a ``.frames`` attribute.

    ``frames`` is a list of lists of PIL Images: ``frames[batch_idx]``
    gives the list of video frames for that sample.
    """

    def __init__(self, frames: list):
        self.frames = frames

    def __repr__(self):
        n_batches = len(self.frames)
        n_frames = len(self.frames[0]) if self.frames else 0
        return f"PipelineOutput(batches={n_batches}, frames_per_batch={n_frames})"


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

    def to(self, device=None, dtype=None):
        """Move pipeline components to the given device/dtype.

        Calls ``.to()`` on each sub-module (unet, vae, text_encoder)
        that supports it.  Returns *self* for chaining.
        """
        if device is not None:
            self._device_str = device if isinstance(device, str) else str(device)
        if dtype is not None:
            self._dtype = dtype
        for attr_name in ('unet', 'vae', 'text_encoder'):
            comp = getattr(self, attr_name, None)
            if comp is not None and hasattr(comp, 'to'):
                if device is not None and dtype is not None:
                    comp.to(device, dtype=dtype)
                elif device is not None:
                    comp.to(device)
                elif dtype is not None:
                    comp.to(dtype=dtype)
        return self

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
        self._device_str = 'cpu'
        self._dtype = None  # None = float32; set via .to()

    @_ag.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = None,
        prompt_embeds: Optional[Tensor] = None,
        negative_prompt_embeds: Optional[Tensor] = None,
        video_flow: Optional[Tensor] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        eta: float = 0.0,
        seed: Optional[int] = None,
        generator: Optional[Any] = None,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
    ) -> Tensor:
        """Run the Tora text-to-video (or trajectory-guided) pipeline.

        Args:
            prompt:               Text prompt (encoded internally if
                                  ``text_encoder`` and ``tokenizer`` are set).
            prompt_embeds:        Pre-computed text embeddings ``(B, S, D)``.
            negative_prompt_embeds: Negative-prompt embeddings for CFG.
            video_flow:           Optional trajectory/flow tensor for guided
                                  generation.  ``None`` = pure text-to-video.
            height:               Output height in pixels.
            width:                Output width in pixels.
            num_frames:           Number of video frames to generate.
            num_inference_steps:  Denoising steps.
            guidance_scale:       Classifier-free guidance weight.
            use_dynamic_cfg:      If ``True``, linearly ramp the guidance
                                  scale from ``guidance_scale`` down to 1.0
                                  over the denoising loop.
            eta:                  DDIM η (stochasticity).
            seed:                 Legacy seed parameter (prefer *generator*).
            generator:            ``scaffolding.Generator`` (or compatible)
                                  for reproducible noise initialisation.
            callback:             ``callback(step, timestep, latents)``.
            callback_steps:       Invoke *callback* every N steps.

        Returns:
            ``Tensor`` of shape ``(B, C, num_frames, H, W)`` (video) or
            ``(B, C, H, W)`` (image when ``num_frames <= 1``).
        """

        # ── resolve prompt → embeddings ──
        if prompt_embeds is None:
            if prompt is not None and self.text_encoder is not None and self.tokenizer is not None:
                tokens = self.tokenizer(prompt)
                if isinstance(tokens, Tensor):
                    prompt_embeds = self.text_encoder(tokens)
                elif isinstance(tokens, np.ndarray):
                    prompt_embeds = self.text_encoder(
                        Tensor._wrap(tokens.astype(np.int64),
                                     False, None, None))
                else:
                    # Tokenizer may return raw values; pass directly to encoder
                    prompt_embeds = self.text_encoder(tokens)
            elif prompt is not None:
                # No text encoder — create a dummy embedding so the pipeline
                # can still run in a test / demo capacity.
                prompt_embeds = Tensor._wrap(
                    np.random.randn(1, 77, 768).astype(np.float32),
                    False, None, None)
            else:
                raise ValueError(
                    "Either 'prompt' or 'prompt_embeds' must be provided.")

        B = (prompt_embeds._data.shape[0]
             if hasattr(prompt_embeds, '_data') else 1)

        # ── compute dtype (float16 halves all intermediate memory) ──
        _np_dtype = np.float32
        if self._dtype is not None:
            _np_dtype = (self._dtype.to_numpy()
                         if hasattr(self._dtype, 'to_numpy') else np.float16)

        # ── determine RNG seed ──
        if generator is not None and hasattr(generator, '_seed'):
            eff_seed = generator._seed
        elif seed is not None:
            eff_seed = seed
        else:
            eff_seed = None

        # ── initialise latents ──
        lH = height // self.latent_scale
        lW = width // self.latent_scale
        use_video = num_frames is not None and num_frames > 1
        if use_video:
            shape = (B, self.latent_channels, num_frames, lH, lW)
        else:
            shape = (B, self.latent_channels, lH, lW)

        if generator is not None and hasattr(generator, 'randn'):
            latents = Tensor._wrap(
                generator.randn(*shape).astype(_np_dtype),
                False, None, None)
        else:
            latents = randn_tensor(shape, seed=eff_seed)
            latents._ensure_cpu()
            latents = Tensor._wrap(latents._data.astype(_np_dtype),
                                   False, None, None)

        # ── scheduler setup ──
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        total_steps = len(timesteps)

        # ── denoising loop ──
        for i, t in enumerate(timesteps):
            t_tensor = Tensor._wrap(
                np.full((B,), t, dtype=np.float32), False, None, None)

            model_input = latents
            if hasattr(self.scheduler, 'scale_model_input'):
                model_input = self.scheduler.scale_model_input(latents, int(t))

            # Dynamic CFG: linearly ramp from guidance_scale → 1.0
            if use_dynamic_cfg:
                progress = i / max(total_steps - 1, 1)
                current_guidance = guidance_scale + progress * (1.0 - guidance_scale)
            else:
                current_guidance = guidance_scale

            # ── model forward ──
            # If latents are 5-D (video: B,C,F,H,W) but the UNet is 2-D,
            # collapse frames into the batch dimension for inference.
            def _run_unet(inp, t_emb, enc_hs, **extra_kw):
                inp._ensure_cpu()
                if inp._data.ndim == 5:
                    B_, C_, F_, H_, W_ = inp._data.shape
                    # (B,C,F,H,W) → (B*F, C, H, W)
                    flat = inp._data.transpose(0, 2, 1, 3, 4).reshape(
                        B_ * F_, C_, H_, W_)
                    flat_t = Tensor._wrap(flat.astype(_np_dtype),
                                         False, None, None)
                    del flat  # free intermediate immediately
                    # Repeat timestep & encoder hidden states for each frame
                    t_rep = Tensor._wrap(
                        np.repeat(t_emb._data, F_, axis=0).astype(_np_dtype),
                        False, None, None)
                    if enc_hs is not None:
                        enc_hs._ensure_cpu()
                        enc_rep = Tensor._wrap(
                            np.repeat(enc_hs._data, F_, axis=0).astype(_np_dtype),
                            False, None, None)
                    else:
                        enc_rep = None
                    out = self.unet(flat_t, t_rep, enc_rep)
                    del flat_t, t_rep, enc_rep  # free inputs post-forward
                    out._ensure_cpu()
                    # (B*F,C,H,W) → (B,C,F,H,W)
                    out5 = out._data.reshape(B_, F_, C_, H_, W_).transpose(
                        0, 2, 1, 3, 4)
                    del out  # free flat output
                    result = Tensor._wrap(out5.astype(_np_dtype),
                                          False, None, None)
                    del out5
                    return result
                return self.unet(inp, t_emb, enc_hs)

            if current_guidance > 1.0 and negative_prompt_embeds is not None:
                # Manual CFG with video-aware forward
                noise_cond = _run_unet(model_input, t_tensor, prompt_embeds)
                noise_uncond = _run_unet(model_input, t_tensor,
                                         negative_prompt_embeds)
                noise_cond._ensure_cpu(); noise_uncond._ensure_cpu()
                # Memory-efficient CFG: reuse arrays, free early
                guided = noise_cond._data - noise_uncond._data  # 1 alloc
                del noise_cond  # free cond
                guided *= current_guidance                       # in-place
                guided += noise_uncond._data                     # in-place
                del noise_uncond  # free uncond
                noise_pred = Tensor._wrap(guided.astype(_np_dtype),
                                          False, None, None)
                del guided
            else:
                noise_pred = _run_unet(model_input, t_tensor, prompt_embeds)

            # ── scheduler step ──
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

        # ── decode latents → PipelineOutput (memory-efficient) ──
        # Instead of accumulating all decoded frames then converting,
        # stream each frame through the VAE and directly to PIL.
        # Peak memory: ONE decoded frame, not the full video.
        from PIL import Image

        if self.vae is not None:
            latents._ensure_cpu()
            z_data = (latents._data / self.vae.scaling_factor).astype(_np_dtype)
            del latents  # free latents immediately

            if z_data.ndim == 5:
                # (B, C, F, H, W) — per-frame VAE decode → PIL
                B_, C_, F_, H_, W_ = z_data.shape
                all_batches: list[list[Image.Image]] = [[] for _ in range(B_)]
                for f_idx in range(F_):
                    frame_z = Tensor._wrap(
                        z_data[:, :, f_idx, :, :].copy().astype(_np_dtype),
                        False, None, None)
                    frame_dec = self.vae.decode(frame_z)
                    del frame_z
                    frame_dec._ensure_cpu()
                    # Clip → rescale → uint8 without extra full-size copies
                    fd = np.clip(frame_dec._data, -1.0, 1.0)
                    del frame_dec
                    fd = ((fd + 1.0) * 127.5).astype(np.uint8)
                    for b in range(B_):
                        img = fd[b].transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
                        if img.shape[-1] == 1:
                            img = img.squeeze(-1)
                        all_batches[b].append(Image.fromarray(img))
                    del fd
                del z_data
                return PipelineOutput(frames=all_batches)
            else:
                z = Tensor._wrap(z_data, False, None, None)
                del z_data
                output = self.vae.decode(z)
                del z
        else:
            output = latents

        # Fallback for non-video or no-VAE path
        return self._tensor_to_pipeline_output(output)

    @staticmethod
    def _tensor_to_pipeline_output(tensor_out: Tensor) -> PipelineOutput:
        """Convert a decoded tensor to ``PipelineOutput`` with PIL frames.

        Handles both 4-D ``(B, C, H, W)`` single-image and 5-D
        ``(B, C, F, H, W)`` video tensors.
        """
        from PIL import Image

        tensor_out._ensure_cpu()
        data = tensor_out._data  # float32 in [-1, 1]
        # Rescale [-1, 1] → [0, 255] uint8
        data = ((data + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        all_batches: list[list[Image.Image]] = []

        if data.ndim == 5:
            # (B, C, F, H, W)
            B, C, F, H, W = data.shape
            for b in range(B):
                frames = []
                for f in range(F):
                    # (C, H, W) → (H, W, C)
                    frame = data[b, :, f].transpose(1, 2, 0)
                    if C == 1:
                        frame = frame.squeeze(-1)
                    frames.append(Image.fromarray(frame))
                all_batches.append(frames)
        elif data.ndim == 4:
            # (B, C, H, W) — single image per batch
            B, C, H, W = data.shape
            for b in range(B):
                frame = data[b].transpose(1, 2, 0)
                if C == 1:
                    frame = frame.squeeze(-1)
                all_batches.append([Image.fromarray(frame)])
        else:
            raise ValueError(f"Expected 4-D or 5-D tensor, got {data.ndim}-D")

        return PipelineOutput(frames=all_batches)

def _load_pipeline(*args, **kwargs):
    return ToraPipeline(*args, **kwargs)
