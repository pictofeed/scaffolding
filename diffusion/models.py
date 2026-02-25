# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Diffusion model architectures — UNet2D, DiT, AutoencoderKL.

Provides the core neural-network building blocks used by diffusion
and flow-matching generative models:

- **UNet2DConditionModel** — 2-D UNet with cross-attention conditioning
  (Stable Diffusion style).
- **DiTModel** — Diffusion Transformer (Peebles & Xie 2023), also the
  backbone of CogVideoX transformers.
- **AutoencoderKL** — Variational autoencoder with KL regularisation
  for latent diffusion.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple

from ..tensor import Tensor
from .. import autograd as _ag
from ..nn.module import Module, ModuleList, Sequential
from ..nn.parameter import Parameter
from ..nn.layers import (
    Linear, Conv2d, Conv3d, ConvTranspose2d,
    GroupNorm, LayerNorm, SiLU, Dropout, Embedding,
    _param_to_device, _default_param_device,
)


class AutoencoderKL(Module):
    # ...existing code...

    def enable_slicing(self):
        """No-op for compatibility with memory optimization APIs."""
        pass

    def enable_tiling(self):
        """No-op for compatibility with memory optimization APIs."""
        pass

def _valid_groups(channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ═════════════════════════════════════════════════════════════════════
#  Timestep embedding
# ═════════════════════════════════════════════════════════════════════

class SinusoidalTimestepEmbedding(Module):
    """Sinusoidal timestep positional encoding (Vaswani / DDPM)."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps._ensure_cpu()
        t = timesteps._data.astype(np.float32).ravel()
        half = self.dim // 2
        freqs = np.exp(
            -math.log(self.max_period)
            * np.arange(half, dtype=np.float32) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        if self.dim % 2:
            emb = np.pad(emb, ((0, 0), (0, 1)))
        return Tensor._wrap(emb.astype(np.float32), False, None,
                            timesteps._device)


class TimestepMLP(Module):
    """MLP to project sinusoidal timestep embeddings."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = Linear(in_dim, out_dim)
        self.act = SiLU()
        self.linear2 = Linear(out_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.act(self.linear1(x)))


# ═════════════════════════════════════════════════════════════════════
#  Attention block (2-D, for UNet)
# ═════════════════════════════════════════════════════════════════════

class CrossAttentionBlock(Module):
    """Multi-head cross-attention with self-attention + cross-attention + FFN.

    Used inside UNet encoder / decoder blocks for text conditioning.
    """

    def __init__(self, channels: int, context_dim: int,
                 num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Norms
        self.norm1 = LayerNorm(channels)
        self.norm2 = LayerNorm(channels)
        self.norm_ff = LayerNorm(channels)

        # Self-attention
        self.to_qkv = Linear(channels, channels * 3, bias=False)
        self.self_out = Linear(channels, channels)

        # Cross-attention
        self.cross_q = Linear(channels, channels, bias=False)
        self.cross_k = Linear(context_dim, channels, bias=False)
        self.cross_v = Linear(context_dim, channels, bias=False)
        self.cross_out = Linear(channels, channels)

        # Feed-forward
        self.ff = Sequential(
            Linear(channels, channels * 4),
            SiLU(),
            Dropout(dropout),
            Linear(channels * 4, channels),
        )

    def _attn(self, q: np.ndarray, k: np.ndarray,
              v: np.ndarray) -> np.ndarray:
        """Scaled dot-product attention (numpy)."""
        B, H, N, D = q.shape
        scale = 1.0 / math.sqrt(D)
        scores = np.einsum('bhnd,bhmd->bhnm', q, k) * scale
        # Softmax
        scores -= scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-9)
        return np.einsum('bhnm,bhmd->bhnd', attn, v)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        x:       (B, N, C) — flattened spatial features.
        context: (B, S, context_dim) — text/conditioning embeddings.
        """
        x._ensure_cpu()
        B, N, C = x._data.shape
        H, D = self.num_heads, self.head_dim

        # ---------- Self-attention ----------
        h = self.norm1(x)
        qkv = self.to_qkv(h)
        qkv._ensure_cpu()
        q, k, v = [
            qkv._data[:, :, i * C:(i + 1) * C].reshape(B, N, H, D)
                .transpose(0, 2, 1, 3).copy()
            for i in range(3)
        ]
        sa = self._attn(q, k, v)  # (B, H, N, D)
        sa = sa.transpose(0, 2, 1, 3).reshape(B, N, C).copy()
        sa_t = Tensor._wrap(sa.astype(np.float32), False, None, x._device)
        sa_out = self.self_out(sa_t)
        sa_out._ensure_cpu(); x._ensure_cpu()
        x = Tensor._wrap((x._data + sa_out._data).astype(np.float32),
                         False, None, x._device)

        # ---------- Cross-attention ----------
        if context is not None:
            h2 = self.norm2(x)
            cq = self.cross_q(h2)
            ck = self.cross_k(context)
            cv = self.cross_v(context)
            cq._ensure_cpu(); ck._ensure_cpu(); cv._ensure_cpu()
            S = ck._data.shape[1]

            cq_h = cq._data.reshape(B, N, H, D).transpose(0, 2, 1, 3).copy()
            ck_h = ck._data.reshape(B, S, H, D).transpose(0, 2, 1, 3).copy()
            cv_h = cv._data.reshape(B, S, H, D).transpose(0, 2, 1, 3).copy()

            ca = self._attn(cq_h, ck_h, cv_h)
            ca = ca.transpose(0, 2, 1, 3).reshape(B, N, C).copy()
            ca_t = Tensor._wrap(ca.astype(np.float32), False, None, x._device)
            ca_out = self.cross_out(ca_t)
            ca_out._ensure_cpu(); x._ensure_cpu()
            x = Tensor._wrap((x._data + ca_out._data).astype(np.float32),
                             False, None, x._device)

        # ---------- Feed-forward ----------
        ff_in = self.norm_ff(x)
        ff_out = self.ff(ff_in)
        ff_out._ensure_cpu(); x._ensure_cpu()
        x = Tensor._wrap((x._data + ff_out._data).astype(np.float32),
                         False, None, x._device)
        return x


# ═════════════════════════════════════════════════════════════════════
#  ResBlock2D
# ═════════════════════════════════════════════════════════════════════

class ResBlock2D(Module):
    """Residual block with 2-D convolutions + timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = GroupNorm(_valid_groups(in_ch), in_ch)
        self.conv1 = Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = GroupNorm(_valid_groups(out_ch), out_ch)
        self.conv2 = Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = SiLU()
        self.dropout = Dropout(dropout)
        self.time_mlp = Sequential(SiLU(), Linear(time_dim, out_ch))
        self.skip = Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Timestep conditioning: (B, out_ch) → (B, out_ch, 1, 1)
        t = self.time_mlp(t_emb)
        t._ensure_cpu(); h._ensure_cpu()
        h = Tensor._wrap(
            (h._data + t._data[:, :, np.newaxis, np.newaxis]).astype(np.float32),
            False, None, h._device)

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        if self.skip is not None:
            x = self.skip(x)
        h._ensure_cpu(); x._ensure_cpu()
        return Tensor._wrap((h._data + x._data).astype(np.float32),
                            False, None, h._device)


# ═════════════════════════════════════════════════════════════════════
#  Downsample2D / Upsample2D
# ═════════════════════════════════════════════════════════════════════

class Downsample2D(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d(channels, channels, kernel_size=3,
                           stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample2D(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x._ensure_cpu()
        B, C, H, W = x._data.shape
        # Nearest-neighbor upsample 2×
        up = x._data.repeat(2, axis=2).repeat(2, axis=3)
        x_up = Tensor._wrap(up.astype(np.float32), False, None, x._device)
        return self.conv(x_up)


# ═════════════════════════════════════════════════════════════════════
#  UNet2DConditionModel
# ═════════════════════════════════════════════════════════════════════

class UNet2DConditionModel(Module):
    """2-D UNet with cross-attention conditioning (Stable Diffusion).

    Architecture::

        Input  → Conv → [Encoder blocks + Down] × L
                      → [Mid block]
                      → [Decoder blocks + Up + Skip-cat] × L
                      → Norm → Act → Conv → Output

    Args:
        in_channels:    Input image / latent channels.
        out_channels:   Output channels (same as in for noise prediction).
        model_channels: Base channel count.
        context_dim:    Dimension of cross-attention context (text embeddings).
        channel_mult:   Per-level channel multiplier tuple.
        num_res_blocks: Number of residual blocks per level.
        attention_levels: Which levels get cross-attention (0-indexed).
        num_heads:      Multi-head attention heads.
        dropout:        Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        context_dim: int = 768,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_levels: Tuple[int, ...] = (0, 1, 2, 3),
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        time_dim = model_channels * 4

        # Timestep MLP
        self.time_embed = SinusoidalTimestepEmbedding(model_channels)
        self.time_proj = TimestepMLP(model_channels, time_dim)

        # Input conv
        self.input_conv = Conv2d(in_channels, model_channels, 3, padding=1)

        # ---- Encoder ----
        self.encoder_blocks = ModuleList()
        self.downsamplers = ModuleList()
        ch = model_channels
        self._enc_channels: list[int] = [ch]

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock2D(ch, out_ch, time_dim, dropout))
                if level in attention_levels:
                    self.encoder_blocks.append(
                        CrossAttentionBlock(out_ch, context_dim,
                                            num_heads, dropout))
                ch = out_ch
                self._enc_channels.append(ch)
            if level < len(channel_mult) - 1:
                self.downsamplers.append(Downsample2D(ch))
                self._enc_channels.append(ch)

        # ---- Bottleneck ----
        self.mid_block1 = ResBlock2D(ch, ch, time_dim, dropout)
        self.mid_attn = CrossAttentionBlock(ch, context_dim, num_heads, dropout)
        self.mid_block2 = ResBlock2D(ch, ch, time_dim, dropout)

        # ---- Decoder ----
        self.decoder_blocks = ModuleList()
        self.upsamplers = ModuleList()

        for level in reversed(range(len(channel_mult))):
            mult = channel_mult[level]
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = self._enc_channels.pop()
                self.decoder_blocks.append(
                    ResBlock2D(ch + skip_ch, out_ch, time_dim, dropout))
                if level in attention_levels:
                    self.decoder_blocks.append(
                        CrossAttentionBlock(out_ch, context_dim,
                                            num_heads, dropout))
                ch = out_ch
            if level > 0:
                self.upsamplers.append(Upsample2D(ch))

        # ---- Output ----
        self.out_norm = GroupNorm(_valid_groups(ch), ch)
        self.out_act = SiLU()
        self.out_conv = Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x: Tensor, timesteps: Tensor,
                encoder_hidden_states: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:                      (B, C, H, W) noisy latent.
            timesteps:              (B,) diffusion timestep.
            encoder_hidden_states:  (B, S, context_dim) text embeddings.

        Returns:
            (B, C, H, W) predicted noise (or v, or sample).
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        t_emb = self.time_proj(t_emb)

        h = self.input_conv(x)
        skips = [h]

        # Encoder
        block_idx = 0
        down_idx = 0
        channel_mult_len = (len(self._enc_channels) > 0)  # just for iteration

        for blk in self.encoder_blocks:
            if isinstance(blk, ResBlock2D):
                h = blk(h, t_emb)
            elif isinstance(blk, CrossAttentionBlock):
                # Flatten spatial → sequence
                h._ensure_cpu()
                B, C, H2, W2 = h._data.shape
                h_seq = Tensor._wrap(
                    h._data.reshape(B, C, -1).transpose(0, 2, 1)
                    .copy().astype(np.float32),
                    False, None, h._device)
                h_seq = blk(h_seq, encoder_hidden_states)
                h_seq._ensure_cpu()
                h = Tensor._wrap(
                    h_seq._data.transpose(0, 2, 1).reshape(B, C, H2, W2)
                    .copy().astype(np.float32),
                    False, None, h._device)
            skips.append(h)

        # Downsamplers (interleave)
        # Simplified: apply downsamplers at right positions
        # (The blocks above already append; downsamplers go between levels)
        # For simplicity we post-apply downsamplers
        for ds in self.downsamplers:
            h = ds(h)
            skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h._ensure_cpu()
        B, C, H2, W2 = h._data.shape
        h_seq = Tensor._wrap(
            h._data.reshape(B, C, -1).transpose(0, 2, 1)
            .copy().astype(np.float32), False, None, h._device)
        h_seq = self.mid_attn(h_seq, encoder_hidden_states)
        h_seq._ensure_cpu()
        h = Tensor._wrap(
            h_seq._data.transpose(0, 2, 1).reshape(B, C, H2, W2)
            .copy().astype(np.float32), False, None, h._device)
        h = self.mid_block2(h, t_emb)

        # Decoder
        up_idx = 0
        for blk in self.decoder_blocks:
            if isinstance(blk, ResBlock2D):
                skip = skips.pop() if skips else h
                h._ensure_cpu(); skip._ensure_cpu()
                # Spatial size matching
                if h._data.shape[2:] != skip._data.shape[2:]:
                    min_h = min(h._data.shape[2], skip._data.shape[2])
                    min_w = min(h._data.shape[3], skip._data.shape[3])
                    h = Tensor._wrap(h._data[:, :, :min_h, :min_w].copy(),
                                     False, None, h._device)
                    skip = Tensor._wrap(skip._data[:, :, :min_h, :min_w].copy(),
                                        False, None, h._device)
                cat = np.concatenate([h._data, skip._data], axis=1)
                h = Tensor._wrap(cat.astype(np.float32), False, None, h._device)
                h = blk(h, t_emb)
            elif isinstance(blk, CrossAttentionBlock):
                h._ensure_cpu()
                B, C, H2, W2 = h._data.shape
                h_seq = Tensor._wrap(
                    h._data.reshape(B, C, -1).transpose(0, 2, 1)
                    .copy().astype(np.float32), False, None, h._device)
                h_seq = blk(h_seq, encoder_hidden_states)
                h_seq._ensure_cpu()
                h = Tensor._wrap(
                    h_seq._data.transpose(0, 2, 1).reshape(B, C, H2, W2)
                    .copy().astype(np.float32), False, None, h._device)

        for us in self.upsamplers:
            h = us(h)

        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)


# ═════════════════════════════════════════════════════════════════════
#  DiTModel — Diffusion Transformer
# ═════════════════════════════════════════════════════════════════════

class AdaLayerNorm(Module):
    """Adaptive Layer Norm — shifts and scales conditioned on timestep."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.linear = Linear(cond_dim, dim * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.norm(x)
        params = self.linear(cond)
        params._ensure_cpu(); h._ensure_cpu()
        C = h._data.shape[-1]
        scale = params._data[..., :C] + 1.0   # shift-by-1 so default is identity
        shift = params._data[..., C:]
        return Tensor._wrap(
            (h._data * scale + shift).astype(np.float32),
            False, None, h._device)


class DiTBlock(Module):
    """Single Diffusion Transformer block with adaptive layer norm.

    Structure:
        adaLN → self-attention → residual
        adaLN → cross-attention → residual  (if context provided)
        adaLN → FFN → residual
    """

    def __init__(self, dim: int, num_heads: int, cond_dim: int,
                 context_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Self-attention
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.to_qkv = Linear(dim, dim * 3, bias=False)
        self.self_out = Linear(dim, dim)

        # Cross-attention (optional)
        self.has_cross = context_dim > 0
        if self.has_cross:
            self.norm2 = AdaLayerNorm(dim, cond_dim)
            self.cross_q = Linear(dim, dim, bias=False)
            self.cross_k = Linear(context_dim, dim, bias=False)
            self.cross_v = Linear(context_dim, dim, bias=False)
            self.cross_out = Linear(dim, dim)

        # FFN
        self.norm3 = AdaLayerNorm(dim, cond_dim)
        self.ff = Sequential(
            Linear(dim, dim * 4),
            SiLU(),
            Dropout(dropout),
            Linear(dim * 4, dim),
        )

    def _attn(self, q, k, v):
        B, H, N, D = q.shape
        scale = 1.0 / math.sqrt(D)
        scores = np.einsum('bhnd,bhmd->bhnm', q, k) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-9)
        return np.einsum('bhnm,bhmd->bhnd', attn, v)

    def forward(self, x: Tensor, cond: Tensor,
                context: Optional[Tensor] = None) -> Tensor:
        """
        x:       (B, N, dim)
        cond:    (B, cond_dim) — timestep + class/label embedding
        context: (B, S, context_dim) — optional cross-attention context
        """
        B = x._data.shape[0] if hasattr(x, '_data') else 1
        H, D = self.num_heads, self.head_dim
        C = self.dim

        # ---- self-attention ----
        h = self.norm1(x, cond)
        qkv = self.to_qkv(h)
        qkv._ensure_cpu()
        N = qkv._data.shape[1]
        q, k, v = [
            qkv._data[:, :, i * C:(i + 1) * C]
               .reshape(B, N, H, D).transpose(0, 2, 1, 3).copy()
            for i in range(3)
        ]
        sa = self._attn(q, k, v).transpose(0, 2, 1, 3).reshape(B, N, C).copy()
        sa_t = self.self_out(Tensor._wrap(sa.astype(np.float32),
                                          False, None, x._device))
        sa_t._ensure_cpu(); x._ensure_cpu()
        x = Tensor._wrap((x._data + sa_t._data).astype(np.float32),
                         False, None, x._device)

        # ---- cross-attention ----
        if self.has_cross and context is not None:
            h2 = self.norm2(x, cond)
            cq = self.cross_q(h2); ck = self.cross_k(context)
            cv = self.cross_v(context)
            cq._ensure_cpu(); ck._ensure_cpu(); cv._ensure_cpu()
            S = ck._data.shape[1]
            cq_h = cq._data.reshape(B, N, H, D).transpose(0, 2, 1, 3).copy()
            ck_h = ck._data.reshape(B, S, H, D).transpose(0, 2, 1, 3).copy()
            cv_h = cv._data.reshape(B, S, H, D).transpose(0, 2, 1, 3).copy()
            ca = self._attn(cq_h, ck_h, cv_h)\
                .transpose(0, 2, 1, 3).reshape(B, N, C).copy()
            ca_out = self.cross_out(
                Tensor._wrap(ca.astype(np.float32), False, None, x._device))
            ca_out._ensure_cpu(); x._ensure_cpu()
            x = Tensor._wrap((x._data + ca_out._data).astype(np.float32),
                             False, None, x._device)

        # ---- FFN ----
        ff_in = self.norm3(x, cond)
        ff_out = self.ff(ff_in)
        ff_out._ensure_cpu(); x._ensure_cpu()
        x = Tensor._wrap((x._data + ff_out._data).astype(np.float32),
                         False, None, x._device)
        return x


class DiTModel(Module):
    """Diffusion Transformer (DiT) — Peebles & Xie 2023.

    A pure-transformer architecture for diffusion generative models.
    Replaces the UNet with a stack of DiT blocks using adaptive layer
    norm for timestep conditioning and optional cross-attention for
    text / class conditioning.

    Also serves as the backbone for CogVideoX-style transformer
    architectures when used with 3-D patch embeddings.

    Args:
        patch_size:    Size of each image patch.
        in_channels:   Input channels (latent or pixel).
        hidden_size:   Transformer hidden dimension.
        depth:         Number of DiT blocks.
        num_heads:     Attention heads.
        context_dim:   Cross-attention context dimension (0 = none).
        num_classes:   If > 0, use class-conditional label embedding.
        dropout:       Dropout rate.
        learn_sigma:   If True, output 2× channels (mean + variance).
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        context_dim: int = 0,
        num_classes: int = 0,
        dropout: float = 0.0,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        # Patch embedding (linear projection of flattened patches)
        patch_dim = in_channels * patch_size * patch_size
        self.patch_proj = Linear(patch_dim, hidden_size)

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(hidden_size)
        cond_dim = hidden_size

        # Optional class embedding
        self.num_classes = num_classes
        if num_classes > 0:
            self.class_embed = Embedding(num_classes, hidden_size)
            cond_dim = hidden_size  # time + class are summed

        # Transformer blocks
        self.blocks = ModuleList()
        for _ in range(depth):
            self.blocks.append(
                DiTBlock(hidden_size, num_heads, cond_dim,
                         context_dim, dropout))

        # Final layer
        self.final_norm = LayerNorm(hidden_size)
        self.final_proj = Linear(hidden_size, patch_dim
                                 if not learn_sigma
                                 else in_channels * 2 * patch_size * patch_size)

    def _patchify(self, x: np.ndarray) -> np.ndarray:
        """(B, C, H, W) → (B, N, patch_dim)."""
        B, C, H, W = x.shape
        p = self.patch_size
        nH, nW = H // p, W // p
        x = x.reshape(B, C, nH, p, nW, p)
        x = x.transpose(0, 2, 4, 1, 3, 5)  # (B, nH, nW, C, p, p)
        return x.reshape(B, nH * nW, C * p * p).copy()

    def _unpatchify(self, x: np.ndarray, H: int, W: int) -> np.ndarray:
        """(B, N, patch_dim) → (B, C, H, W)."""
        p = self.patch_size
        C_out = self.out_channels
        nH, nW = H // p, W // p
        x = x.reshape(-1, nH, nW, C_out, p, p)
        x = x.transpose(0, 3, 1, 4, 2, 5)  # (B, C, nH, p, nW, p)
        return x.reshape(-1, C_out, H, W).copy()

    def forward(self, x: Tensor, timesteps: Tensor,
                class_labels: Optional[Tensor] = None,
                encoder_hidden_states: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:              (B, C, H, W) noisy input.
            timesteps:      (B,) diffusion timestep.
            class_labels:   (B,) class indices (optional).
            encoder_hidden_states: (B, S, context_dim) text embeddings.

        Returns:
            (B, C_out, H, W) model prediction.
        """
        x._ensure_cpu()
        B, _, H_in, W_in = x._data.shape

        # Patchify
        patches = self._patchify(x._data)
        h = self.patch_proj(
            Tensor._wrap(patches.astype(np.float32), False, None, x._device))

        # Positional embedding (simple learned-free sinusoidal)
        N = patches.shape[1]
        pos = np.arange(N, dtype=np.float32)[None, :, None]
        dim_range = np.arange(self.hidden_size, dtype=np.float32)[None, None, :]
        pe = np.sin(pos / (10000 ** (dim_range / self.hidden_size)))
        h._ensure_cpu()
        h = Tensor._wrap((h._data + pe).astype(np.float32),
                         False, None, x._device)

        # Conditioning
        cond = self.time_embed(timesteps)
        if self.num_classes > 0 and class_labels is not None:
            cls_emb = self.class_embed(class_labels)
            cls_emb._ensure_cpu(); cond._ensure_cpu()
            cond = Tensor._wrap(
                (cond._data + cls_emb._data).astype(np.float32),
                False, None, x._device)

        # Transformer
        for blk in self.blocks:
            h = blk(h, cond, encoder_hidden_states)

        # Final projection
        h = self.final_norm(h)
        h = self.final_proj(h)

        # Unpatchify
        h._ensure_cpu()
        out = self._unpatchify(h._data, H_in, W_in)
        return Tensor._wrap(out.astype(np.float32), False, None, x._device)


# ═════════════════════════════════════════════════════════════════════
#  AutoencoderKL — Variational Autoencoder
# ═════════════════════════════════════════════════════════════════════

class AutoencoderKL(Module):
    """KL-regularised Variational Autoencoder for Latent Diffusion.

    Encodes images (B, C, H, W) to latent representations (B, z, H/f, W/f)
    and decodes back.  Used as the VAE in Stable Diffusion.

    Args:
        in_channels:      Input image channels (3 for RGB).
        latent_channels:  Latent space channels (4 for SD).
        base_channels:    First conv layer width.
        channel_mult:     Channel multipliers for encoder stages.
        scaling_factor:   Multiplied with latent before passing to UNet.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.latent_channels = latent_channels

        # ---- Encoder ----
        enc_layers = [Conv2d(in_channels, base_channels, 3, padding=1), SiLU()]
        ch = base_channels
        for mult in channel_mult:
            out_ch = base_channels * mult
            enc_layers += [
                Conv2d(ch, out_ch, 3, padding=1), SiLU(),
                Conv2d(out_ch, out_ch, 3, stride=2, padding=1), SiLU(),
            ]
            ch = out_ch
        # To mean + logvar
        enc_layers.append(Conv2d(ch, latent_channels * 2, 3, padding=1))
        self.encoder = Sequential(*enc_layers)

        # ---- Decoder ----
        dec_layers = [Conv2d(latent_channels, ch, 3, padding=1), SiLU()]
        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            dec_layers += [
                Conv2d(ch, out_ch, 3, padding=1), SiLU(),
                ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1), SiLU(),
            ]
            ch = out_ch
        dec_layers.append(Conv2d(ch, in_channels, 3, padding=1))
        self.decoder = Sequential(*dec_layers)

        # Quant conv (post-encoder / pre-decoder 1×1)
        self.quant_conv = Conv2d(latent_channels * 2, latent_channels * 2,
                                 kernel_size=1)
        self.post_quant_conv = Conv2d(latent_channels, latent_channels,
                                      kernel_size=1)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode → (mean, logvar)."""
        h = self.encoder(x)
        h = self.quant_conv(h)
        h._ensure_cpu()
        C = h._data.shape[1] // 2
        mean = Tensor._wrap(h._data[:, :C].copy(), False, None, h._device)
        logvar = Tensor._wrap(h._data[:, C:].copy(), False, None, h._device)
        return mean, logvar

    def decode(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def sample(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterisation trick."""
        mean._ensure_cpu(); logvar._ensure_cpu()
        std = np.exp(0.5 * logvar._data)
        eps = np.random.randn(*std.shape).astype(np.float32)
        z = mean._data + eps * std
        return Tensor._wrap(z.astype(np.float32), False, None, mean._device)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Full forward: encode → sample → decode.

        Returns ``(reconstruction, mean, logvar)`` for loss computation.
        """
        mean, logvar = self.encode(x)
        z = self.sample(mean, logvar)
        z_scaled = Tensor._wrap(
            (z._data * self.scaling_factor).astype(np.float32),
            False, None, z._device)
        return self.decode(z_scaled), mean, logvar


# ═════════════════════════════════════════════════════════════════════
#  Exports
# ═════════════════════════════════════════════════════════════════════

__all__ = [
    'UNet2DConditionModel',
    'DiTModel',
    'AutoencoderKL',
    # Building blocks (public but secondary)
    'ResBlock2D',
    'CrossAttentionBlock',
    'Downsample2D',
    'Upsample2D',
    'DiTBlock',
    'AdaLayerNorm',
    'SinusoidalTimestepEmbedding',
    'TimestepMLP',
]
