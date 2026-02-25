# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""Text-to-Video generation pipeline using 3D UNet diffusion.

Architecture:
    Text Encoder  → text embeddings (context vector)
    3D UNet       → denoising network (spatial + temporal convolutions)
    DDPM/DDIM     → noise scheduler for iterative denoising
    Pipeline      → orchestrates encoding → diffusion → decoding

Example usage::

    import scaffolding as torch
    from scaffolding.nn.video import TextToVideoPipeline

    pipeline = TextToVideoPipeline(
        num_frames=16,
        frame_height=64,
        frame_width=64,
        text_embed_dim=256,
        model_channels=64,
    )
    video = pipeline.generate("a cat walking on grass", num_steps=50)
    # video: Tensor of shape (1, 3, 16, 64, 64) — (B, C, T, H, W)
"""
from __future__ import annotations

import math
import numpy as np

from ..tensor import Tensor
from .. import autograd as _ag
from .module import Module, ModuleList, Sequential
from .parameter import Parameter
from .layers import (
    Linear, Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d,
    GroupNorm, SiLU, Dropout, Upsample, LayerNorm,
    _param_to_device, _default_param_device,
)


def _valid_groups(channels: int, max_groups: int = 32) -> int:
    """Return the largest divisor of *channels* that is <= *max_groups*."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ══════════════════════════════════════════════════════════════════════
#  Text Encoder — lightweight transformer-based text encoder
# ══════════════════════════════════════════════════════════════════════

class TextTokenizer:
    """Simple word-level tokenizer with a small vocabulary.

    In production this would be a BPE or SentencePiece tokenizer;
    here we use a hash-based vocabulary for self-contained operation.
    """

    def __init__(self, vocab_size: int = 8192, max_length: int = 77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        # Special tokens
        self.bos_token = 0
        self.eos_token = 1
        self.pad_token = 2

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs via deterministic hashing."""
        words = text.lower().split()
        tokens = [self.bos_token]
        for w in words:
            # Deterministic hash → vocab range [3, vocab_size)
            h = hash(w) % (self.vocab_size - 3) + 3
            tokens.append(h)
        tokens.append(self.eos_token)

        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens += [self.pad_token] * (self.max_length - len(tokens))
        return tokens

    def __call__(self, text: str) -> Tensor:
        tokens = self.encode(text)
        return Tensor(np.array([tokens], dtype=np.int64))


class TextEncoderBlock(Module):
    """Single transformer encoder block for text encoding."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Self-attention projections
        self.q_proj = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim)

        # Feed-forward
        self.ff = Sequential(
            Linear(embed_dim, ff_dim),
            SiLU(),
            Linear(ff_dim, embed_dim),
        )

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        """x: (B, S, D) → (B, S, D)"""
        B, S, D = x.shape
        h = self.num_heads
        hd = self.head_dim

        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).reshape(B, S, h, hd).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, h, hd).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, h, hd).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        from . import functional as F
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, D)

        x = residual + self.dropout(self.out_proj(attn_out))

        # Pre-norm feed-forward
        residual = x
        x = residual + self.dropout(self.ff(self.norm2(x)))

        return x


class TextEncoder(Module):
    """Lightweight transformer text encoder.

    Encodes text tokens into a sequence of embeddings that condition
    the video diffusion model.
    """

    def __init__(self, vocab_size: int = 8192, embed_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 4,
                 max_length: int = 77, ff_mult: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        from .layers import Embedding
        self.token_embed = Embedding(vocab_size, embed_dim)
        self.pos_embed = Parameter(
            Tensor(np.random.randn(1, max_length, embed_dim).astype(np.float32) * 0.02))

        self.blocks = ModuleList([
            TextEncoderBlock(embed_dim, num_heads, embed_dim * ff_mult)
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNorm(embed_dim)
        self.tokenizer = TextTokenizer(vocab_size, max_length)

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        token_ids: (B, S) int64
        Returns:   (B, S, D) float32 — contextualized text embeddings
        """
        x = self.token_embed(token_ids)
        # Add positional embeddings
        self.pos_embed._ensure_cpu()
        x._ensure_cpu()
        pos = self.pos_embed._data[:, :x.shape[1], :]
        x_data = x._data + pos
        x = Tensor._wrap(x_data, x._requires_grad, None, x._device)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return x

    def encode_text(self, text: str) -> Tensor:
        """Convenience: tokenize + encode in one call."""
        tokens = self.tokenizer(text)
        return self.forward(tokens)


# ══════════════════════════════════════════════════════════════════════
#  Timestep embedding — sinusoidal positional encoding for diffusion
# ══════════════════════════════════════════════════════════════════════

class TimestepEmbedding(Module):
    """Sinusoidal timestep embedding (as in DDPM / Transformer)."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = Sequential(
            Linear(dim, dim * 4),
            SiLU(),
            Linear(dim * 4, dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        timesteps: (B,) int or float tensor
        Returns:   (B, dim) float32
        """
        timesteps._ensure_cpu()
        t = timesteps._data.astype(np.float32)
        half = self.dim // 2
        freqs = np.exp(-math.log(self.max_period) *
                       np.arange(half, dtype=np.float32) / half)
        args = t[:, None] * freqs[None, :]
        embedding = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        if self.dim % 2:
            embedding = np.pad(embedding, ((0, 0), (0, 1)))
        emb_tensor = Tensor._wrap(embedding.astype(np.float32), False, None,
                                   timesteps._device)
        return self.mlp(emb_tensor)


# ══════════════════════════════════════════════════════════════════════
#  3D UNet building blocks — ResBlock3D, SpatialTemporalAttention,
#                             Downsample3D, Upsample3D
# ══════════════════════════════════════════════════════════════════════

class ResBlock3D(Module):
    """Residual block with 3D convolutions + timestep conditioning.

    Applies:
        h = norm1(x)
        h = silu(h)
        h = conv1(h)
        h += time_mlp(t_emb)
        h = norm2(h)
        h = silu(h)
        h = conv2(h)
        return h + skip(x)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm(_valid_groups(in_channels), in_channels)
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm(_valid_groups(out_channels), out_channels)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = SiLU()
        self.dropout = Dropout(dropout)

        # Timestep conditioning projection
        self.time_mlp = Sequential(
            SiLU(),
            Linear(time_emb_dim, out_channels),
        )

        # Skip connection (channel-match if needed)
        if in_channels != out_channels:
            self.skip_proj = Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = None

    def forward(self, x, t_emb):
        """
        x:     (B, C, D, H, W)
        t_emb: (B, time_emb_dim)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Add timestep embedding: (B, out_channels) → (B, out_channels, 1, 1, 1)
        t = self.time_mlp(t_emb)
        t._ensure_cpu()
        h._ensure_cpu()
        t_expanded = t._data[:, :, np.newaxis, np.newaxis, np.newaxis]
        h = Tensor._wrap((h._data + t_expanded).astype(np.float32),
                         h._requires_grad, None, h._device)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        if self.skip_proj is not None:
            x = self.skip_proj(x)

        h._ensure_cpu()
        x._ensure_cpu()
        return Tensor._wrap((h._data + x._data).astype(np.float32),
                            h._requires_grad or x._requires_grad,
                            None, h._device)


class SpatialTemporalAttention(Module):
    """Cross-attention block for injecting text conditioning.

    Performs self-attention over the spatial-temporal flattened features,
    then cross-attention with the text embeddings.
    """

    def __init__(self, channels: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = GroupNorm(_valid_groups(channels), channels)

        # Self-attention
        self.to_qkv = Linear(channels, channels * 3, bias=False)
        self.self_attn_out = Linear(channels, channels)

        # Cross-attention with text context
        self.cross_q = Linear(channels, channels, bias=False)
        self.cross_k = Linear(context_dim, channels, bias=False)
        self.cross_v = Linear(context_dim, channels, bias=False)
        self.cross_out = Linear(channels, channels)

        self.ff_norm = LayerNorm(channels)
        self.ff = Sequential(
            Linear(channels, channels * 4),
            SiLU(),
            Linear(channels * 4, channels),
        )

    def forward(self, x, context=None):
        """
        x:       (B, C, D, H, W)
        context: (B, S, context_dim) or None
        """
        x._ensure_cpu()
        B, C, D, H, W = x._data.shape
        N = D * H * W

        # Reshape to sequence: (B, N, C)
        h = self.norm(x)
        h._ensure_cpu()
        h_seq = Tensor._wrap(
            h._data.reshape(B, C, N).transpose(0, 2, 1).copy().astype(np.float32),
            h._requires_grad, None, h._device)

        # Self-attention
        qkv = self.to_qkv(h_seq)
        qkv._ensure_cpu()
        qkv_data = qkv._data.reshape(B, N, 3, self.num_heads, self.head_dim)
        q = Tensor._wrap(qkv_data[:, :, 0].transpose(0, 2, 1, 3).copy().astype(np.float32),
                         False, None, x._device)
        k = Tensor._wrap(qkv_data[:, :, 1].transpose(0, 2, 1, 3).copy().astype(np.float32),
                         False, None, x._device)
        v = Tensor._wrap(qkv_data[:, :, 2].transpose(0, 2, 1, 3).copy().astype(np.float32),
                         False, None, x._device)

        from . import functional as F
        attn = F.scaled_dot_product_attention(q, k, v)
        attn._ensure_cpu()
        attn_out = Tensor._wrap(
            attn._data.transpose(0, 2, 1, 3).reshape(B, N, C).copy().astype(np.float32),
            False, None, x._device)
        h_seq._ensure_cpu()
        h_seq = Tensor._wrap((h_seq._data + self.self_attn_out(attn_out)._ensure_cpu()).astype(np.float32),
                             False, None, x._device)

        # Cross-attention with text
        if context is not None:
            cq = self.cross_q(h_seq)
            ck = self.cross_k(context)
            cv = self.cross_v(context)

            cq._ensure_cpu()
            ck._ensure_cpu()
            cv._ensure_cpu()

            cq_h = Tensor._wrap(
                cq._data.reshape(B, N, self.num_heads, self.head_dim)
                    .transpose(0, 2, 1, 3).copy().astype(np.float32),
                False, None, x._device)
            ck_h = Tensor._wrap(
                ck._data.reshape(B, -1, self.num_heads, self.head_dim)
                    .transpose(0, 2, 1, 3).copy().astype(np.float32),
                False, None, x._device)
            cv_h = Tensor._wrap(
                cv._data.reshape(B, -1, self.num_heads, self.head_dim)
                    .transpose(0, 2, 1, 3).copy().astype(np.float32),
                False, None, x._device)

            cross_attn = F.scaled_dot_product_attention(cq_h, ck_h, cv_h)
            cross_attn._ensure_cpu()
            cross_out = Tensor._wrap(
                cross_attn._data.transpose(0, 2, 1, 3)
                    .reshape(B, N, C).copy().astype(np.float32),
                False, None, x._device)
            h_seq._ensure_cpu()
            h_seq = Tensor._wrap(
                (h_seq._data + self.cross_out(cross_out)._ensure_cpu()).astype(np.float32),
                False, None, x._device)

        # Feed-forward
        ff_in = self.ff_norm(h_seq)
        ff_out = self.ff(ff_in)
        ff_out._ensure_cpu()
        h_seq._ensure_cpu()
        h_seq = Tensor._wrap(
            (h_seq._data + ff_out._data).astype(np.float32),
            False, None, x._device)

        # Reshape back to (B, C, D, H, W)
        h_seq._ensure_cpu()
        out_data = h_seq._data.transpose(0, 2, 1).reshape(B, C, D, H, W).copy()

        x._ensure_cpu()
        return Tensor._wrap(
            (x._data + out_data).astype(np.float32),
            x._requires_grad, None, x._device)


class Downsample3D(Module):
    """Spatial downsampling by 2x using strided 3D convolution.

    Temporal dimension is preserved.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Stride (1,2,2) — downsample spatial, preserve temporal
        self.conv = Conv3d(channels, channels, kernel_size=3,
                           stride=(1, 2, 2), padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(Module):
    """Spatial upsampling by 2x using interpolation + convolution.

    Temporal dimension is preserved.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x._ensure_cpu()
        B, C, D, H, W = x._data.shape
        # Nearest-neighbor upsample spatial dims only
        up = Upsample(scale_factor=(1, 2, 2), mode='nearest')
        x_up = up(x)
        return self.conv(x_up)


# ══════════════════════════════════════════════════════════════════════
#  3D UNet — the denoising backbone
# ══════════════════════════════════════════════════════════════════════

class UNet3D(Module):
    """3D UNet for video diffusion denoising.

    Architecture (for model_channels=64):

        Encoder:
          [64]  → ResBlock3D → Attn → Down → [128]
          [128] → ResBlock3D → Attn → Down → [256]

        Bottleneck:
          [256] → ResBlock3D → Attn → ResBlock3D

        Decoder:
          [256] → Up → cat(skip) → ResBlock3D → Attn → [128]
          [128] → Up → cat(skip) → ResBlock3D → Attn → [64]

        Output:
          [64] → GroupNorm → SiLU → Conv3d → [out_channels]
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 model_channels: int = 64, context_dim: int = 256,
                 channel_mult: tuple = (1, 2, 4),
                 num_res_blocks: int = 2,
                 attention_levels: tuple = (1, 2),
                 num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        time_dim = model_channels * 4

        # Timestep embedding
        self.time_embed = TimestepEmbedding(model_channels)
        self.time_mlp = Sequential(
            Linear(model_channels, time_dim),
            SiLU(),
            Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_conv = Conv3d(in_channels, model_channels, kernel_size=3,
                                  padding=1)

        # Encoder
        self.encoder_blocks = ModuleList()
        self.encoder_downsamples = ModuleList()

        ch = model_channels
        encoder_channels = [ch]

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                block = ResBlock3D(ch, out_ch, time_dim, dropout)
                self.encoder_blocks.append(block)
                if level in attention_levels:
                    attn = SpatialTemporalAttention(out_ch, context_dim, num_heads)
                    self.encoder_blocks.append(attn)
                ch = out_ch
                encoder_channels.append(ch)

            if level < len(channel_mult) - 1:
                self.encoder_downsamples.append(Downsample3D(ch))
                encoder_channels.append(ch)

        # Bottleneck
        self.mid_block1 = ResBlock3D(ch, ch, time_dim, dropout)
        self.mid_attn = SpatialTemporalAttention(ch, context_dim, num_heads)
        self.mid_block2 = ResBlock3D(ch, ch, time_dim, dropout)

        # Decoder
        self.decoder_blocks = ModuleList()
        self.decoder_upsamples = ModuleList()

        for level in reversed(range(len(channel_mult))):
            mult = channel_mult[level]
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                block = ResBlock3D(ch + skip_ch, out_ch, time_dim, dropout)
                self.decoder_blocks.append(block)
                if level in attention_levels:
                    attn = SpatialTemporalAttention(out_ch, context_dim, num_heads)
                    self.decoder_blocks.append(attn)
                ch = out_ch

            if level > 0:
                self.decoder_upsamples.append(Upsample3D(ch))

        # Output
        self.out_norm = GroupNorm(_valid_groups(ch), ch)
        self.out_act = SiLU()
        self.out_conv = Conv3d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, timesteps, context=None):
        """
        x:         (B, C, D, H, W) — noisy video
        timesteps: (B,) — diffusion timestep for each sample
        context:   (B, S, context_dim) — text embeddings

        Returns:   (B, C, D, H, W) — predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        t_emb = self.time_mlp(t_emb)

        # Input projection
        h = self.input_conv(x)

        # Encoder with skip connections
        skips = [h]
        block_idx = 0
        down_idx = 0

        for level in range(len(self._get_channel_mult())):
            for _ in range(self._num_res_blocks):
                block = self.encoder_blocks[block_idx]
                if isinstance(block, ResBlock3D):
                    h = block(h, t_emb)
                    block_idx += 1
                    # Check if next is attention
                    if block_idx < len(self.encoder_blocks) and \
                       isinstance(self.encoder_blocks[block_idx], SpatialTemporalAttention):
                        h = self.encoder_blocks[block_idx](h, context)
                        block_idx += 1
                skips.append(h)

            if down_idx < len(self.encoder_downsamples):
                h = self.encoder_downsamples[down_idx](h)
                skips.append(h)
                down_idx += 1

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)

        # Decoder with skip connections
        block_idx = 0
        up_idx = 0

        for level in reversed(range(len(self._get_channel_mult()))):
            for _ in range(self._num_res_blocks + 1):
                skip = skips.pop()

                # Concatenate skip connection
                h._ensure_cpu()
                skip._ensure_cpu()
                # Handle spatial size mismatch by padding/cropping
                if h._data.shape[2:] != skip._data.shape[2:]:
                    # Crop skip to match h
                    slices = [slice(None), slice(None)]
                    for d1, d2 in zip(h._data.shape[2:], skip._data.shape[2:]):
                        slices.append(slice(0, min(d1, d2)))
                    skip = Tensor._wrap(skip._data[tuple(slices)].copy(),
                                        False, None, h._device)
                    # Pad h if needed
                    if h._data.shape != skip._data.shape:
                        h_data = h._data
                        for dim_i in range(2, h_data.ndim):
                            if h_data.shape[dim_i] > skip._data.shape[dim_i]:
                                sl = [slice(None)] * h_data.ndim
                                sl[dim_i] = slice(0, skip._data.shape[dim_i])
                                h_data = h_data[tuple(sl)]
                        h = Tensor._wrap(h_data.copy(), False, None, h._device)

                h_cat = np.concatenate([h._data, skip._data], axis=1)
                h = Tensor._wrap(h_cat.astype(np.float32), False, None, h._device)

                block = self.decoder_blocks[block_idx]
                if isinstance(block, ResBlock3D):
                    h = block(h, t_emb)
                    block_idx += 1
                    if block_idx < len(self.decoder_blocks) and \
                       isinstance(self.decoder_blocks[block_idx], SpatialTemporalAttention):
                        h = self.decoder_blocks[block_idx](h, context)
                        block_idx += 1

            if up_idx < len(self.decoder_upsamples):
                h = self.decoder_upsamples[up_idx](h)
                up_idx += 1

        # Output
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h

    def _get_channel_mult(self):
        """Recover channel_mult from encoder structure."""
        # Count encoder blocks to infer levels
        # Simple heuristic based on downsamples
        n_levels = len(self.encoder_downsamples) + 1
        return tuple(range(n_levels))

    @property
    def _num_res_blocks(self):
        """Infer num_res_blocks from construction."""
        return 2


# ══════════════════════════════════════════════════════════════════════
#  Simplified 3D UNet — a more practical self-contained version
# ══════════════════════════════════════════════════════════════════════

class SimpleUNet3D(Module):
    """A simplified but fully functional 3D UNet for video diffusion.

    This version uses an explicit encoder-bottleneck-decoder structure
    without the complexity of dynamic block counting.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 64, context_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        ch = base_channels
        time_dim = ch * 4

        # Timestep embedding
        self.time_embed = TimestepEmbedding(ch)
        self.time_proj = Sequential(
            Linear(ch, time_dim),
            SiLU(),
            Linear(time_dim, time_dim),
        )

        # Input
        self.input_conv = Conv3d(in_channels, ch, kernel_size=3, padding=1)

        # Encoder path
        self.enc1 = ResBlock3D(ch, ch, time_dim, dropout)
        self.enc1_attn = SpatialTemporalAttention(ch, context_dim, num_heads)
        self.down1 = Downsample3D(ch)

        self.enc2 = ResBlock3D(ch, ch * 2, time_dim, dropout)
        self.enc2_attn = SpatialTemporalAttention(ch * 2, context_dim, num_heads)
        self.down2 = Downsample3D(ch * 2)

        # Bottleneck
        self.mid1 = ResBlock3D(ch * 2, ch * 4, time_dim, dropout)
        self.mid_attn = SpatialTemporalAttention(ch * 4, context_dim, num_heads)
        self.mid2 = ResBlock3D(ch * 4, ch * 4, time_dim, dropout)

        # Decoder path (with skip concatenation, so double input channels)
        self.up2 = Upsample3D(ch * 4)
        self.dec2 = ResBlock3D(ch * 4 + ch * 2, ch * 2, time_dim, dropout)
        self.dec2_attn = SpatialTemporalAttention(ch * 2, context_dim, num_heads)

        self.up1 = Upsample3D(ch * 2)
        self.dec1 = ResBlock3D(ch * 2 + ch, ch, time_dim, dropout)
        self.dec1_attn = SpatialTemporalAttention(ch, context_dim, num_heads)

        # Output
        self.out_norm = GroupNorm(_valid_groups(ch), ch)
        self.out_act = SiLU()
        self.out_conv = Conv3d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, timesteps, context=None):
        """
        x:         (B, C, D, H, W) — noisy video tensor
        timesteps: (B,) — diffusion timestep
        context:   (B, S, context_dim) — text conditioning

        Returns:   (B, C, D, H, W) — predicted noise epsilon
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        t_emb = self.time_proj(t_emb)

        # Input
        h = self.input_conv(x)

        # Encoder
        h1 = self.enc1(h, t_emb)
        h1 = self.enc1_attn(h1, context)
        h1d = self.down1(h1)

        h2 = self.enc2(h1d, t_emb)
        h2 = self.enc2_attn(h2, context)
        h2d = self.down2(h2)

        # Bottleneck
        mid = self.mid1(h2d, t_emb)
        mid = self.mid_attn(mid, context)
        mid = self.mid2(mid, t_emb)

        # Decoder
        up2 = self.up2(mid)
        # Crop/pad to match skip
        up2._ensure_cpu()
        h2._ensure_cpu()
        up2, h2_matched = _match_spatial(up2, h2)
        cat2_data = np.concatenate([up2._data, h2_matched._data], axis=1)
        cat2 = Tensor._wrap(cat2_data.astype(np.float32), False, None, x._device)
        d2 = self.dec2(cat2, t_emb)
        d2 = self.dec2_attn(d2, context)

        up1 = self.up1(d2)
        up1._ensure_cpu()
        h1._ensure_cpu()
        up1, h1_matched = _match_spatial(up1, h1)
        cat1_data = np.concatenate([up1._data, h1_matched._data], axis=1)
        cat1 = Tensor._wrap(cat1_data.astype(np.float32), False, None, x._device)
        d1 = self.dec1(cat1, t_emb)
        d1 = self.dec1_attn(d1, context)

        # Output
        out = self.out_norm(d1)
        out = self.out_act(out)
        out = self.out_conv(out)
        return out


def _match_spatial(a, b):
    """Crop two tensors to matching spatial dimensions."""
    a._ensure_cpu()
    b._ensure_cpu()
    min_dims = []
    for d1, d2 in zip(a._data.shape[2:], b._data.shape[2:]):
        min_dims.append(min(d1, d2))

    slices = [slice(None), slice(None)]
    for md in min_dims:
        slices.append(slice(0, md))
    slices = tuple(slices)

    a_crop = Tensor._wrap(a._data[slices].copy().astype(np.float32),
                          False, None, a._device)
    b_crop = Tensor._wrap(b._data[slices].copy().astype(np.float32),
                          False, None, b._device)
    return a_crop, b_crop


# ══════════════════════════════════════════════════════════════════════
#  Noise scheduler — DDPM / DDIM
# ══════════════════════════════════════════════════════════════════════

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Model noise scheduler.

    Implements the forward (noise addition) and reverse (denoising) processes
    with a linear beta schedule.
    """

    def __init__(self, num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule: str = 'linear'):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Compute beta schedule
        if schedule == 'linear':
            self.betas = np.linspace(beta_start, beta_end,
                                     num_train_timesteps, dtype=np.float32)
        elif schedule == 'cosine':
            steps = np.arange(num_train_timesteps + 1, dtype=np.float32) / num_train_timesteps
            alpha_bar = np.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = np.clip(betas, 0.0001, 0.9999).astype(np.float32)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]]).astype(np.float32)

        # Pre-compute sqrt values for efficiency
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = 1.0 / np.sqrt(self.alphas)

        # Posterior variance for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_start: Tensor, noise: Tensor,
                  timesteps: Tensor) -> Tensor:
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise"""
        x_start._ensure_cpu()
        noise._ensure_cpu()
        timesteps._ensure_cpu()
        t = timesteps._data.astype(int).flatten()

        sqrt_ab = self.sqrt_alphas_cumprod[t]
        sqrt_1_ab = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting: (B,) → (B, 1, 1, 1, 1)
        ndim = x_start._data.ndim
        shape = [-1] + [1] * (ndim - 1)
        sqrt_ab = sqrt_ab.reshape(shape)
        sqrt_1_ab = sqrt_1_ab.reshape(shape)

        noisy = sqrt_ab * x_start._data + sqrt_1_ab * noise._data
        return Tensor._wrap(noisy.astype(np.float32), False, None, x_start._device)

    def step(self, model_output: Tensor, timestep: int,
             sample: Tensor) -> Tensor:
        """Reverse diffusion step: p(x_{t-1} | x_t).

        Uses the predicted noise to estimate x_{t-1}.
        """
        model_output._ensure_cpu()
        sample._ensure_cpu()
        t = timestep

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]

        # Predict x_{t-1} mean
        pred_mean = sqrt_recip_alpha_t * (
            sample._data - beta_t / self.sqrt_one_minus_alphas_cumprod[t] *
            model_output._data
        )

        if t > 0:
            noise = np.random.randn(*sample._data.shape).astype(np.float32)
            sigma = np.sqrt(self.posterior_variance[t])
            prev_sample = pred_mean + sigma * noise
        else:
            prev_sample = pred_mean

        return Tensor._wrap(prev_sample.astype(np.float32), False, None,
                            sample._device)


class DDIMScheduler:
    """Denoising Diffusion Implicit Models — deterministic sampling.

    Allows fewer sampling steps than DDPM while maintaining quality.
    """

    def __init__(self, num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule: str = 'linear'):
        self.num_train_timesteps = num_train_timesteps

        if schedule == 'linear':
            betas = np.linspace(beta_start, beta_end,
                               num_train_timesteps, dtype=np.float32)
        elif schedule == 'cosine':
            steps = np.arange(num_train_timesteps + 1, dtype=np.float32) / num_train_timesteps
            alpha_bar = np.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = np.clip(betas, 0.0001, 0.9999).astype(np.float32)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)

    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps for inference."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = np.arange(0, num_inference_steps)[::-1] * step_ratio
        self.timesteps = self.timesteps.astype(np.int64)

    def add_noise(self, x_start: Tensor, noise: Tensor,
                  timesteps: Tensor) -> Tensor:
        """Forward process: add noise at timestep t."""
        x_start._ensure_cpu()
        noise._ensure_cpu()
        timesteps._ensure_cpu()
        t = timesteps._data.astype(int).flatten()

        sqrt_ab = np.sqrt(self.alphas_cumprod[t])
        sqrt_1_ab = np.sqrt(1.0 - self.alphas_cumprod[t])

        ndim = x_start._data.ndim
        shape = [-1] + [1] * (ndim - 1)
        sqrt_ab = sqrt_ab.reshape(shape)
        sqrt_1_ab = sqrt_1_ab.reshape(shape)

        noisy = sqrt_ab * x_start._data + sqrt_1_ab * noise._data
        return Tensor._wrap(noisy.astype(np.float32), False, None, x_start._device)

    def step(self, model_output: Tensor, timestep: int,
             sample: Tensor, eta: float = 0.0) -> Tensor:
        """DDIM reverse step (deterministic when eta=0)."""
        model_output._ensure_cpu()
        sample._ensure_cpu()
        t = timestep

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t - 1] if t > 0 else np.float32(1.0)

        # Predict x_0 from x_t and predicted noise
        pred_x0 = (sample._data - np.sqrt(1 - alpha_bar_t) * model_output._data) / \
                  np.sqrt(alpha_bar_t)

        # Clip prediction to [-1, 1]
        pred_x0 = np.clip(pred_x0, -1.0, 1.0)

        # Direction pointing to x_t
        sigma_t = eta * np.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))

        pred_dir = np.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * model_output._data

        # Compute x_{t-1}
        prev_sample = np.sqrt(alpha_bar_prev) * pred_x0 + pred_dir

        if eta > 0 and t > 0:
            noise = np.random.randn(*sample._data.shape).astype(np.float32)
            prev_sample += sigma_t * noise

        return Tensor._wrap(prev_sample.astype(np.float32), False, None,
                            sample._device)


# ══════════════════════════════════════════════════════════════════════
#  Video VAE — simple variational autoencoder for encoding/decoding
# ══════════════════════════════════════════════════════════════════════

class VideoEncoder(Module):
    """Simple video encoder: (B, C, T, H, W) → (B, latent_dim, T', H', W')."""

    def __init__(self, in_channels: int = 3, latent_channels: int = 4,
                 base_channels: int = 64):
        super().__init__()
        ch = base_channels
        self.encoder = Sequential(
            Conv3d(in_channels, ch, kernel_size=3, padding=1),
            SiLU(),
            Conv3d(ch, ch, kernel_size=3, stride=(1, 2, 2), padding=1),
            SiLU(),
            Conv3d(ch, ch * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
            SiLU(),
            Conv3d(ch * 2, latent_channels * 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        h = self.encoder(x)
        h._ensure_cpu()
        # Split into mean and log_var
        C = h._data.shape[1]
        mean = Tensor._wrap(h._data[:, :C // 2].copy(), False, None, h._device)
        log_var = Tensor._wrap(h._data[:, C // 2:].copy(), False, None, h._device)
        return mean, log_var

    def sample(self, mean, log_var):
        """Reparameterization trick."""
        mean._ensure_cpu()
        log_var._ensure_cpu()
        std = np.exp(0.5 * log_var._data)
        eps = np.random.randn(*std.shape).astype(np.float32)
        z = mean._data + eps * std
        return Tensor._wrap(z.astype(np.float32), False, None, mean._device)


class VideoDecoder(Module):
    """Simple video decoder: (B, latent_dim, T', H', W') → (B, C, T, H, W)."""

    def __init__(self, latent_channels: int = 4, out_channels: int = 3,
                 base_channels: int = 64):
        super().__init__()
        ch = base_channels
        self.decoder = Sequential(
            Conv3d(latent_channels, ch * 2, kernel_size=3, padding=1),
            SiLU(),
            Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            Conv3d(ch * 2, ch, kernel_size=3, padding=1),
            SiLU(),
            Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            Conv3d(ch, ch, kernel_size=3, padding=1),
            SiLU(),
            Conv3d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        return self.decoder(z)


# ══════════════════════════════════════════════════════════════════════
#  Text-to-Video Pipeline — orchestrates the full generation flow
# ══════════════════════════════════════════════════════════════════════

class TextToVideoPipeline(Module):
    """Complete text-to-video generation pipeline.

    Combines:
    - TextEncoder for prompt conditioning
    - SimpleUNet3D denoising model
    - DDIMScheduler for efficient sampling
    - Optional VideoEncoder/Decoder for latent diffusion

    Example::

        pipe = TextToVideoPipeline(
            num_frames=16,
            frame_height=64,
            frame_width=64,
        )
        video = pipe.generate("a dog running", num_steps=50)
        print(video.shape)  # (1, 3, 16, 64, 64)
    """

    def __init__(self, num_frames: int = 16,
                 frame_height: int = 64,
                 frame_width: int = 64,
                 in_channels: int = 3,
                 text_embed_dim: int = 256,
                 model_channels: int = 64,
                 num_heads: int = 4,
                 vocab_size: int = 8192,
                 num_train_timesteps: int = 1000,
                 use_latent: bool = False,
                 latent_channels: int = 4):
        super().__init__()
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.in_channels = in_channels
        self.use_latent = use_latent
        self.latent_channels = latent_channels

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            num_heads=num_heads,
            num_layers=4,
            max_length=77,
        )

        # UNet denoising model
        unet_in = latent_channels if use_latent else in_channels
        unet_out = latent_channels if use_latent else in_channels
        self.unet = SimpleUNet3D(
            in_channels=unet_in,
            out_channels=unet_out,
            base_channels=model_channels,
            context_dim=text_embed_dim,
            num_heads=num_heads,
        )

        # Optional VAE for latent diffusion
        if use_latent:
            self.encoder = VideoEncoder(in_channels, latent_channels,
                                        model_channels)
            self.decoder = VideoDecoder(latent_channels, in_channels,
                                        model_channels)
        else:
            self.encoder = None
            self.decoder = None

        # Noise scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.00085,
            beta_end=0.012,
            schedule='linear',
        )

    @_ag.no_grad()
    def generate(self, prompt: str, num_steps: int = 50,
                 guidance_scale: float = 7.5,
                 eta: float = 0.0,
                 seed: int | None = None) -> Tensor:
        """Generate a video from a text prompt.

        Args:
            prompt:         Text description of the video to generate.
            num_steps:      Number of denoising steps (more = higher quality).
            guidance_scale: Classifier-free guidance strength.
            eta:            DDIM randomness (0 = deterministic, 1 = DDPM-like).
            seed:           Optional random seed for reproducibility.

        Returns:
            Tensor of shape (1, C, T, H, W) with pixel values in [-1, 1].
        """
        if seed is not None:
            np.random.seed(seed)

        self.eval()

        # Encode text
        text_emb = self.text_encoder.encode_text(prompt)

        # Prepare unconditional embedding for classifier-free guidance
        null_emb = self.text_encoder.encode_text("")

        # Set up scheduler
        self.scheduler.set_timesteps(num_steps)

        # Initialize with pure noise
        if self.use_latent:
            T = self.num_frames
            H = self.frame_height // 4   # 2x downsampling twice
            W = self.frame_width // 4
            C = self.latent_channels
        else:
            T = self.num_frames
            H = self.frame_height
            W = self.frame_width
            C = self.in_channels

        latent = Tensor._wrap(
            np.random.randn(1, C, T, H, W).astype(np.float32),
            False, None, None)

        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            t_tensor = Tensor._wrap(
                np.array([t], dtype=np.float32), False, None, None)

            if guidance_scale > 1.0:
                # Classifier-free guidance: run UNet twice
                # Conditional prediction
                noise_pred_cond = self.unet(latent, t_tensor, text_emb)

                # Unconditional prediction
                noise_pred_uncond = self.unet(latent, t_tensor, null_emb)

                # Guided noise prediction
                noise_pred_cond._ensure_cpu()
                noise_pred_uncond._ensure_cpu()
                guided = noise_pred_uncond._data + guidance_scale * (
                    noise_pred_cond._data - noise_pred_uncond._data)
                noise_pred = Tensor._wrap(guided.astype(np.float32),
                                          False, None, None)
            else:
                noise_pred = self.unet(latent, t_tensor, text_emb)

            latent = self.scheduler.step(noise_pred, int(t), latent, eta=eta)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Step {i + 1}/{num_steps}, t={t}")

        # Decode from latent space if using VAE
        if self.use_latent and self.decoder is not None:
            video = self.decoder(latent)
        else:
            video = latent

        # Clamp output to [-1, 1]
        video._ensure_cpu()
        video = Tensor._wrap(
            np.clip(video._data, -1.0, 1.0).astype(np.float32),
            False, None, None)

        return video

    def training_step(self, video: Tensor, prompt_tokens: Tensor) -> Tensor:
        """Single training step: compute diffusion loss on a video batch.

        Args:
            video:         (B, C, T, H, W) real video tensor in [-1, 1]
            prompt_tokens: (B, S) text token IDs

        Returns:
            Scalar loss tensor (MSE between predicted and true noise).
        """
        B = video.shape[0]

        # Encode text
        text_emb = self.text_encoder(prompt_tokens)

        # Get latent representation if using VAE
        if self.use_latent and self.encoder is not None:
            mean, log_var = self.encoder(video)
            x_start = self.encoder.sample(mean, log_var)
        else:
            x_start = video

        # Sample random timesteps
        t = np.random.randint(0, self.scheduler.num_train_timesteps,
                              size=(B,)).astype(np.int64)
        t_tensor = Tensor._wrap(t.astype(np.float32), False, None, video._device)

        # Add noise
        x_start._ensure_cpu()
        noise = Tensor._wrap(
            np.random.randn(*x_start._data.shape).astype(np.float32),
            False, None, video._device)

        timestep_tensor = Tensor._wrap(t.astype(np.int64), False, None,
                                        video._device)
        noisy = self.scheduler.add_noise(x_start, noise, timestep_tensor)

        # Predict noise
        noise_pred = self.unet(noisy, t_tensor, text_emb)

        # MSE loss between predicted and actual noise
        noise_pred._ensure_cpu()
        noise._ensure_cpu()
        diff = noise_pred._data - noise._data
        loss_val = np.mean(diff ** 2)

        return Tensor._wrap(np.float32(loss_val),
                            requires_grad=True, device=video._device)

    def save_video_frames(self, video: Tensor, output_dir: str,
                          format: str = 'png') -> list[str]:
        """Save generated video frames to disk.

        Args:
            video:      (1, C, T, H, W) tensor in [-1, 1]
            output_dir: Directory to save frames
            format:     Image format ('png' or 'jpg')

        Returns:
            List of saved file paths.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        video._ensure_cpu()
        frames = video._data[0]  # (C, T, H, W)
        T = frames.shape[1]

        paths = []
        for t in range(T):
            frame = frames[:, t]  # (C, H, W)
            # Convert from [-1, 1] to [0, 255]
            frame = ((frame + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            frame = frame.transpose(1, 2, 0)  # (H, W, C)

            path = os.path.join(output_dir, f"frame_{t:04d}.{format}")

            # Save as raw PPM if PIL not available
            try:
                from PIL import Image
                img = Image.fromarray(frame)
                img.save(path)
            except ImportError:
                # Fallback: save as PPM (no dependency needed)
                path = os.path.join(output_dir, f"frame_{t:04d}.ppm")
                H, W, C = frame.shape
                with open(path, 'wb') as f:
                    f.write(f"P6\n{W} {H}\n255\n".encode())
                    f.write(frame.tobytes())

            paths.append(path)

        return paths


# ══════════════════════════════════════════════════════════════════════
#  Convenience exports
# ══════════════════════════════════════════════════════════════════════

__all__ = [
    # Text encoding
    'TextTokenizer',
    'TextEncoder',
    'TextEncoderBlock',
    # UNet
    'SimpleUNet3D',
    'UNet3D',
    'ResBlock3D',
    'SpatialTemporalAttention',
    'Downsample3D',
    'Upsample3D',
    'TimestepEmbedding',
    # Scheduling
    'DDPMScheduler',
    'DDIMScheduler',
    # VAE
    'VideoEncoder',
    'VideoDecoder',
    # Pipeline
    'TextToVideoPipeline',
]
