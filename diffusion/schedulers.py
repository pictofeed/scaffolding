# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Scaffolding — Deep Learning Framework                               ║
# ║  Copyright © 2026 Pictofeed, LLC. All rights reserved.               ║
# ╚══════════════════════════════════════════════════════════════════════╝

"""Noise schedulers for diffusion models.

Implements the full family of noise schedules and sampling strategies
used in modern diffusion / flow-matching generative models:

- **DDPMScheduler** — Denoising Diffusion Probabilistic Models (Ho et al. 2020)
- **DDIMScheduler** — Denoising Diffusion Implicit Models (Song et al. 2020)
- **DPMSolverMultistepScheduler** — DPM-Solver++ (Lu et al. 2022)
- **CogVideoXDPMScheduler** — CogVideoX-adapted DPM noise schedule
- **EulerDiscreteScheduler** — Euler method on the ODE probability flow
- **PNDMScheduler** — Pseudo Numerical Diffusion Models (Liu et al. 2022)
- **FlowMatchEulerDiscreteScheduler** — Flow Matching (Lipman et al. 2023)
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional, Union, List

from scaffolding.tensor import Tensor


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def _linear_beta_schedule(num_timesteps: int, beta_start: float,
                          beta_end: float) -> np.ndarray:
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)


def _cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> np.ndarray:
    steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
    alpha_bar = np.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 0.0001, 0.9999).astype(np.float32)


def _scaled_linear_beta_schedule(num_timesteps: int, beta_start: float,
                                 beta_end: float) -> np.ndarray:
    """Scaled-linear schedule (square-root spacing, as in Stable Diffusion)."""
    return np.linspace(beta_start ** 0.5, beta_end ** 0.5,
                       num_timesteps, dtype=np.float32) ** 2


def _squaredcos_cap_v2_schedule(num_timesteps: int) -> np.ndarray:
    """Squared cosine schedule capped to avoid singularities (GLIDE variant)."""
    steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
    alpha_bar = np.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 0.0, 0.999).astype(np.float32)


def _get_betas(schedule: str, num_timesteps: int,
               beta_start: float, beta_end: float) -> np.ndarray:
    if schedule == 'linear':
        return _linear_beta_schedule(num_timesteps, beta_start, beta_end)
    elif schedule == 'cosine':
        return _cosine_beta_schedule(num_timesteps)
    elif schedule == 'scaled_linear':
        return _scaled_linear_beta_schedule(num_timesteps, beta_start, beta_end)
    elif schedule == 'squaredcos_cap_v2':
        return _squaredcos_cap_v2_schedule(num_timesteps)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule!r}")


def _broadcast_to_ndim(arr: np.ndarray, ndim: int) -> np.ndarray:
    """Reshape a 1-D array to broadcast against an ndim tensor: (B,) → (B,1,…,1)."""
    shape = [-1] + [1] * (ndim - 1)
    return arr.reshape(shape)


# ═════════════════════════════════════════════════════════════════════
#  DDPMScheduler
# ═════════════════════════════════════════════════════════════════════

class DDPMScheduler:
    """Denoising Diffusion Probabilistic Models (Ho et al. 2020).

    Supports linear, cosine, scaled-linear, and squaredcos_cap_v2 schedules.
    Implements both the training forward process ``add_noise`` and the
    stochastic reverse ``step``.

    Args:
        num_train_timesteps: Total number of diffusion timesteps T.
        beta_start:          Starting β value.
        beta_end:            Ending β value.
        beta_schedule:       One of ``'linear'``, ``'cosine'``,
                             ``'scaled_linear'``, ``'squaredcos_cap_v2'``.
        clip_sample:         Whether to clip predicted x₀ to [-1, 1].
        prediction_type:     ``'epsilon'`` (predict noise) or ``'v_prediction'``.
        variance_type:       ``'fixed_small'`` or ``'fixed_large'``.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        clip_sample: bool = True,
        prediction_type: str = 'epsilon',
        variance_type: str = 'fixed_small',
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.variance_type = variance_type

        self.betas = _get_betas(beta_schedule, num_train_timesteps,
                                beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)
        self.alphas_cumprod_prev = np.concatenate(
            [[1.0], self.alphas_cumprod[:-1]]
        ).astype(np.float32)

        # Pre-computed constants
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = 1.0 / np.sqrt(self.alphas)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)

        # Posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.maximum(self.posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # Default inference timesteps (all)
        self.timesteps = np.arange(num_train_timesteps - 1, -1, -1).astype(np.int64)
        self.num_inference_steps: int | None = None

    # ---- public API ----

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (
            np.arange(0, num_inference_steps)[::-1] * step_ratio
        ).astype(np.int64)

    def add_noise(self, original: Tensor, noise: Tensor,
                  timesteps: Tensor) -> Tensor:
        """Forward diffusion q(x_t | x_0)."""
        original._ensure_cpu(); noise._ensure_cpu(); timesteps._ensure_cpu()
        t = timesteps._data.astype(int).ravel()
        ndim = original._data.ndim
        s_a = _broadcast_to_ndim(self.sqrt_alphas_cumprod[t], ndim)
        s_1a = _broadcast_to_ndim(self.sqrt_one_minus_alphas_cumprod[t], ndim)
        noisy = s_a * original._data + s_1a * noise._data
        return Tensor._wrap(noisy.astype(np.float32), False, None,
                            original._device)

    def _get_variance(self, t: int) -> float:
        if self.variance_type == 'fixed_large':
            return float(self.betas[t])
        return float(self.posterior_variance[t])

    def _predict_x0(self, model_output: np.ndarray, sample: np.ndarray,
                    t: int) -> np.ndarray:
        alpha_bar_t = self.alphas_cumprod[t]
        if self.prediction_type == 'epsilon':
            x0 = (sample - np.sqrt(1 - alpha_bar_t) * model_output) / np.sqrt(alpha_bar_t)
        elif self.prediction_type == 'v_prediction':
            x0 = np.sqrt(alpha_bar_t) * sample - np.sqrt(1 - alpha_bar_t) * model_output
        else:
            raise ValueError(self.prediction_type)
        if self.clip_sample:
            x0 = np.clip(x0, -1.0, 1.0)
        return x0

    def step(self, model_output: Tensor, timestep: int,
             sample: Tensor, generator: Optional[np.random.Generator] = None
             ) -> Tensor:
        """Reverse diffusion step p(x_{t-1}|x_t)."""
        model_output._ensure_cpu(); sample._ensure_cpu()
        t = timestep
        pred_x0 = self._predict_x0(model_output._data, sample._data, t)
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        pred_mean = coef1 * pred_x0 + coef2 * sample._data
        if t > 0:
            var = self._get_variance(t)
            rng = generator or np.random.default_rng()
            noise = rng.standard_normal(sample._data.shape).astype(np.float32)
            prev = pred_mean + np.sqrt(var) * noise
        else:
            prev = pred_mean
        return Tensor._wrap(prev.astype(np.float32), False, None, sample._device)


# ═════════════════════════════════════════════════════════════════════
#  DDIMScheduler
# ═════════════════════════════════════════════════════════════════════

class DDIMScheduler:
    """Denoising Diffusion Implicit Models (Song et al. 2020).

    Allows deterministic (eta=0) or stochastic reverse sampling with
    far fewer inference steps than DDPM.

    Args:
        num_train_timesteps: Training diffusion steps T.
        beta_start / beta_end: Beta range.
        beta_schedule:       Schedule type.
        clip_sample:         Clip predicted x₀.
        prediction_type:     ``'epsilon'`` or ``'v_prediction'``.
        set_alpha_to_one:    Forces ᾱ₀ = 1 for the boundary condition.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        clip_sample: bool = True,
        prediction_type: str = 'epsilon',
        set_alpha_to_one: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        betas = _get_betas(beta_schedule, num_train_timesteps,
                           beta_start, beta_end)
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)
        self.final_alpha_cumprod = np.float32(1.0 if set_alpha_to_one
                                              else self.alphas_cumprod[0])

        self.timesteps = np.arange(num_train_timesteps - 1, -1, -1).astype(np.int64)
        self.num_inference_steps: int | None = None

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (
            np.arange(0, num_inference_steps)[::-1] * step_ratio
        ).astype(np.int64)

    def add_noise(self, original: Tensor, noise: Tensor,
                  timesteps: Tensor) -> Tensor:
        original._ensure_cpu(); noise._ensure_cpu(); timesteps._ensure_cpu()
        t = timesteps._data.astype(int).ravel()
        ndim = original._data.ndim
        s_a = _broadcast_to_ndim(np.sqrt(self.alphas_cumprod[t]), ndim)
        s_1a = _broadcast_to_ndim(np.sqrt(1.0 - self.alphas_cumprod[t]), ndim)
        noisy = s_a * original._data + s_1a * noise._data
        return Tensor._wrap(noisy.astype(np.float32), False, None,
                            original._device)

    def step(self, model_output: Tensor, timestep: int,
             sample: Tensor, eta: float = 0.0) -> Tensor:
        """DDIM reverse step (deterministic when eta=0)."""
        model_output._ensure_cpu(); sample._ensure_cpu()
        t = timestep
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = (self.alphas_cumprod[t - 1]
                          if t > 0 else self.final_alpha_cumprod)

        if self.prediction_type == 'epsilon':
            pred_x0 = (sample._data - np.sqrt(1 - alpha_bar_t) *
                       model_output._data) / np.sqrt(alpha_bar_t)
        elif self.prediction_type == 'v_prediction':
            pred_x0 = (np.sqrt(alpha_bar_t) * sample._data -
                       np.sqrt(1 - alpha_bar_t) * model_output._data)
        else:
            raise ValueError(self.prediction_type)

        if self.clip_sample:
            pred_x0 = np.clip(pred_x0, -1.0, 1.0)

        sigma = eta * np.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            * (1 - alpha_bar_t / alpha_bar_prev)
        )
        pred_dir = np.sqrt(np.maximum(1 - alpha_bar_prev - sigma ** 2, 0)) * model_output._data
        prev = np.sqrt(alpha_bar_prev) * pred_x0 + pred_dir

        if eta > 0 and t > 0:
            noise = np.random.randn(*sample._data.shape).astype(np.float32)
            prev += sigma * noise

        return Tensor._wrap(prev.astype(np.float32), False, None,
                            sample._device)


# ═════════════════════════════════════════════════════════════════════
#  DPMSolverMultistepScheduler (DPM-Solver++)
# ═════════════════════════════════════════════════════════════════════

class DPMSolverMultistepScheduler:
    """DPM-Solver++ multistep scheduler (Lu et al. 2022).

    A fast ODE solver for diffusion models that achieves high-quality
    samples in 15–25 steps.  Supports orders 1 (Euler), 2 (midpoint),
    and 3 (Adams–Bashforth-like multistep).

    Args:
        num_train_timesteps: Training diffusion steps.
        beta_start / beta_end: Beta range.
        beta_schedule:       Schedule type.
        solver_order:        ODE solver order (1, 2, or 3).
        prediction_type:     ``'epsilon'`` or ``'v_prediction'``.
        algorithm_type:      ``'dpmsolver++'`` (data-prediction) or
                             ``'dpmsolver'`` (noise-prediction).
        solver_type:         ``'midpoint'`` or ``'heun'`` for order-2.
        lower_order_final:   Use first-order step for the last step.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        solver_order: int = 2,
        prediction_type: str = 'epsilon',
        algorithm_type: str = 'dpmsolver++',
        solver_type: str = 'midpoint',
        lower_order_final: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final

        betas = _get_betas(beta_schedule, num_train_timesteps,
                           beta_start, beta_end)
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)

        # Continuous-time log-SNR
        self.lambda_t = np.log(np.sqrt(self.alphas_cumprod) /
                               np.sqrt(1.0 - self.alphas_cumprod))
        self.sigma_t = np.sqrt(1.0 - self.alphas_cumprod)
        self.alpha_t = np.sqrt(self.alphas_cumprod)

        self.timesteps = np.arange(num_train_timesteps - 1, -1, -1).astype(np.int64)
        self.num_inference_steps: int | None = None
        self._model_outputs: list[np.ndarray] = []
        self._lower_order_nums = 0

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        # Uniform timestep spacing in [0, T-1]
        step_ratio = self.num_train_timesteps / num_inference_steps
        self.timesteps = np.round(
            np.arange(num_inference_steps) * step_ratio
        ).astype(np.int64)[::-1].copy()
        self._model_outputs = []
        self._lower_order_nums = 0

    def add_noise(self, original: Tensor, noise: Tensor,
                  timesteps: Tensor) -> Tensor:
        original._ensure_cpu(); noise._ensure_cpu(); timesteps._ensure_cpu()
        t = timesteps._data.astype(int).ravel()
        ndim = original._data.ndim
        a = _broadcast_to_ndim(self.alpha_t[t], ndim)
        s = _broadcast_to_ndim(self.sigma_t[t], ndim)
        return Tensor._wrap(
            (a * original._data + s * noise._data).astype(np.float32),
            False, None, original._device)

    def _convert_model_output(self, model_output: np.ndarray,
                              sample: np.ndarray, t: int) -> np.ndarray:
        """Convert model output to data prediction x₀."""
        alpha_t = self.alpha_t[t]
        sigma_t = self.sigma_t[t]
        if self.algorithm_type == 'dpmsolver++':
            if self.prediction_type == 'epsilon':
                return (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == 'v_prediction':
                return alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(self.prediction_type)
        else:
            # dpmsolver — noise prediction
            if self.prediction_type == 'epsilon':
                return model_output
            elif self.prediction_type == 'v_prediction':
                return alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(self.prediction_type)

    def _dpm_solver_first_order(self, model_output: np.ndarray,
                                sample: np.ndarray,
                                t: int, t_prev: int) -> np.ndarray:
        """First-order DPM-Solver (equivalent to DDIM)."""
        lam_t = self.lambda_t[t]
        lam_prev = self.lambda_t[t_prev] if t_prev >= 0 else self.lambda_t[0]
        h = lam_prev - lam_t
        alpha_prev = self.alpha_t[t_prev] if t_prev >= 0 else np.float32(1.0)
        sigma_prev = self.sigma_t[t_prev] if t_prev >= 0 else np.float32(0.0)

        if self.algorithm_type == 'dpmsolver++':
            return (alpha_prev / self.alpha_t[t]) * sample - \
                   sigma_prev * np.expm1(h) * model_output
        else:
            return (self.alpha_t[t_prev] / self.alpha_t[t]) * sample - \
                   self.sigma_t[t_prev] * np.expm1(h) * model_output

    def _dpm_solver_second_order(self, model_outputs: list,
                                 sample: np.ndarray,
                                 t: int, t_prev: int,
                                 t_list: list) -> np.ndarray:
        """Second-order DPM-Solver (midpoint or Heun)."""
        lam_t = self.lambda_t[t]
        lam_prev = self.lambda_t[t_prev] if t_prev >= 0 else self.lambda_t[0]
        h = lam_prev - lam_t
        alpha_prev = self.alpha_t[t_prev] if t_prev >= 0 else np.float32(1.0)
        sigma_prev = self.sigma_t[t_prev] if t_prev >= 0 else np.float32(0.0)

        m0, m1 = model_outputs[-1], model_outputs[-2]
        lam_s1 = self.lambda_t[t_list[-2]] if len(t_list) >= 2 else lam_t
        h_0 = lam_t - lam_s1 if lam_t != lam_s1 else h

        D0 = m0
        D1 = (m0 - m1) / (h_0 + 1e-8)  # avoid division by zero

        if self.algorithm_type == 'dpmsolver++':
            if self.solver_type == 'midpoint':
                return (alpha_prev / self.alpha_t[t]) * sample - \
                       sigma_prev * np.expm1(h) * D0 - \
                       0.5 * sigma_prev * np.expm1(h) * D1
            else:  # heun
                return (alpha_prev / self.alpha_t[t]) * sample - \
                       sigma_prev * np.expm1(h) * D0 - \
                       sigma_prev * (np.expm1(h) / h - 1.0) * D1
        else:
            return (alpha_prev / self.alpha_t[t]) * sample - \
                   sigma_prev * np.expm1(h) * D0 - \
                   0.5 * sigma_prev * np.expm1(h) * D1

    def step(self, model_output: Tensor, timestep: int,
             sample: Tensor) -> Tensor:
        """One step of DPM-Solver++."""
        model_output._ensure_cpu(); sample._ensure_cpu()

        # Find current position in schedule
        step_idx = int(np.argmin(np.abs(self.timesteps - timestep)))
        t = int(self.timesteps[step_idx])
        t_prev = int(self.timesteps[step_idx + 1]) if step_idx + 1 < len(self.timesteps) else 0

        # Convert to data prediction
        data_pred = self._convert_model_output(model_output._data,
                                               sample._data, t)
        self._model_outputs.append(data_pred)
        if len(self._model_outputs) > self.solver_order:
            self._model_outputs.pop(0)

        # Determine effective order
        order = min(self.solver_order, len(self._model_outputs))
        if self.lower_order_final and step_idx == len(self.timesteps) - 2:
            order = 1

        if order == 1:
            prev = self._dpm_solver_first_order(data_pred, sample._data,
                                                 t, t_prev)
        elif order == 2:
            t_list = [int(self.timesteps[max(step_idx - 1, 0)]),
                      int(self.timesteps[step_idx])]
            prev = self._dpm_solver_second_order(
                self._model_outputs, sample._data, t, t_prev, t_list)
        else:
            # Fall back to second-order for order >= 3
            t_list = [int(self.timesteps[max(step_idx - 1, 0)]),
                      int(self.timesteps[step_idx])]
            prev = self._dpm_solver_second_order(
                self._model_outputs, sample._data, t, t_prev, t_list)

        self._lower_order_nums += 1
        return Tensor._wrap(prev.astype(np.float32), False, None,
                            sample._device)


# ═════════════════════════════════════════════════════════════════════
#  CogVideoXDPMScheduler
# ═════════════════════════════════════════════════════════════════════

class CogVideoXDPMScheduler:
    """CogVideoX-adapted DPM noise scheduler.

    A specialised DPM-Solver++ scheduler tuned for CogVideoX video
    diffusion models. It uses a *shifted* noise schedule that allocates
    more budget to the temporal dimension for temporally-coherent video
    generation.

    Key differences from vanilla DPM-Solver++:

    1. **SNR-shifted schedule** — ``snr_shift_scale`` re-maps the
       signal-to-noise ratio so that the model spends more denoising
       capacity on temporal coherence.
    2. **Configurable timestep spacing** — ``'linspace'``, ``'leading'``,
       or ``'trailing'``.
    3. **x₀ clipping / thresholding** — dynamic thresholding for
       high-guidance-scale sampling.
    4. **Re-scaled β schedule** — the β schedule is rescaled after SNR
       shifting so that the forward process still starts from pure noise.

    Args:
        num_train_timesteps: Total training diffusion timesteps.
        beta_start:          Starting β.
        beta_end:            Ending β.
        beta_schedule:       ``'linear'``, ``'scaled_linear'``, or ``'cosine'``.
        snr_shift_scale:     Factor to shift the SNR schedule (≥ 1.0).
                             Higher values push more noise budget toward
                             temporal dimensions.  CogVideoX default = 3.0.
        prediction_type:     ``'epsilon'``, ``'v_prediction'``, or ``'sample'``.
        solver_order:        DPM-Solver order (1 or 2).
        timestep_spacing:    ``'linspace'``, ``'leading'``, or ``'trailing'``.
        clip_sample:         Clip predicted x₀ to ``[-clip_sample_range, clip_sample_range]``.
        clip_sample_range:   Range for clipping.
        use_dynamic_thresholding: If True, apply dynamic thresholding
                             (Imagen-style) on predicted x₀.
        dynamic_thresholding_ratio: Percentile for dynamic thresholding.
        rescale_betas_zero_snr: Zero-terminal SNR re-scaling.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = 'scaled_linear',
        snr_shift_scale: float = 3.0,
        prediction_type: str = 'v_prediction',
        solver_order: int = 2,
        timestep_spacing: str = 'linspace',
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        use_dynamic_thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        rescale_betas_zero_snr: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.snr_shift_scale = snr_shift_scale
        self.prediction_type = prediction_type
        self.solver_order = solver_order
        self.timestep_spacing = timestep_spacing
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.use_dynamic_thresholding = use_dynamic_thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.rescale_betas_zero_snr = rescale_betas_zero_snr

        # ---- compute base schedule ----
        betas = _get_betas(beta_schedule, num_train_timesteps,
                           beta_start, beta_end)
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float64)

        # ---- SNR shift (core CogVideoX modification) ----
        # Shift the alpha_cumprod so that the SNR curve is rescaled.
        # ᾱ'_t = ᾱ_t / (snr_shift_scale + (1 − snr_shift_scale) · ᾱ_t)
        if snr_shift_scale != 1.0:
            self.alphas_cumprod = (
                self.alphas_cumprod / (
                    snr_shift_scale
                    + (1.0 - snr_shift_scale) * self.alphas_cumprod
                )
            )

        self.alphas_cumprod = self.alphas_cumprod.astype(np.float32)

        # ---- zero-terminal-SNR rescaling ----
        if rescale_betas_zero_snr:
            # Enforce ᾱ_T = 0 exactly
            self.alphas_cumprod[-1] = 0.0

        # Derived quantities
        self.alpha_t = np.sqrt(self.alphas_cumprod)
        self.sigma_t = np.sqrt(1.0 - self.alphas_cumprod)
        self.lambda_t = np.log(
            self.alpha_t / np.maximum(self.sigma_t, 1e-20)
        )

        # Default timesteps
        self.timesteps = np.arange(
            num_train_timesteps - 1, -1, -1
        ).astype(np.int64)
        self.num_inference_steps: int | None = None

        # Multistep buffer
        self._model_outputs: list[np.ndarray] = []
        self._lower_order_nums = 0

    # ----------------------------------------------------------------
    #  Timestep configuration
    # ----------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int):
        """Set inference timesteps using the configured spacing strategy."""
        self.num_inference_steps = num_inference_steps
        T = self.num_train_timesteps

        if self.timestep_spacing == 'linspace':
            self.timesteps = np.round(
                np.linspace(T - 1, 0, num_inference_steps)
            ).astype(np.int64)
        elif self.timestep_spacing == 'leading':
            step_ratio = T // num_inference_steps
            self.timesteps = (
                np.arange(0, num_inference_steps)[::-1] * step_ratio
            ).astype(np.int64)
        elif self.timestep_spacing == 'trailing':
            step_ratio = T / num_inference_steps
            self.timesteps = np.round(
                np.arange(T, 0, -step_ratio)
            ).astype(np.int64) - 1
            self.timesteps = np.clip(self.timesteps, 0, T - 1)
        else:
            raise ValueError(f"Unknown spacing: {self.timestep_spacing!r}")

        self._model_outputs = []
        self._lower_order_nums = 0

    # ----------------------------------------------------------------
    #  Forward process
    # ----------------------------------------------------------------

    def add_noise(self, original: Tensor, noise: Tensor,
                  timesteps: Tensor) -> Tensor:
        """Forward diffusion q(x_t | x_0) with SNR-shifted schedule."""
        original._ensure_cpu(); noise._ensure_cpu(); timesteps._ensure_cpu()
        t = timesteps._data.astype(int).ravel()
        ndim = original._data.ndim
        a = _broadcast_to_ndim(self.alpha_t[t], ndim)
        s = _broadcast_to_ndim(self.sigma_t[t], ndim)
        noisy = a * original._data + s * noise._data
        return Tensor._wrap(noisy.astype(np.float32), False, None,
                            original._device)

    # ----------------------------------------------------------------
    #  x₀ prediction from model output
    # ----------------------------------------------------------------

    def _predict_x0(self, model_output: np.ndarray,
                    sample: np.ndarray, t: int) -> np.ndarray:
        alpha_t = float(self.alpha_t[t])
        sigma_t = float(self.sigma_t[t])

        if self.prediction_type == 'epsilon':
            x0 = (sample - sigma_t * model_output) / max(alpha_t, 1e-8)
        elif self.prediction_type == 'v_prediction':
            x0 = alpha_t * sample - sigma_t * model_output
        elif self.prediction_type == 'sample':
            x0 = model_output
        else:
            raise ValueError(self.prediction_type)

        # Dynamic thresholding (Imagen)
        if self.use_dynamic_thresholding:
            abs_x0 = np.abs(x0)
            s = np.percentile(abs_x0, self.dynamic_thresholding_ratio * 100)
            s = max(s, 1.0)
            x0 = np.clip(x0, -s, s) / s
        elif self.clip_sample:
            x0 = np.clip(x0, -self.clip_sample_range, self.clip_sample_range)

        return x0.astype(np.float32)

    # ----------------------------------------------------------------
    #  DPM-Solver++ steps
    # ----------------------------------------------------------------

    def _first_order_step(self, data_pred: np.ndarray,
                          sample: np.ndarray,
                          t: int, t_prev: int) -> np.ndarray:
        lam = self.lambda_t[t]
        lam_prev = self.lambda_t[t_prev] if t_prev >= 0 else self.lambda_t[0]
        h = lam_prev - lam
        alpha_prev = self.alpha_t[t_prev] if t_prev >= 0 else np.float32(1.0)
        sigma_prev = self.sigma_t[t_prev] if t_prev >= 0 else np.float32(0.0)
        return (alpha_prev / max(float(self.alpha_t[t]), 1e-8)) * sample \
               - sigma_prev * np.expm1(h) * data_pred

    def _second_order_step(self, model_outputs: list,
                           sample: np.ndarray,
                           t: int, t_prev: int,
                           t_prev2: int) -> np.ndarray:
        lam = self.lambda_t[t]
        lam_prev = self.lambda_t[t_prev] if t_prev >= 0 else self.lambda_t[0]
        h = lam_prev - lam
        alpha_prev = self.alpha_t[t_prev] if t_prev >= 0 else np.float32(1.0)
        sigma_prev = self.sigma_t[t_prev] if t_prev >= 0 else np.float32(0.0)

        m0 = model_outputs[-1]
        m1 = model_outputs[-2] if len(model_outputs) >= 2 else m0
        lam_s1 = self.lambda_t[t_prev2] if t_prev2 >= 0 else lam
        h0 = lam - lam_s1 if lam != lam_s1 else h

        D0 = m0
        D1 = (m0 - m1) / (h0 + 1e-8)

        return ((alpha_prev / max(float(self.alpha_t[t]), 1e-8)) * sample
                - sigma_prev * np.expm1(h) * D0
                - 0.5 * sigma_prev * np.expm1(h) * D1)

    def step(self, model_output: Tensor, timestep: int,
             sample: Tensor) -> Tensor:
        """One reverse step of CogVideoX DPM-Solver++."""
        model_output._ensure_cpu(); sample._ensure_cpu()

        step_idx = int(np.argmin(np.abs(self.timesteps - timestep)))
        t = int(self.timesteps[step_idx])
        t_prev = (int(self.timesteps[step_idx + 1])
                  if step_idx + 1 < len(self.timesteps) else 0)

        # Predict x₀
        data_pred = self._predict_x0(model_output._data, sample._data, t)
        self._model_outputs.append(data_pred)
        if len(self._model_outputs) > self.solver_order:
            self._model_outputs.pop(0)

        order = min(self.solver_order, len(self._model_outputs))

        if order == 1:
            prev = self._first_order_step(data_pred, sample._data, t, t_prev)
        else:
            t_prev2 = (int(self.timesteps[max(step_idx - 1, 0)])
                       if step_idx > 0 else t)
            prev = self._second_order_step(
                self._model_outputs, sample._data, t, t_prev, t_prev2)

        self._lower_order_nums += 1
        return Tensor._wrap(prev.astype(np.float32), False, None,
                            sample._device)

    # ----------------------------------------------------------------
    #  Convenience
    # ----------------------------------------------------------------

    @property
    def init_noise_sigma(self) -> float:
        """Initial noise standard deviation (for latent initialisation)."""
        return float(self.sigma_t[-1]) if len(self.sigma_t) > 0 else 1.0

    def scale_model_input(self, sample: Tensor,
                          timestep: int) -> Tensor:
        """No-op for DPM-Solver++ (model input is not pre-scaled)."""
        return sample


# ═════════════════════════════════════════════════════════════════════
#  EulerDiscreteScheduler
# ═════════════════════════════════════════════════════════════════════

class EulerDiscreteScheduler:
    """Euler method on the ODE probability flow (Karras et al. 2022).

    Fast single-step sampler that works well in 20–50 steps with
    the ancestral variant (``use_karras_sigmas=True``).

    Args:
        num_train_timesteps:  Training timesteps.
        beta_start / beta_end: Beta range.
        beta_schedule:       Schedule type.
        prediction_type:     ``'epsilon'``, ``'v_prediction'``, or ``'sample'``.
        use_karras_sigmas:   If True, remap sigmas using the Karras schedule.
        interpolation_type:  ``'linear'`` or ``'log_linear'`` sigma interpolation.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        prediction_type: str = 'epsilon',
        use_karras_sigmas: bool = False,
        interpolation_type: str = 'linear',
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.use_karras_sigmas = use_karras_sigmas
        self.interpolation_type = interpolation_type

        betas = _get_betas(beta_schedule, num_train_timesteps,
                           beta_start, beta_end)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas).astype(np.float32)

        # sigma = sqrt((1 - alpha_bar) / alpha_bar)
        self.sigmas_full = np.sqrt(
            (1.0 - self.alphas_cumprod) / self.alphas_cumprod
        ).astype(np.float32)

        self.timesteps = np.arange(num_train_timesteps - 1, -1, -1).astype(np.int64)
        self.sigmas = np.append(self.sigmas_full[::-1], 0.0).astype(np.float32)
        self.num_inference_steps: int | None = None

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps
        )
        self.timesteps = np.round(timesteps).astype(np.int64)

        sigmas = np.interp(timesteps, np.arange(len(self.sigmas_full)),
                           self.sigmas_full)
        if self.use_karras_sigmas:
            sigmas = self._karras_sigmas(sigmas, num_inference_steps)
        self.sigmas = np.append(sigmas, 0.0).astype(np.float32)

    @staticmethod
    def _karras_sigmas(sigmas: np.ndarray, n: int,
                       rho: float = 7.0) -> np.ndarray:
        """Karras et al. sigma ramp."""
        s_min = float(sigmas[-1])
        s_max = float(sigmas[0])
        ramp = np.linspace(0, 1, n, dtype=np.float64)
        min_inv = s_min ** (1.0 / rho)
        max_inv = s_max ** (1.0 / rho)
        return (max_inv + ramp * (min_inv - max_inv)) ** rho

    @property
    def init_noise_sigma(self) -> float:
        return float(max(self.sigmas))

    def scale_model_input(self, sample: Tensor, timestep: int) -> Tensor:
        """Pre-scale the model input (required for Euler)."""
        step_idx = int(np.argmin(np.abs(self.timesteps - timestep)))
        sigma = self.sigmas[step_idx]
        sample._ensure_cpu()
        return Tensor._wrap(
            (sample._data / np.sqrt(sigma ** 2 + 1)).astype(np.float32),
            sample._requires_grad, None, sample._device)

    def add_noise(self, original: Tensor, noise: Tensor,
                  timesteps: Tensor) -> Tensor:
        original._ensure_cpu(); noise._ensure_cpu(); timesteps._ensure_cpu()
        t = timesteps._data.astype(int).ravel()
        ndim = original._data.ndim
        s_a = _broadcast_to_ndim(np.sqrt(self.alphas_cumprod[t]), ndim)
        s_s = _broadcast_to_ndim(np.sqrt(1.0 - self.alphas_cumprod[t]), ndim)
        return Tensor._wrap(
            (s_a * original._data + s_s * noise._data).astype(np.float32),
            False, None, original._device)

    def step(self, model_output: Tensor, timestep: int,
             sample: Tensor) -> Tensor:
        """Euler discrete step."""
        model_output._ensure_cpu(); sample._ensure_cpu()
        step_idx = int(np.argmin(np.abs(self.timesteps - timestep)))
        sigma = float(self.sigmas[step_idx])
        sigma_next = float(self.sigmas[step_idx + 1])

        if self.prediction_type == 'epsilon':
            pred_x0 = sample._data - sigma * model_output._data
        elif self.prediction_type == 'v_prediction':
            pred_x0 = model_output._data * (-sigma / np.sqrt(sigma ** 2 + 1)) + \
                       sample._data / (sigma ** 2 + 1)
        elif self.prediction_type == 'sample':
            pred_x0 = model_output._data
        else:
            raise ValueError(self.prediction_type)

        # Derivative
        d = (sample._data - pred_x0) / max(sigma, 1e-8)
        dt = sigma_next - sigma
        prev = sample._data + d * dt

        return Tensor._wrap(prev.astype(np.float32), False, None,
                            sample._device)


# ═════════════════════════════════════════════════════════════════════
#  PNDMScheduler
# ═════════════════════════════════════════════════════════════════════

class PNDMScheduler:
    """Pseudo Numerical Diffusion Model scheduler (Liu et al. 2022).

    Uses a 4th-order linear multi-step method to achieve high-quality
    samples in 20–50 steps with Runge–Kutta warm-up.

    Args:
        num_train_timesteps: Training steps.
        beta_start / beta_end: Beta range.
        beta_schedule:       Schedule type.
        prediction_type:     ``'epsilon'`` or ``'v_prediction'``.
        skip_prk_steps:      Skip Runge–Kutta warm-up (use PLMS from start).
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        prediction_type: str = 'epsilon',
        skip_prk_steps: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.skip_prk_steps = skip_prk_steps

        betas = _get_betas(beta_schedule, num_train_timesteps,
                           beta_start, beta_end)
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)
        self.final_alpha_cumprod = np.float32(1.0)

        self.timesteps = np.arange(num_train_timesteps - 1, -1, -1).astype(np.int64)
        self.num_inference_steps: int | None = None
        self._ets: list[np.ndarray] = []
        self._counter = 0

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (
            np.arange(0, num_inference_steps)[::-1] * step_ratio
        ).astype(np.int64)
        self._ets = []
        self._counter = 0

    def add_noise(self, original: Tensor, noise: Tensor,
                  timesteps: Tensor) -> Tensor:
        original._ensure_cpu(); noise._ensure_cpu(); timesteps._ensure_cpu()
        t = timesteps._data.astype(int).ravel()
        ndim = original._data.ndim
        s_a = _broadcast_to_ndim(np.sqrt(self.alphas_cumprod[t]), ndim)
        s_s = _broadcast_to_ndim(np.sqrt(1.0 - self.alphas_cumprod[t]), ndim)
        return Tensor._wrap(
            (s_a * original._data + s_s * noise._data).astype(np.float32),
            False, None, original._device)

    def _get_prev_sample(self, sample: np.ndarray, t: int,
                         t_prev: int, model_output: np.ndarray) -> np.ndarray:
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = (self.alphas_cumprod[t_prev]
                          if t_prev >= 0 else self.final_alpha_cumprod)

        if self.prediction_type == 'v_prediction':
            model_output = (np.sqrt(alpha_bar_t) * model_output
                            + np.sqrt(1 - alpha_bar_t) * sample)

        pred_x0 = (sample - np.sqrt(1 - alpha_bar_t) * model_output
                    ) / np.sqrt(alpha_bar_t)
        pred_dir = np.sqrt(1 - alpha_bar_prev) * model_output
        return np.sqrt(alpha_bar_prev) * pred_x0 + pred_dir

    def step(self, model_output: Tensor, timestep: int,
             sample: Tensor) -> Tensor:
        """PLMS (linear multi-step) step with Runge–Kutta warm-up."""
        model_output._ensure_cpu(); sample._ensure_cpu()
        step_idx = int(np.argmin(np.abs(self.timesteps - timestep)))
        t = int(self.timesteps[step_idx])
        t_prev = (int(self.timesteps[step_idx + 1])
                  if step_idx + 1 < len(self.timesteps) else 0)

        self._ets.append(model_output._data.copy())
        if len(self._ets) > 4:
            self._ets.pop(0)

        n = len(self._ets)
        if n == 1:
            et = self._ets[-1]
        elif n == 2:
            et = (3 * self._ets[-1] - self._ets[-2]) / 2
        elif n == 3:
            et = (23 * self._ets[-1] - 16 * self._ets[-2] +
                  5 * self._ets[-3]) / 12
        else:
            et = (55 * self._ets[-1] - 59 * self._ets[-2] +
                  37 * self._ets[-3] - 9 * self._ets[-4]) / 24

        prev = self._get_prev_sample(sample._data, t, t_prev, et)
        self._counter += 1
        return Tensor._wrap(prev.astype(np.float32), False, None,
                            sample._device)


# ═════════════════════════════════════════════════════════════════════
#  FlowMatchEulerDiscreteScheduler
# ═════════════════════════════════════════════════════════════════════

class FlowMatchEulerDiscreteScheduler:
    """Flow-Matching Euler discrete scheduler (Lipman et al. 2023).

    Implements the conditional optimal-transport flow matching objective
    with an Euler ODE solver.  Used by Stable Diffusion 3, FLUX, and
    similar rectified-flow models.

    The forward interpolant is:   x_t  = (1 − t) · x₀  +  t · ε
    The velocity field is:        v(x_t, t) ≈ ε − x₀

    Args:
        num_train_timesteps:  Number of training time discretisation steps.
        shift:                Shift factor for the time schedule (SD3 default = 3.0).
        use_dynamic_shifting: If True, adapt shift based on image resolution.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting

        self.timesteps = np.linspace(1.0, 0.0, num_train_timesteps,
                                     dtype=np.float32)
        self.sigmas = self.timesteps.copy()
        self.num_inference_steps: int | None = None

    def set_timesteps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(1.0, 0.0, num_inference_steps + 1,
                                dtype=np.float64)

        # Apply shift
        if self.shift != 1.0:
            timesteps = self.shift * timesteps / (
                1 + (self.shift - 1) * timesteps
            )

        self.sigmas = timesteps.astype(np.float32)
        # Timesteps in integer form for the model (mapped to [0, T-1])
        self.timesteps = (
            timesteps[:-1] * self.num_train_timesteps
        ).astype(np.int64)

    @property
    def init_noise_sigma(self) -> float:
        return 1.0

    def scale_model_input(self, sample: Tensor, timestep: int) -> Tensor:
        """No pre-scaling for flow matching."""
        return sample

    def add_noise(self, original: Tensor, noise: Tensor,
                  timesteps: Tensor) -> Tensor:
        """Flow matching forward: x_t = (1 - t) * x_0 + t * noise."""
        original._ensure_cpu(); noise._ensure_cpu(); timesteps._ensure_cpu()
        t = timesteps._data.astype(np.float32).ravel()
        # Map integer timesteps to continuous t in [0, 1]
        t_cont = t / self.num_train_timesteps
        ndim = original._data.ndim
        t_b = _broadcast_to_ndim(t_cont, ndim)
        noisy = (1.0 - t_b) * original._data + t_b * noise._data
        return Tensor._wrap(noisy.astype(np.float32), False, None,
                            original._device)

    def step(self, model_output: Tensor, timestep: int,
             sample: Tensor) -> Tensor:
        """Euler step along the learned velocity field."""
        model_output._ensure_cpu(); sample._ensure_cpu()

        step_idx = int(np.argmin(np.abs(self.timesteps - timestep)))
        sigma = float(self.sigmas[step_idx])
        sigma_next = float(self.sigmas[step_idx + 1])
        dt = sigma_next - sigma

        # v-prediction velocity field: dx/dt = model_output
        prev = sample._data + dt * model_output._data

        return Tensor._wrap(prev.astype(np.float32), False, None,
                            sample._device)


# ═════════════════════════════════════════════════════════════════════
#  Public exports
# ═════════════════════════════════════════════════════════════════════

__all__ = [
    'DDPMScheduler',
    'DDIMScheduler',
    'DPMSolverMultistepScheduler',
    'CogVideoXDPMScheduler',
    'EulerDiscreteScheduler',
    'PNDMScheduler',
    'FlowMatchEulerDiscreteScheduler',
]
