"""
Test for scaffolding.diffusion module and its main components.
"""
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
    classifier_free_guidance,
    rescale_noise_cfg,
    randn_tensor,
    get_beta_schedule,
)

def test_diffusion_imports():
    # Just check that all imports are available
    assert DDPMScheduler is not None
    assert DDIMScheduler is not None
    assert DPMSolverMultistepScheduler is not None
    assert CogVideoXDPMScheduler is not None
    assert EulerDiscreteScheduler is not None
    assert PNDMScheduler is not None
    assert FlowMatchEulerDiscreteScheduler is not None
    assert DiffusionPipeline is not None
    assert StableDiffusionPipeline is not None
    assert CogVideoXPipeline is not None
    assert UNet2DConditionModel is not None
    assert DiTModel is not None
    assert AutoencoderKL is not None
    assert classifier_free_guidance is not None
    assert rescale_noise_cfg is not None
    assert randn_tensor is not None
    assert get_beta_schedule is not None

def test_cogvideoxdpmscheduler_smoke():
    sched = CogVideoXDPMScheduler(
        num_train_timesteps=1000,
        snr_shift_scale=3.0,
        prediction_type='v_prediction',
    )
    sched.set_timesteps(50)
    assert len(sched.timesteps) == 50
    x0 = torch.randn(1, 4, 8, 8)
    noise = torch.randn(1, 4, 8, 8)
    t = torch.tensor([500])
    noisy = sched.add_noise(x0, noise, t)
    assert noisy.shape == x0.shape
    pred = torch.randn(1, 4, 8, 8)
    prev = sched.step(pred, int(sched.timesteps[0]), noisy)
    assert prev.shape == x0.shape

def test_diffusion_schedulers():
    for cls in [DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler,
                EulerDiscreteScheduler, PNDMScheduler, FlowMatchEulerDiscreteScheduler]:
        s = cls()
        s.set_timesteps(20)
        assert len(s.timesteps) == 20

def test_beta_schedule():
    betas = get_beta_schedule('cosine', 1000)
    assert betas.shape[0] == 1000
    assert betas.min() >= 0
    assert betas.max() <= 1

def test_randn_tensor():
    r = randn_tensor((2, 4, 16, 16), seed=42)
    assert r.shape == (2, 4, 16, 16)
