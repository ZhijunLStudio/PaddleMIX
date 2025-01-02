import paddle
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
class DDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """
    prev_sample: paddle.Tensor
    pred_original_sample: Optional[paddle.Tensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999,
    alpha_transform_type='cosine'):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == 'cosine':

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == 'exp':

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(
            f'Unsupported alpha_transform_type: {alpha_transform_type}')
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return paddle.to_tensor(data=betas, dtype='float32')


def rescale_zero_terminal_snr(alphas_cumprod):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 -
        alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2
    return alphas_bar


class CogVideoXDDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDIMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        set_alpha_to_one (`bool`, defaults to `True`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps: int=1000, beta_start: float=
        0.00085, beta_end: float=0.012, beta_schedule: str='scaled_linear',
        trained_betas: Optional[Union[np.ndarray, List[float]]]=None,
        clip_sample: bool=True, set_alpha_to_one: bool=True, steps_offset:
        int=0, prediction_type: str='epsilon', clip_sample_range: float=1.0,
        sample_max_value: float=1.0, timestep_spacing: str='leading',
        rescale_betas_zero_snr: bool=False, snr_shift_scale: float=3.0):
        if trained_betas is not None:
            self.betas = paddle.to_tensor(data=trained_betas, dtype='float32')
        elif beta_schedule == 'linear':
            self.betas = paddle.linspace(start=beta_start, stop=beta_end,
                num=num_train_timesteps, dtype='float32')
        elif beta_schedule == 'scaled_linear':
            self.betas = paddle.linspace(start=beta_start ** 0.5, stop=
                beta_end ** 0.5, num=num_train_timesteps, dtype='float64') ** 2
        elif beta_schedule == 'squaredcos_cap_v2':
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(
                f'{beta_schedule} is not implemented for {self.__class__}')
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = paddle.cumprod(x=self.alphas, dim=0)
        self.alphas_cumprod = self.alphas_cumprod / (snr_shift_scale + (1 -
            snr_shift_scale) * self.alphas_cumprod)
        if rescale_betas_zero_snr:
            self.alphas_cumprod = rescale_zero_terminal_snr(self.alphas_cumprod
                )
        self.final_alpha_cumprod = paddle.to_tensor(data=1.0
            ) if set_alpha_to_one else self.alphas_cumprod[0]
        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = paddle.to_tensor(data=np.arange(0,
            num_train_timesteps)[::-1].copy().astype(np.int64))

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
            ] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = beta_prod_t_prev / beta_prod_t * (1 - alpha_prod_t /
            alpha_prod_t_prev)
        return variance

    def scale_model_input(self, sample: paddle.Tensor, timestep: Optional[
        int]=None) ->paddle.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f'`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`: {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle maximal {self.config.num_train_timesteps} timesteps.'
                )
        self.num_inference_steps = num_inference_steps
        if self.config.timestep_spacing == 'linspace':
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1,
                num_inference_steps).round()[::-1].copy().astype(np.int64)
        elif self.config.timestep_spacing == 'leading':
            step_ratio = (self.config.num_train_timesteps // self.
                num_inference_steps)
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round(
                )[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == 'trailing':
            step_ratio = (self.config.num_train_timesteps / self.
                num_inference_steps)
            timesteps = np.round(np.arange(self.config.num_train_timesteps,
                0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
                )
        self.timesteps = paddle.to_tensor(data=timesteps)

    def step(self, model_output: paddle.Tensor, timestep: int, sample:
        paddle.Tensor, eta: float=0.0, use_clipped_model_output: bool=False,
        generator=None, variance_noise: Optional[paddle.Tensor]=None,
        return_dict: bool=True) ->Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
                )
        prev_timestep = (timestep - self.config.num_train_timesteps // self
            .num_inference_steps)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
            ] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        if self.config.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
                ) / alpha_prod_t ** 0.5
        elif self.config.prediction_type == 'sample':
            pred_original_sample = model_output
        elif self.config.prediction_type == 'v_prediction':
            pred_original_sample = (alpha_prod_t ** 0.5 * sample - 
                beta_prod_t ** 0.5 * model_output)
        else:
            raise ValueError(
                f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`'
                )
        a_t = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5
        b_t = alpha_prod_t_prev ** 0.5 - alpha_prod_t ** 0.5 * a_t
        prev_sample = a_t * sample + b_t * pred_original_sample
        if not return_dict:
            return prev_sample,
        return DDIMSchedulerOutput(prev_sample=prev_sample,
            pred_original_sample=pred_original_sample)

    def add_noise(self, original_samples: paddle.Tensor, noise: paddle.
        Tensor, timesteps: paddle.Tensor) ->paddle.Tensor:
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.place)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(tuple(sqrt_alpha_prod.shape)) < len(tuple(
            original_samples.shape)):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(axis=-1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(tuple(sqrt_one_minus_alpha_prod.shape)) < len(tuple(
            original_samples.shape)):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(
                axis=-1)
        noisy_samples = (sqrt_alpha_prod * original_samples + 
            sqrt_one_minus_alpha_prod * noise)
        return noisy_samples

    def get_velocity(self, sample: paddle.Tensor, noise: paddle.Tensor,
        timesteps: paddle.Tensor) ->paddle.Tensor:
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.place)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(tuple(sqrt_alpha_prod.shape)) < len(tuple(sample.shape)):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(axis=-1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(tuple(sqrt_one_minus_alpha_prod.shape)) < len(tuple(
            sample.shape)):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(
                axis=-1)
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps