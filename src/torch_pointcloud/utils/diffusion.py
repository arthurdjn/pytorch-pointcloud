r"""Denoising diffusion schedules and samplers.

Minimal, dependency-free implementation of the DDIM sampler from
:arxiv: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), matching the
`diffusers`-style API (`add_noise` / `get_velocity` / `set_timesteps` / `step`) used by latent diffusion
models such as XCube.
"""

from typing import Optional, Union

import torch
from torch import Tensor


class DDIMScheduler:
    r"""DDIM noise schedule and sampling step.

    Implements the deterministic-to-stochastic DDIM update of formula (12) in
    :arxiv: [DDIM](https://arxiv.org/abs/2010.02502) over a linear $\beta$ schedule, with `epsilon`,
    `sample` and `v_prediction` parameterizations.

    Args:
        num_train_timesteps: Number of diffusion steps $T$ used at training time.
        beta_start: First value of the linear $\beta$ schedule.
        beta_end: Last value of the linear $\beta$ schedule.
        prediction_type: Quantity predicted by the model, one of `"epsilon"`, `"sample"` or
            `"v_prediction"`.
        set_alpha_to_one: Use $\bar\alpha_{-1} = 1$ for the final denoising step instead of
            $\bar\alpha_0$.

    Example:
        ```python
        scheduler = DDIMScheduler(prediction_type="v_prediction")
        scheduler.set_timesteps(100)
        sample = torch.randn(1024, 8)
        for t in scheduler.timesteps:
            sample = scheduler.step(model(sample, t), int(t), sample)
        ```
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        set_alpha_to_one: bool = True,
    ) -> None:
        if prediction_type not in ("epsilon", "sample", "v_prediction"):
            raise ValueError(f"Unknown prediction type {prediction_type!r}.")
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.num_inference_steps: Optional[int] = None
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device, None] = None) -> None:
        r"""Select the `num_inference_steps` evenly spaced timesteps used for sampling.

        Args:
            num_inference_steps: Number of denoising steps.
            device: Device for the timestep tensor.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(num_inference_steps) * step_ratio).round().flip(0).long()
        self.timesteps = timesteps.to(device)

    def _predict_x0(self, model_output: Tensor, timestep: int, sample: Tensor) -> tuple[Tensor, Tensor]:
        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1.0 - alpha_prod_t
        if self.prediction_type == "epsilon":
            x0 = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
            eps = model_output
        elif self.prediction_type == "sample":
            x0 = model_output
            eps = (sample - alpha_prod_t**0.5 * x0) / beta_prod_t**0.5
        else:
            x0 = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
            eps = alpha_prod_t**0.5 * model_output + beta_prod_t**0.5 * sample
        return x0, eps

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        eta: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        r"""Run one reverse diffusion step $x_t \rightarrow x_{t-1}$.

        Args:
            model_output: Model prediction at `timestep` (interpreted per `prediction_type`).
            timestep: Current discrete timestep $t$.
            sample: Current sample $x_t$.
            eta: Noise scale $\eta$ of formula (16); $\eta = 0$ is deterministic DDIM, $\eta = 1$
                matches DDPM-level stochasticity.
            generator: Random generator for the added noise.

        Returns:
            The previous sample $x_{t-1}$.

        Shape:
            - `model_output`: $(N, C)$ or any shape.
            - `sample`: same shape as `model_output`.
            - Output: same shape as `sample`.
        """
        if self.num_inference_steps is None:
            raise RuntimeError("Call `set_timesteps` before `step`.")

        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1.0 - alpha_prod_t

        x0, eps = self._predict_x0(model_output, timestep, sample)

        variance = (1.0 - alpha_prod_prev) / beta_prod_t * (1.0 - alpha_prod_t / alpha_prod_prev)
        std_dev = eta * variance**0.5

        direction = (1.0 - alpha_prod_prev - std_dev**2) ** 0.5 * eps
        prev_sample = alpha_prod_prev**0.5 * x0 + direction
        if eta > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + std_dev * noise
        return prev_sample

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        r"""Diffuse `original_samples` to the given timesteps (forward process).

        Args:
            original_samples: Clean samples $x_0$.
            noise: Gaussian noise of the same shape.
            timesteps: Per-row timesteps.

        Returns:
            The noisy samples $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$.

        Shape:
            - `original_samples`, `noise`: $(N, C)$ or any shape.
            - `timesteps`: $(N,)$ or broadcastable to the leading dimension.
            - Output: same shape as `original_samples`.
        """
        alphas = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        sqrt_alpha = alphas[timesteps] ** 0.5
        sqrt_one_minus = (1.0 - alphas[timesteps]) ** 0.5
        while sqrt_alpha.dim() < original_samples.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * original_samples + sqrt_one_minus * noise

    def get_velocity(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        r"""Compute the `v_prediction` target $v_t = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$.

        Args:
            sample: Clean samples $x_0$.
            noise: Gaussian noise of the same shape.
            timesteps: Per-row timesteps.

        Returns:
            The velocity target.

        Shape:
            - `sample`, `noise`: $(N, C)$ or any shape.
            - `timesteps`: $(N,)$ or broadcastable to the leading dimension.
            - Output: same shape as `sample`.
        """
        alphas = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        sqrt_alpha = alphas[timesteps] ** 0.5
        sqrt_one_minus = (1.0 - alphas[timesteps]) ** 0.5
        while sqrt_alpha.dim() < sample.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * noise - sqrt_one_minus * sample
