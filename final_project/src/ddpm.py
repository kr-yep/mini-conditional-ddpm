from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, device: torch.device, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alpha_bars = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999)


class DDPM:
    def __init__(
        self,
        model,
        image_size: int,
        channels: int = 3,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device: torch.device | str = "cpu",
    ):
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.device = torch.device(device)

        self.beta_schedule = beta_schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps, device=self.device)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def _extract(self, values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alpha_bars, t, x_start.shape)
        return sqrt_alpha_bar * x_start + sqrt_one_minus * noise

    def training_loss(self, x_start: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, labels)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        labels: torch.Tensor,
        sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        batch_size = labels.shape[0]
        x = torch.randn(
            batch_size,
            self.channels,
            self.image_size,
            self.image_size,
            device=self.device,
            generator=generator,
        )
        if sampling_steps is None or sampling_steps >= self.timesteps:
            time_indices = list(range(self.timesteps - 1, -1, -1))
        else:
            time_indices = torch.linspace(self.timesteps - 1, 0, sampling_steps, device=self.device).long().tolist()

        for i, t_int in enumerate(time_indices):
            t = torch.full((batch_size,), int(t_int), device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t, labels)
            alpha_bar_t = self._extract(self.alpha_bars, t, x.shape)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            if i == len(time_indices) - 1:
                x = pred_x0
                continue

            prev_t_int = int(time_indices[i + 1])
            prev_t = torch.full((batch_size,), prev_t_int, device=self.device, dtype=torch.long)
            alpha_bar_prev = self._extract(self.alpha_bars, prev_t, x.shape)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            sigma = 0.0 if len(time_indices) < self.timesteps else torch.sqrt(self._extract(self.betas, t, x.shape))
            direction = torch.sqrt(torch.clamp(1 - alpha_bar_prev - (sigma if isinstance(sigma, float) else sigma**2), min=0)) * pred_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction
            if not isinstance(sigma, float):
                x = x + sigma * noise
        return x.clamp(-1, 1)
