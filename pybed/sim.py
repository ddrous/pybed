"""Batched Torch simulators used by the built-in environments."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F


def location(
    theta: torch.Tensor,
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    strength: float | torch.Tensor = 1.0,
    background: float = 0.1,
    softening: float = 1e-4,
    noise: float = 0.5,
    log_signal: bool = True,
    source_mask: torch.Tensor | None = None,
    theta_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Take source locations and sensor positions and return noisy inverse-square signals."""
    theta, x = torch.as_tensor(theta), torch.as_tensor(x)
    difference2 = (theta - x.unsqueeze(-2)).square()
    if theta_mask is not None:
        difference2 = difference2 * torch.as_tensor(theta_mask, dtype=theta.dtype, device=theta.device)
    distance2 = difference2.sum(-1)
    strength = torch.as_tensor(strength, dtype=theta.dtype, device=theta.device)
    signal = strength / (softening + distance2)
    if source_mask is not None:
        signal = signal * torch.as_tensor(source_mask, dtype=theta.dtype, device=theta.device)
    mean = background + signal.sum(-1)
    mean = mean.clamp_min(torch.finfo(mean.dtype).tiny)
    if log_signal:
        mean = mean.log()
    if noise:
        mean = mean + noise * torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    return mean.unsqueeze(-1)


def image_mask(
    theta: torch.Tensor,
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    half_width: float = 3.5,
    smooth: float = 0.1,
    noise: float = 1e-3,
) -> torch.Tensor:
    """Take images and mask centres in ``[0,1]^2`` and return smooth masked images."""
    theta, x = torch.as_tensor(theta), torch.as_tensor(x)
    height, width = theta.shape[-2:]
    leading = theta.shape[:-3]
    if tuple(x.shape[:-1]) != tuple(leading):
        raise ValueError(f"x batch shape {x.shape[:-1]} must match image batch shape {leading}")
    yy = torch.arange(height, dtype=theta.dtype, device=theta.device)
    xx = torch.arange(width, dtype=theta.dtype, device=theta.device)
    center_y = x[..., 0] * (height - 1)
    center_x = x[..., 1] * (width - 1)
    while center_y.ndim < theta.ndim:
        center_y, center_x = center_y.unsqueeze(-1), center_x.unsqueeze(-1)
    yy = yy.reshape(*(1 for _ in leading), 1, height, 1)
    xx = xx.reshape(*(1 for _ in leading), 1, 1, width)
    vertical = (
        torch.sigmoid((yy - center_y + half_width) / smooth) + torch.sigmoid((center_y + half_width - yy) / smooth) - 1
    )
    horizontal = (
        torch.sigmoid((xx - center_x + half_width) / smooth) + torch.sigmoid((center_x + half_width - xx) / smooth) - 1
    )
    mean = theta * vertical * horizontal
    if noise:
        mean = mean + noise * torch.rand(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    return mean.clamp(0, 1)


def image_patch(
    theta: torch.Tensor,
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    patch_size: int = 5,
    noise: float = 0.1,
) -> torch.Tensor:
    """Take images and continuous patch corners in ``[0,1]^2`` and return noisy sampled patches."""
    theta, x = torch.as_tensor(theta), torch.as_tensor(x)
    leading = theta.shape[:-3]
    channels, height, width = theta.shape[-3:]
    if tuple(x.shape[:-1]) != tuple(leading):
        raise ValueError(f"x batch shape {x.shape[:-1]} must match image batch shape {leading}")
    flat_theta = theta.reshape(-1, channels, height, width)
    flat_x = x.reshape(-1, 2)
    offsets = torch.arange(patch_size, dtype=theta.dtype, device=theta.device)
    rows = flat_x[:, 0, None, None] * (height - 1) + offsets[None, :, None]
    cols = flat_x[:, 1, None, None] * (width - 1) + offsets[None, None, :]
    rows = rows.expand(-1, patch_size, patch_size)
    cols = cols.expand(-1, patch_size, patch_size)
    grid = torch.stack((2 * cols / max(1, width - 1) - 1, 2 * rows / max(1, height - 1) - 1), -1)
    patches = F.grid_sample(flat_theta, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    patches = patches.reshape(*leading, channels, patch_size, patch_size)
    if noise:
        patches = patches + noise * torch.randn(
            patches.shape, dtype=patches.dtype, device=patches.device, generator=generator
        )
    return patches


def advdiff(
    theta: torch.Tensor,
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    times: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    velocity: tuple[float, float] | torch.Tensor | Callable[[torch.Tensor], torch.Tensor] = (0.15, 0.0),
    diffusion: float = 0.003,
    reaction: float = 0.0,
    dt: float = 0.01,
    noise: float = 0.01,
    boundary: str = "replicate",
) -> torch.Tensor:
    """Take an initial field and sensor locations and return simulated sensor time series.

    The unknown ``theta`` is an initial field ``(..., H, W)`` or
    ``(..., 1, H, W)``. ``x`` holds one or more sensor locations with shape
    ``(..., sensors, 2)`` in ``[0, 1]^2``.  A semi-Lagrangian advection step and
    explicit five-point diffusion step advance

    ``u_t - kappa Laplacian(u) + v dot grad(u) + reaction*u = 0``.

    It returns the time series ``(..., sensors, len(times))``.  The solver is
    reusable for any initial-condition inverse problem; sensor placement and PDE
    discretization are not tied to the environment wrapper.
    """
    theta, x = torch.as_tensor(theta), torch.as_tensor(x)
    had_channel = theta.ndim >= 3 and theta.shape[-3] == 1
    field = theta if had_channel else theta.unsqueeze(-3)
    leading = field.shape[:-3]
    height, width = field.shape[-2:]
    sensors = x.shape[-2]
    if tuple(x.shape[:-2]) != tuple(leading):
        raise ValueError(f"x batch shape {x.shape[:-2]} must match field batch shape {leading}")
    flat = field.reshape(-1, 1, height, width)
    flat_x = x.reshape(-1, sensors, 2)
    batch = flat.shape[0]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, dtype=flat.dtype, device=flat.device),
        torch.linspace(-1, 1, width, dtype=flat.dtype, device=flat.device),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), -1).expand(batch, -1, -1, -1)
    laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=flat.dtype, device=flat.device).reshape(1, 1, 3, 3)
    max_time = float(max(times))
    steps = max(1, math.ceil(max_time / dt))
    actual_dt = max_time / steps
    dx = 1.0 / max(height - 1, width - 1)
    if diffusion * actual_dt / dx**2 > 0.24:
        raise ValueError("Explicit diffusion step is unstable; reduce dt or diffusion, or use a coarser grid")
    sample_steps = {max(1, round(float(t) / actual_dt)): i for i, t in enumerate(times)}
    observations: list[torch.Tensor | None] = [None] * len(times)

    for step in range(1, steps + 1):
        time = torch.as_tensor((step - 1) * actual_dt, dtype=flat.dtype, device=flat.device)
        current_velocity = velocity(time) if callable(velocity) else velocity
        if torch.is_tensor(current_velocity) and current_velocity.ndim >= 3:
            vel = current_velocity.to(flat).reshape(-1, 2, height, width)
            shift = torch.stack((2 * actual_dt * vel[:, 0], 2 * actual_dt * vel[:, 1]), -1)
            grid = base_grid - shift
        else:
            vel = torch.as_tensor(current_velocity, dtype=flat.dtype, device=flat.device)
            shift = torch.stack((2 * actual_dt * vel[0], 2 * actual_dt * vel[1]))
            grid = base_grid - shift
        advected = F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=True)
        padded = F.pad(advected, (1, 1, 1, 1), mode=boundary)
        flat = advected + actual_dt * (diffusion * F.conv2d(padded, laplace) / dx**2 - reaction * advected)
        if step in sample_steps:
            sensor_grid = flat_x.mul(2).sub(1).flip(-1).reshape(batch, sensors, 1, 2)
            values = F.grid_sample(flat, sensor_grid, mode="bilinear", padding_mode="border", align_corners=True)
            observations[sample_steps[step]] = values[:, 0, :, 0]

    result = torch.stack([value for value in observations if value is not None], -1)
    result = result.reshape(*leading, sensors, len(times))
    if noise:
        result = result + noise * torch.randn(
            result.shape, dtype=result.dtype, device=result.device, generator=generator
        )
    return result


def sde(
    state: torch.Tensor,
    drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    diffusion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | float,
    *,
    dt: float,
    steps: int,
    generator: torch.Generator | None = None,
    keep_every: int = 1,
) -> torch.Tensor:
    """Take an initial state and SDE terms and return an Euler--Maruyama trajectory."""
    state = torch.as_tensor(state)
    trajectory = [state]
    for step in range(steps):
        time = torch.as_tensor(step * dt, dtype=state.dtype, device=state.device)
        scale = diffusion(state, time) if callable(diffusion) else diffusion
        noise = torch.randn(state.shape, dtype=state.dtype, device=state.device, generator=generator)
        state = state + drift(state, time) * dt + torch.as_tensor(scale, device=state.device) * math.sqrt(dt) * noise
        if (step + 1) % keep_every == 0:
            trajectory.append(state)
    return torch.stack(trajectory, -2)


def pendulum(
    theta: torch.Tensor,
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    dt: float = 0.02,
    steps: int = 100,
    keep_every: int = 10,
    process_noise: float = 0.02,
    obs_noise: float = 0.01,
) -> torch.Tensor:
    """Take pendulum parameters and initial states and return noisy angle trajectories.

    ``theta[..., 0:2]`` stores gravity and damping; ``x[..., 0:2]`` stores
    initial angle and angular velocity. The observation is a retained angle time
    series, making the simulator suitable for likelihood-free BED examples.
    """
    theta, x = torch.as_tensor(theta), torch.as_tensor(x)
    gravity, damping = theta[..., 0], theta[..., 1]

    def dynamics(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Take pendulum state and time and return its instantaneous derivative."""
        angle, velocity = state[..., 0], state[..., 1]
        return torch.stack((velocity, -gravity * torch.sin(angle) - damping * velocity), -1)

    trajectory = sde(
        x[..., :2],
        dynamics,
        torch.tensor([0.0, process_noise], device=theta.device),
        dt=dt,
        steps=steps,
        keep_every=keep_every,
        generator=generator,
    )[..., 0]
    if obs_noise:
        trajectory = trajectory + obs_noise * torch.randn(
            trajectory.shape, dtype=trajectory.dtype, device=trajectory.device, generator=generator
        )
    return trajectory


def ces(
    theta: torch.Tensor,
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    noise: float = 0.05,
) -> torch.Tensor:
    """Take CES parameters and two baskets and return a noisy preference.

    ``x`` concatenates two K-good baskets. ``theta`` contains rho, K
    simplex weights and a utility scale. The returned response is in ``[0, 1]``.
    """
    theta, x = torch.as_tensor(theta), torch.as_tensor(x)
    goods = x.shape[-1] // 2
    rho = theta[..., :1].clamp(0.05, 1.0)
    alpha = theta[..., 1 : 1 + goods].clamp_min(1e-6)
    alpha = alpha / alpha.sum(-1, keepdim=True)
    utility_scale = theta[..., 1 + goods : 2 + goods].exp()
    left, right = x[..., :goods].clamp_min(1e-6), x[..., goods:].clamp_min(1e-6)
    utility_left = (alpha * left.pow(rho)).sum(-1, keepdim=True).pow(1 / rho)
    utility_right = (alpha * right.pow(rho)).sum(-1, keepdim=True).pow(1 / rho)
    mean = torch.sigmoid((utility_left - utility_right) / utility_scale.clamp_min(1e-4))
    if noise:
        mean = mean + noise * torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    return mean.clamp(0, 1)


def death(
    theta: torch.Tensor,
    x: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    population: int = 50,
) -> torch.Tensor:
    """Take death rates and observation times and return affected population counts."""
    theta, x = torch.as_tensor(theta), torch.as_tensor(x)
    rate = theta[..., :1].clamp_min(1e-6)
    time = x[..., :1].clamp_min(0)
    uniforms = torch.rand((*rate.shape[:-1], population), dtype=rate.dtype, device=rate.device, generator=generator)
    event_times = -uniforms.clamp_min(1e-8).log() / rate
    infected = (event_times <= time).sum(-1, keepdim=True)
    return infected.to(rate.dtype)
