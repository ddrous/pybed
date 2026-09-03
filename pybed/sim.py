"""Reference simulators and numerical building blocks.

All built-ins are batched PyTorch functions with explicit generators.  They are
deliberately ordinary callables: users may call them directly, wrap them, or
replace an environment's simulator without inheriting from a framework class.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F


def location(
    theta: torch.Tensor,
    design: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    strength: float | torch.Tensor = 1.0,
    background: float = 0.1,
    softening: float = 1e-4,
    noise: float = 0.5,
    log_signal: bool = True,
) -> torch.Tensor:
    """Inverse-square multi-source location-finding simulator.

    Parameters
    ----------
    theta:
        Source positions with shape ``(..., sources, dimensions)``.
    design:
        Sensor position with shape ``(..., dimensions)``.
    Returns
    -------
    Tensor
        One noisy (log-)intensity per batch item, shape ``(..., 1)``.
    """
    theta, design = torch.as_tensor(theta), torch.as_tensor(design)
    distance2 = (theta - design.unsqueeze(-2)).square().sum(-1)
    strength = torch.as_tensor(strength, dtype=theta.dtype, device=theta.device)
    mean = background + (strength / (softening + distance2)).sum(-1)
    mean = mean.clamp_min(torch.finfo(mean.dtype).tiny)
    if log_signal:
        mean = mean.log()
    if noise:
        mean = mean + noise * torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    return mean.unsqueeze(-1)


def image_mask(
    theta: torch.Tensor,
    design: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    half_width: float = 3.5,
    smooth: float = 0.1,
    noise: float = 1e-3,
) -> torch.Tensor:
    """JADAI/CoDiff-style smooth image-discovery measurement.

    ``theta`` has shape ``(..., C, H, W)`` and the continuous design is in
    normalized ``[0, 1]^2`` coordinates.  The result is a full noisy image whose
    useful signal is localized by a differentiable bivariate-logistic mask.
    """
    theta, design = torch.as_tensor(theta), torch.as_tensor(design)
    height, width = theta.shape[-2:]
    leading = theta.shape[:-3]
    if tuple(design.shape[:-1]) != tuple(leading):
        raise ValueError(f"design batch shape {design.shape[:-1]} must match image batch shape {leading}")
    yy = torch.arange(height, dtype=theta.dtype, device=theta.device)
    xx = torch.arange(width, dtype=theta.dtype, device=theta.device)
    center_y = design[..., 0] * (height - 1)
    center_x = design[..., 1] * (width - 1)
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


def advdiff(
    theta: torch.Tensor,
    design: torch.Tensor,
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
    """Differentiable 2-D advection-diffusion(-reaction) solver and sensor map.

    The unknown ``theta`` is an initial field ``(..., H, W)`` or
    ``(..., 1, H, W)``. ``design`` holds one or more sensor locations with shape
    ``(..., sensors, 2)`` in ``[0, 1]^2``.  A semi-Lagrangian advection step and
    explicit five-point diffusion step advance

    ``u_t - kappa Laplacian(u) + v dot grad(u) + reaction*u = 0``.

    It returns the time series ``(..., sensors, len(times))``.  The solver is
    reusable for any initial-condition inverse problem; sensor placement and PDE
    discretization are not tied to the environment wrapper.
    """
    theta, design = torch.as_tensor(theta), torch.as_tensor(design)
    had_channel = theta.ndim >= 3 and theta.shape[-3] == 1
    field = theta if had_channel else theta.unsqueeze(-3)
    leading = field.shape[:-3]
    height, width = field.shape[-2:]
    sensors = design.shape[-2]
    if tuple(design.shape[:-2]) != tuple(leading):
        raise ValueError(f"design batch shape {design.shape[:-2]} must match field batch shape {leading}")
    flat = field.reshape(-1, 1, height, width)
    flat_design = design.reshape(-1, sensors, 2)
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
            sensor_grid = flat_design.mul(2).sub(1).flip(-1).reshape(batch, sensors, 1, 2)
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
    """Batched Euler-Maruyama integrator returning the full retained trajectory."""
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
    design: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    dt: float = 0.02,
    steps: int = 100,
    keep_every: int = 10,
    process_noise: float = 0.02,
    obs_noise: float = 0.01,
) -> torch.Tensor:
    """Stochastic pendulum with unknown gravity and damping.

    ``theta[..., 0:2]`` stores gravity and damping; ``design[..., 0:2]`` stores
    initial angle and angular velocity. The observation is a retained angle time
    series, making the simulator suitable for likelihood-free BED examples.
    """
    theta, design = torch.as_tensor(theta), torch.as_tensor(design)
    gravity, damping = theta[..., 0], theta[..., 1]

    def dynamics(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        angle, velocity = state[..., 0], state[..., 1]
        return torch.stack((velocity, -gravity * torch.sin(angle) - damping * velocity), -1)

    trajectory = sde(
        design[..., :2],
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
    design: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    noise: float = 0.05,
) -> torch.Tensor:
    """Constant-elasticity-of-substitution preference experiment.

    ``design`` concatenates two K-good baskets. ``theta`` contains rho, K
    simplex weights and a utility scale. The returned response is in ``[0, 1]``.
    """
    theta, design = torch.as_tensor(theta), torch.as_tensor(design)
    goods = design.shape[-1] // 2
    rho = theta[..., :1].clamp(0.05, 1.0)
    alpha = theta[..., 1 : 1 + goods].clamp_min(1e-6)
    alpha = alpha / alpha.sum(-1, keepdim=True)
    utility_scale = theta[..., 1 + goods : 2 + goods].exp()
    left, right = design[..., :goods].clamp_min(1e-6), design[..., goods:].clamp_min(1e-6)
    utility_left = (alpha * left.pow(rho)).sum(-1, keepdim=True).pow(1 / rho)
    utility_right = (alpha * right.pow(rho)).sum(-1, keepdim=True).pow(1 / rho)
    mean = torch.sigmoid((utility_left - utility_right) / utility_scale.clamp_min(1e-4))
    if noise:
        mean = mean + noise * torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    return mean.clamp(0, 1)


def death(
    theta: torch.Tensor,
    design: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    population: int = 50,
) -> torch.Tensor:
    """Likelihood-free pure-death process observed at a chosen time."""
    theta, design = torch.as_tensor(theta), torch.as_tensor(design)
    rate = theta[..., :1].clamp_min(1e-6)
    time = design[..., :1].clamp_min(0)
    uniforms = torch.rand((*rate.shape[:-1], population), dtype=rate.dtype, device=rate.device, generator=generator)
    event_times = -uniforms.clamp_min(1e-8).log() / rate
    infected = (event_times <= time).sum(-1, keepdim=True)
    return infected.to(rate.dtype)
