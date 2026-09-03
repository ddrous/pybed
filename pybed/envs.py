"""Built-in BED environment factories.

Factories expose paper-faithful defaults where practical, but every numerical
choice remains an argument and every returned component is replaceable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import sim, vs
from .core import BED, Normal, Spec, Uniform, register


@dataclass(frozen=True)
class ImageBankPrior:
    """Uniform empirical prior over images, retaining optional class labels."""

    images: torch.Tensor
    labels: torch.Tensor | None = None

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> torch.Tensor:
        count = int(np.prod(shape)) if shape else 1
        indices = torch.randint(len(self.images), (count,), generator=generator)
        return self.images[indices].reshape(*shape, *self.images.shape[1:])

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.full(value.shape[:-3], -np.log(len(self.images)), dtype=value.dtype, device=value.device)

    def classify(self, value: torch.Tensor) -> torch.Tensor:
        if self.labels is None:
            raise RuntimeError("This image bank has no labels")
        flat = value.reshape(-1, value.shape[-3] * value.shape[-2] * value.shape[-1])
        bank = self.images.to(value).reshape(len(self.images), -1)
        nearest = torch.cdist(flat, bank).argmin(-1)
        return self.labels.to(value.device)[nearest].reshape(value.shape[:-3])


@dataclass(frozen=True)
class FieldPrior:
    """Smooth Gaussian random-field sampler for PDE inverse problems."""

    shape: tuple[int, int] = (24, 24)
    length_scale: float = 0.12
    mean: float = 0.0
    std: float = 1.0

    def sample(self, sample_shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> torch.Tensor:
        height, width = self.shape
        count = int(np.prod(sample_shape)) if sample_shape else 1
        noise = torch.randn((count, 1, height, width), generator=generator)
        sigma = max(0.6, self.length_scale * max(height, width))
        radius = max(1, int(3 * sigma))
        x = torch.arange(-radius, radius + 1, dtype=noise.dtype)
        kernel1 = torch.exp(-0.5 * (x / sigma).square())
        kernel1 = kernel1 / kernel1.sum()
        kernel2 = torch.outer(kernel1, kernel1).reshape(1, 1, len(kernel1), len(kernel1))
        smooth = F.conv2d(F.pad(noise, (radius,) * 4, mode="reflect"), kernel2)
        smooth = (smooth - smooth.mean((-2, -1), keepdim=True)) / smooth.std((-2, -1), keepdim=True).clamp_min(1e-6)
        return (self.mean + self.std * smooth).reshape(*sample_shape, 1, height, width)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FieldPrior is sampler-defined; use simulation-based EIG or supply an explicit prior")


def location(
    *,
    sources: int = 1,
    dims: int = 2,
    low: float = 0.0,
    high: float = 1.0,
    budget: int = 30,
    prior: Any = None,
    strength: float = 1.0,
    background: float = 0.1,
    softening: float = 1e-4,
    noise: float = 0.5,
    log_signal: bool = True,
) -> BED:
    """Location-finding benchmark used by DAD, iDAD, ALINE, and JADAI."""
    prior = prior or Uniform(torch.full((sources, dims), low), torch.full((sources, dims), high))

    def simulator(theta: torch.Tensor, design: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return sim.location(
            theta,
            design,
            generator=generator,
            strength=strength,
            background=background,
            softening=softening,
            noise=noise,
            log_signal=log_signal,
        )

    return BED(
        "location-v0",
        prior,
        simulator,
        specs={
            "theta": Spec((sources, dims), labels=tuple(f"source {k + 1}" for k in range(sources))),
            "design": Spec((dims,), low=low, high=high),
            "obs": Spec((1,)),
        },
        budget=budget,
        visualizers={"episode": vs.location, "prior": vs.prior},
        cfg=dict(
            sources=sources,
            dims=dims,
            low=low,
            high=high,
            strength=strength,
            background=background,
            softening=softening,
            noise=noise,
            log_signal=log_signal,
        ),
    )


def mnist(
    *,
    images: np.ndarray | torch.Tensor | None = None,
    labels: np.ndarray | torch.Tensor | None = None,
    root: str | Path = "data",
    train: bool = True,
    download: bool = False,
    budget: int = 6,
    half_width: float = 3.5,
    smooth: float = 0.1,
    noise: float = 1e-3,
) -> BED:
    """MNIST image discovery/reconstruction with optional classification labels.

    Passing arrays keeps the environment dataset-agnostic.  If arrays are omitted,
    torchvision is used lazily and never downloads data unless ``download=True``.
    """
    if images is None:
        try:
            from torchvision.datasets import MNIST
        except ImportError as error:
            raise ImportError("Install pybed[mnist] or pass images= and labels= arrays") from error
        dataset = MNIST(str(root), train=train, download=download)
        images = dataset.data[:, None].float() / 255
        labels = dataset.targets
    images = torch.as_tensor(images, dtype=torch.get_default_dtype())
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4:
        raise ValueError("images must have shape (N,H,W) or (N,C,H,W)")
    labels_tensor = None if labels is None else torch.as_tensor(labels, dtype=torch.long)
    prior = ImageBankPrior(images, labels_tensor)
    channels, height, width = images.shape[1:]

    def simulator(theta: torch.Tensor, design: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return sim.image_mask(theta, design, generator=generator, half_width=half_width, smooth=smooth, noise=noise)

    return BED(
        "mnist-discovery-v0",
        prior,
        simulator,
        specs={
            "theta": Spec((channels, height, width)),
            "design": Spec((2,), low=0.0, high=1.0, labels=("row", "column")),
            "obs": Spec((channels, height, width)),
        },
        budget=budget,
        visualizers={"episode": vs.image_discovery, "prior": vs.prior},
        cfg=dict(
            image_shape=(channels, height, width),
            half_width=half_width,
            smooth=smooth,
            noise=noise,
            labels=labels_tensor,
        ),
        target=(prior.classify if labels_tensor is not None else lambda theta: theta),
    )


def advdiff(
    *,
    shape: tuple[int, int] = (24, 24),
    sensors: int = 1,
    times: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    velocity: tuple[float, float] = (0.15, 0.0),
    diffusion: float = 0.003,
    reaction: float = 0.0,
    dt: float = 0.01,
    noise: float = 0.01,
    budget: int = 8,
    prior: Any = None,
) -> BED:
    """Initial-field inference from sensor time series in an advection-diffusion PDE."""
    prior = prior or FieldPrior(shape)

    def simulator(theta: torch.Tensor, design: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return sim.advdiff(
            theta,
            design,
            generator=generator,
            times=times,
            velocity=velocity,
            diffusion=diffusion,
            reaction=reaction,
            dt=dt,
            noise=noise,
        )

    return BED(
        "advdiff-v0",
        prior,
        simulator,
        specs={
            "theta": Spec((1, *shape), labels=("initial field",)),
            "design": Spec((sensors, 2), low=0.0, high=1.0, labels=("sensor row", "sensor column")),
            "obs": Spec((sensors, len(times)), labels=tuple(f"t={t:g}" for t in times)),
        },
        budget=budget,
        visualizers={"episode": vs.pde, "prior": vs.prior},
        cfg=dict(
            shape=shape,
            sensors=sensors,
            times=tuple(times),
            velocity=velocity,
            diffusion=diffusion,
            reaction=reaction,
            dt=dt,
            noise=noise,
        ),
    )


def pendulum(
    *,
    budget: int = 6,
    dt: float = 0.02,
    steps: int = 100,
    keep_every: int = 10,
    process_noise: float = 0.02,
    obs_noise: float = 0.01,
) -> BED:
    """Stochastic-pendulum system-identification benchmark."""
    prior = Uniform(torch.tensor([5.0, 0.02]), torch.tensor([15.0, 0.8]))

    def simulator(theta: torch.Tensor, design: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return sim.pendulum(
            theta,
            design,
            generator=generator,
            dt=dt,
            steps=steps,
            keep_every=keep_every,
            process_noise=process_noise,
            obs_noise=obs_noise,
        )

    return BED(
        "pendulum-v0",
        prior,
        simulator,
        specs={
            "theta": Spec((2,), labels=("gravity", "damping")),
            "design": Spec((2,), low=-np.pi, high=np.pi, labels=("initial angle", "angular velocity")),
            "obs": Spec((steps // keep_every + 1,), labels=("angle time series",)),
        },
        budget=budget,
        visualizers={"episode": vs.timeseries, "prior": vs.prior},
        cfg=dict(dt=dt, steps=steps, keep_every=keep_every, process_noise=process_noise, obs_noise=obs_noise),
    )


def ces(*, goods: int = 3, budget: int = 20, noise: float = 0.05) -> BED:
    """Compact CES preference benchmark from the adaptive-design literature."""
    mean = torch.cat((torch.tensor([0.5]), torch.full((goods,), 1 / goods), torch.tensor([1.0])))
    std = torch.cat((torch.tensor([0.2]), torch.full((goods,), 0.25), torch.tensor([0.5])))

    def simulator(theta: torch.Tensor, design: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return sim.ces(theta, design, generator=generator, noise=noise)

    return BED(
        "ces-v0",
        Normal(mean, std),
        simulator,
        specs={"theta": Spec((goods + 2,)), "design": Spec((2 * goods,), low=0, high=100), "obs": Spec((1,))},
        budget=budget,
        visualizers={"episode": vs.timeseries, "prior": vs.prior},
        cfg=dict(goods=goods, noise=noise),
    )


def death(*, population: int = 50, budget: int = 4, max_time: float = 5.0) -> BED:
    """Pure-death process benchmark used by Deep Adaptive Design."""
    prior = Uniform(torch.tensor([0.01]), torch.tensor([2.0]))

    def simulator(theta: torch.Tensor, design: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return sim.death(theta, design, generator=generator, population=population)

    return BED(
        "death-v0",
        prior,
        simulator,
        specs={"theta": Spec((1,), low=0), "design": Spec((1,), low=0, high=max_time), "obs": Spec((1,))},
        budget=budget,
        visualizers={"episode": vs.timeseries, "prior": vs.prior},
        cfg=dict(population=population, max_time=max_time),
    )


for env_name, factory in {
    "location-v0": location,
    "mnist-discovery-v0": mnist,
    "advdiff-v0": advdiff,
    "pendulum-v0": pendulum,
    "ces-v0": ces,
    "death-v0": death,
}.items():
    register(env_name, factory)
