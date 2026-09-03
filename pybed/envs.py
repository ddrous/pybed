"""Build the reference environments with user-visible numerical settings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import sim, vs
from .core import BED, Batch, Normal, Spec, Uniform, register


@dataclass(frozen=True)
class ImageBankPrior:
    """Uniform empirical prior over images, retaining optional class labels."""

    images: torch.Tensor
    labels: torch.Tensor | None = None

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take a leading shape and return images sampled from the bank."""
        count = int(np.prod(shape)) if shape else 1
        indices = torch.randint(len(self.images), (count,), generator=generator)
        return self.images[indices].reshape(*shape, *self.images.shape[1:])

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Take images and return their uniform empirical log mass."""
        return torch.full(value.shape[:-3], -np.log(len(self.images)), dtype=value.dtype, device=value.device)

    def classify(self, value: torch.Tensor) -> torch.Tensor:
        """Take bank images and return the label of each nearest stored image."""
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
        """Take a leading shape and return smooth random-field draws."""
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
        """Take fields and report that this sampler has no implemented density."""
        raise NotImplementedError("FieldPrior is sampler-defined; use simulation-based EIG or supply an explicit prior")


@dataclass(frozen=True)
class VariableLocationPrior:
    """Sample padded location problems with different source counts and dimensions."""

    source_counts: tuple[int, ...]
    dimensions: tuple[int, ...]
    low: float
    high: float

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> Batch:
        """Take a leading shape and return padded locations plus masks describing each draw."""
        if len(shape) != 1:
            raise ValueError("VariableLocationPrior currently expects one batch axis")
        count = shape[0]
        source_choices = torch.as_tensor(self.source_counts)
        dim_choices = torch.as_tensor(self.dimensions)
        num_sources = source_choices[torch.randint(len(source_choices), (count,), generator=generator)]
        source_dim = dim_choices[torch.randint(len(dim_choices), (count,), generator=generator)]
        max_sources, max_dim = max(self.source_counts), max(self.dimensions)
        theta = self.low + (self.high - self.low) * torch.rand(
            (count, max_sources, max_dim), generator=generator
        )
        source_mask = torch.arange(max_sources)[None, :] < num_sources[:, None]
        dimension_mask = torch.arange(max_dim)[None, :] < source_dim[:, None]
        theta_mask = source_mask[:, :, None] & dimension_mask[:, None, :]
        theta = torch.where(theta_mask, theta, 0)
        return Batch(
            theta=theta,
            context={
                "num_sources": num_sources,
                "source_dim": source_dim,
                "source_mask": source_mask,
                "theta_mask": theta_mask,
            },
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Take padded locations and report that their mixed-shape density needs their masks."""
        raise NotImplementedError("Use each draw's masks when evaluating a variable location prior")


def location(
    *,
    sources: int | Sequence[int] = 1,
    dims: int | Sequence[int] = 2,
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
    """Take location-problem settings and return the DAD/ALINE/JADAI benchmark."""
    source_counts = (sources,) if isinstance(sources, int) else tuple(sources)
    dimensions = (dims,) if isinstance(dims, int) else tuple(dims)
    if not source_counts or not dimensions or min(source_counts) < 1 or min(dimensions) < 1:
        raise ValueError("sources and dims must contain positive integers")
    max_sources, max_dim = max(source_counts), max(dimensions)
    if prior is None:
        prior = (
            Uniform(torch.full((max_sources, max_dim), low), torch.full((max_sources, max_dim), high))
            if len(source_counts) == len(dimensions) == 1
            else VariableLocationPrior(source_counts, dimensions, low, high)
        )

    def simulator(
        theta: torch.Tensor,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
        source_mask: torch.Tensor | None = None,
        theta_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Take source locations, designs, and masks and return noisy signals."""
        return sim.location(
            theta,
            x,
            generator=generator,
            strength=strength,
            background=background,
            softening=softening,
            noise=noise,
            log_signal=log_signal,
            source_mask=source_mask,
            theta_mask=theta_mask,
        )

    def likelihood(
        y: torch.Tensor,
        theta: torch.Tensor,
        x: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        theta_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Take signals, source locations, designs, and masks and return Gaussian log likelihoods."""
        if noise <= 0:
            raise RuntimeError("location log_prob requires noise > 0")
        mean = sim.location(
            theta,
            x,
            strength=strength,
            background=background,
            softening=softening,
            noise=0,
            log_signal=log_signal,
            source_mask=source_mask,
            theta_mask=theta_mask,
        )
        return torch.distributions.Normal(mean, noise).log_prob(y).sum(-1)

    return BED(
        "location-v0",
        prior,
        simulator,
        specs={
            "theta": Spec((max_sources, max_dim), labels=tuple(f"source {k + 1}" for k in range(max_sources))),
            "x": Spec((max_dim,), low=low, high=high),
            "y": Spec((1,)),
        },
        budget=budget,
        likelihood=likelihood,
        visualizers={"episode": vs.location, "prior": vs.prior},
        cfg=dict(
            sources=source_counts[0] if len(source_counts) == 1 else source_counts,
            dims=dimensions[0] if len(dimensions) == 1 else dimensions,
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
    """Take MNIST arrays or loading settings and return the image-discovery benchmark.

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

    def simulator(theta: torch.Tensor, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take images and mask centres and return smooth masked images."""
        return sim.image_mask(theta, x, generator=generator, half_width=half_width, smooth=smooth, noise=noise)

    return BED(
        "mnist-discovery-v0",
        prior,
        simulator,
        specs={
            "theta": Spec((channels, height, width)),
            "x": Spec((2,), low=0.0, high=1.0, labels=("row", "column")),
            "y": Spec((channels, height, width)),
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


@dataclass(frozen=True)
class ImageLabelPrior:
    """Sample an image and its class label together from a finite dataset."""

    images: torch.Tensor
    labels: torch.Tensor

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> Batch:
        """Take a leading shape and return matching image and label draws in a Batch."""
        count = int(np.prod(shape)) if shape else 1
        indices = torch.randint(len(self.images), (count,), generator=generator)
        return Batch(
            theta=self.images[indices].reshape(*shape, *self.images.shape[1:]),
            target=self.labels[indices].reshape(*shape),
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Take images and return their uniform empirical log mass."""
        return torch.full(value.shape[:-3], -np.log(len(self.images)), dtype=value.dtype, device=value.device)


def mnist_classification(
    *,
    images: np.ndarray | torch.Tensor | None = None,
    labels: np.ndarray | torch.Tensor | None = None,
    root: str | Path = "data",
    train: bool = True,
    download: bool = False,
    budget: int = 5,
    patch_size: int = 5,
    noise: float = 0.1,
) -> BED:
    """Take MNIST data and patch settings and return the Action-BED classification benchmark."""
    if images is None:
        try:
            from torchvision.datasets import MNIST
        except ImportError as error:
            raise ImportError("Install pybed[mnist] or pass images= and labels= arrays") from error
        dataset = MNIST(str(root), train=train, download=download)
        images = dataset.data[:, None].float() / 255
        labels = dataset.targets
    if labels is None:
        raise ValueError("mnist-classification-v0 requires class labels")
    images = torch.as_tensor(images, dtype=torch.get_default_dtype())
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4:
        raise ValueError("images must have shape (N,H,W) or (N,C,H,W)")
    labels = torch.as_tensor(labels, dtype=torch.long)
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    prior = ImageLabelPrior(images, labels)
    channels, height, width = images.shape[1:]

    def simulator(theta: torch.Tensor, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take images and patch corners and return differentiably sampled noisy patches."""
        return sim.image_patch(theta, x, generator=generator, patch_size=patch_size, noise=noise)

    def likelihood(y: torch.Tensor, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Take patches, images, and patch corners and return their Gaussian log likelihoods."""
        if noise <= 0:
            raise RuntimeError("MNIST patch log_prob requires noise > 0")
        mean = sim.image_patch(theta, x, patch_size=patch_size, noise=0)
        return torch.distributions.Normal(mean, noise).log_prob(y).flatten(-3).sum(-1)

    return BED(
        "mnist-classification-v0",
        prior,
        simulator,
        specs={
            "theta": Spec((channels, height, width)),
            "x": Spec((2,), low=0.0, high=1.0, labels=("patch row", "patch column")),
            "y": Spec((channels, patch_size, patch_size)),
        },
        budget=budget,
        likelihood=likelihood,
        visualizers={"episode": vs.image_classification, "prior": vs.prior},
        cfg=dict(image_shape=(channels, height, width), classes=10, patch_size=patch_size, noise=noise),
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
    """Take PDE and sensor settings and return an advection--diffusion inverse problem."""
    prior = prior if prior is not None else FieldPrior(shape)

    def simulator(theta: torch.Tensor, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take initial fields and sensor locations and return sensor time series."""
        return sim.advdiff(
            theta,
            x,
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
            "x": Spec((sensors, 2), low=0.0, high=1.0, labels=("sensor row", "sensor column")),
            "y": Spec((sensors, len(times)), labels=tuple(f"t={t:g}" for t in times)),
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
    """Take pendulum settings and return the stochastic system-identification benchmark."""
    prior = Uniform(torch.tensor([5.0, 0.02]), torch.tensor([15.0, 0.8]))

    def simulator(theta: torch.Tensor, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take physical parameters and initial states and return angle trajectories."""
        return sim.pendulum(
            theta,
            x,
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
            "x": Spec((2,), low=-np.pi, high=np.pi, labels=("initial angle", "angular velocity")),
            "y": Spec((steps // keep_every + 1,), labels=("angle time series",)),
        },
        budget=budget,
        visualizers={"episode": vs.timeseries, "prior": vs.prior},
        cfg=dict(dt=dt, steps=steps, keep_every=keep_every, process_noise=process_noise, obs_noise=obs_noise),
    )


def ces(*, goods: int = 3, budget: int = 20, noise: float = 0.05) -> BED:
    """Take CES settings and return the preference-learning benchmark."""
    mean = torch.cat((torch.tensor([0.5]), torch.full((goods,), 1 / goods), torch.tensor([1.0])))
    std = torch.cat((torch.tensor([0.2]), torch.full((goods,), 0.25), torch.tensor([0.5])))

    def simulator(theta: torch.Tensor, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take CES parameters and baskets and return noisy preferences."""
        return sim.ces(theta, x, generator=generator, noise=noise)

    return BED(
        "ces-v0",
        Normal(mean, std),
        simulator,
        specs={"theta": Spec((goods + 2,)), "x": Spec((2 * goods,), low=0, high=100), "y": Spec((1,))},
        budget=budget,
        visualizers={"episode": vs.timeseries, "prior": vs.prior},
        cfg=dict(goods=goods, noise=noise),
    )


def death(*, population: int = 50, budget: int = 4, max_time: float = 5.0) -> BED:
    """Take population settings and return the DAD pure-death benchmark."""
    prior = Uniform(torch.tensor([0.01]), torch.tensor([2.0]))

    def simulator(theta: torch.Tensor, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take rates and observation times and return affected population counts."""
        return sim.death(theta, x, generator=generator, population=population)

    return BED(
        "death-v0",
        prior,
        simulator,
        specs={"theta": Spec((1,), low=0), "x": Spec((1,), low=0, high=max_time), "y": Spec((1,))},
        budget=budget,
        visualizers={"episode": vs.timeseries, "prior": vs.prior},
        cfg=dict(population=population, max_time=max_time),
    )


def inverse(
    *,
    prior: Any,
    simulator: Any,
    theta_shape: tuple[int | None, ...],
    y_shape: tuple[int | None, ...],
    name: str = "inverse-v0",
    likelihood: Any = None,
) -> BED:
    """Take a prior, forward simulator, and shapes and return a no-design inverse problem."""
    return BED(
        name=name,
        prior=prior,
        sim=simulator,
        specs={"theta": Spec(theta_shape), "y": Spec(y_shape)},
        likelihood=likelihood,
        budget=1,
        vectorized=True,
    )


for env_name, factory in {
    "location-v0": location,
    "mnist-discovery-v0": mnist,
    "mnist-classification-v0": mnist_classification,
    "advdiff-v0": advdiff,
    "pendulum-v0": pendulum,
    "ces-v0": ces,
    "death-v0": death,
}.items():
    register(env_name, factory)
