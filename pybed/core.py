"""Core contracts shared by every PyBED environment and external method adapter."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

Array = np.ndarray | torch.Tensor


@runtime_checkable
class Prior(Protocol):
    """Minimal prior protocol.  Any user object with these methods is accepted."""

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> Array: ...
    def log_prob(self, value: Array) -> Array: ...


@dataclass(frozen=True)
class Spec:
    """A lightweight event-space declaration used for summaries and runtime checks.

    ``shape`` describes only event dimensions, so a ``Spec((2,))`` accepts both
    ``(2,)`` and ``(batch, 2)``.  ``None`` is a wildcard event dimension.  Bounds
    are descriptive by default and can be checked with ``bounds=True``.
    """

    shape: tuple[int | None, ...]
    dtype: str = "float"
    low: float | None = None
    high: float | None = None
    labels: tuple[str, ...] = ()

    def check(self, value: Any, name: str = "value", *, bounds: bool = False) -> Any:
        if value is None:
            raise ValueError(f"{name} is required by its Spec")
        actual = tuple(value.shape) if hasattr(value, "shape") else tuple(np.asarray(value).shape)
        if len(actual) < len(self.shape):
            raise ValueError(f"{name} has shape {actual}; expected trailing event shape {self.shape}")
        tail = actual[-len(self.shape) :] if self.shape else ()
        bad = [
            i for i, (got, want) in enumerate(zip(tail, self.shape, strict=True)) if want is not None and got != want
        ]
        if bad:
            raise ValueError(f"{name} has shape {actual}; expected trailing event shape {self.shape}")
        if bounds and (self.low is not None or self.high is not None):
            if torch.is_tensor(value):
                too_low = self.low is not None and bool(torch.any(value < self.low))
                too_high = self.high is not None and bool(torch.any(value > self.high))
            else:
                arr = np.asarray(value)
                too_low = self.low is not None and bool(np.any(arr < self.low))
                too_high = self.high is not None and bool(np.any(arr > self.high))
            if too_low or too_high:
                raise ValueError(f"{name} falls outside [{self.low}, {self.high}]")
        return value

    def __repr__(self) -> str:
        interval = "" if self.low is None and self.high is None else f", range=[{self.low}, {self.high}]"
        return f"Spec(shape={self.shape}, dtype={self.dtype}{interval})"


@dataclass
class Batch:
    """Portable BED episode batch.

    All fields may be NumPy arrays or tensors; ``theta`` is optional at deployment.
    ``mask`` has shape ``(..., T)``
    and supports padded histories such as ALINE's context/query/target attention
    patterns. ``target`` may hold a changing parameter mask, labels, images, or a
    predictive target without changing the environment simulator.
    """

    theta: Array | None = None
    design: Array | None = None
    obs: Array | None = None
    mask: Array | None = None
    target: Any = None
    context: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        for value in (self.theta, self.design, self.obs, self.mask):
            if getattr(value, "ndim", 0):
                return int(value.shape[0])
        raise ValueError("An empty Batch has no inferable batch size; pass batch_size in component context")

    def history(self, steps: int | None = None) -> Batch:
        if self.design is None or self.obs is None or steps is None:
            return self
        return replace(
            self,
            design=self.design[..., :steps, :],
            obs=self.obs[..., :steps, :],
            mask=None if self.mask is None else self.mask[..., :steps],
        )

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> Batch:
        values: dict[str, Any] = {}
        for name in ("theta", "design", "obs", "mask", "target"):
            value = getattr(self, name)
            if torch.is_tensor(value):
                values[name] = value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            elif isinstance(value, np.ndarray):
                tensor = torch.as_tensor(value)
                values[name] = tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else None)
            else:
                values[name] = value
        return Batch(**values, context=self.context, meta=self.meta)

    def numpy(self) -> Batch:
        values: dict[str, Any] = {}
        for name in ("theta", "design", "obs", "mask", "target"):
            value = getattr(self, name)
            values[name] = value.detach().cpu().numpy() if torch.is_tensor(value) else value
        return Batch(**values, context=self.context, meta=self.meta)


@dataclass(frozen=True)
class Uniform:
    low: Array | float
    high: Array | float

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> torch.Tensor:
        low = torch.as_tensor(self.low, dtype=torch.get_default_dtype())
        high = torch.as_tensor(self.high, dtype=torch.get_default_dtype())
        return low + (high - low) * torch.rand((*shape, *low.shape), generator=generator, device=low.device)

    def log_prob(self, value: Array) -> torch.Tensor:
        value = torch.as_tensor(value)
        low, high = torch.as_tensor(self.low, device=value.device), torch.as_tensor(self.high, device=value.device)
        inside = ((value >= low) & (value <= high)).all(dim=-1)
        log_density = -torch.log(high - low).sum()
        return torch.where(inside, log_density, torch.full_like(inside, -torch.inf, dtype=value.dtype))


@dataclass(frozen=True)
class Normal:
    mean: Array | float
    std: Array | float

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=torch.get_default_dtype())
        std = torch.as_tensor(self.std, dtype=torch.get_default_dtype())
        return mean + std * torch.randn((*shape, *mean.shape), generator=generator, device=mean.device)

    def log_prob(self, value: Array) -> torch.Tensor:
        value = torch.as_tensor(value)
        mean, std = torch.as_tensor(self.mean, device=value.device), torch.as_tensor(self.std, device=value.device)
        return (-0.5 * ((value - mean) / std).square() - torch.log(std) - 0.5 * np.log(2 * np.pi)).sum(-1)


@dataclass(frozen=True)
class EmpiricalPrior:
    """A point-cloud prior for PSST-style methods and simulation-based inference."""

    points: Array
    weights: Array | None = None
    jitter: float = 0.0

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> torch.Tensor:
        points = torch.as_tensor(self.points, dtype=torch.get_default_dtype())
        weights = None if self.weights is None else torch.as_tensor(self.weights, dtype=points.dtype)
        count = int(np.prod(shape)) if shape else 1
        indices = torch.multinomial(
            torch.ones(len(points), dtype=points.dtype) if weights is None else weights,
            count,
            replacement=True,
            generator=generator,
        )
        samples = points[indices].reshape(*shape, *points.shape[1:])
        if self.jitter:
            samples = samples + self.jitter * torch.randn(samples.shape, generator=generator, dtype=samples.dtype)
        return samples

    def log_prob(self, value: Array) -> torch.Tensor:
        if not self.jitter:
            raise NotImplementedError("EmpiricalPrior.log_prob requires jitter > 0")
        value, points = torch.as_tensor(value), torch.as_tensor(self.points)
        diff = (value.unsqueeze(-2) - points).flatten(-1).square().sum(-1) / self.jitter**2
        weights = (
            torch.ones(len(points), device=value.device) if self.weights is None else torch.as_tensor(self.weights)
        )
        log_components = -0.5 * diff + torch.log(weights / weights.sum())
        event_dim = points[0].numel()
        return torch.logsumexp(log_components, -1) - event_dim * np.log(self.jitter * np.sqrt(2 * np.pi))


@dataclass(frozen=True)
class BED:
    """Composable Bayesian experimental-design environment.

    Components are ordinary callables.  A method may be replaced with
    ``env.with_components(infer=my_model.infer)`` without subclassing or mutating
    the canonical benchmark.  The environment holds no episode RNG state: every
    stochastic call accepts a seed or generator, which keeps comparisons explicit.
    """

    name: str
    prior: Prior | Callable[..., Array]
    sim: Callable[..., Array]
    specs: Mapping[str, Spec]
    budget: int = 1
    policy: Callable[..., Array] | None = None
    posterior: Callable[..., Any] | None = None
    predictor: Callable[..., Any] | None = None
    visualizers: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    cfg: Mapping[str, Any] = field(default_factory=dict)
    target: Callable[..., Any] | None = None

    def with_components(self, **components: Any) -> BED:
        aliases = {"simulate": "sim", "design": "policy", "infer": "posterior", "predict": "predictor"}
        updates = {aliases.get(name, name): value for name, value in components.items()}
        unknown = set(updates) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"Unknown BED component(s): {sorted(unknown)}")
        return replace(self, **updates)

    def invoke(self, component: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call a component while passing only context that its signature accepts."""
        fn = component
        signature = inspect.signature(fn)
        accepts_context = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        call_kwargs = kwargs if accepts_context else {k: v for k, v in kwargs.items() if k in signature.parameters}
        try:
            return fn(*args, **call_kwargs)
        except TypeError as error:
            name = getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__))
            raise TypeError(f"PyBED could not call component {name}{signature}: {error}") from error

    def sample_prior(
        self, n: int, *, seed: int | None = None, generator: torch.Generator | None = None, **context: Any
    ) -> Array:
        if generator is None:
            generator = torch.Generator().manual_seed(0 if seed is None else seed)
        prior = (
            self.invoke(self.prior, **context)
            if callable(self.prior) and not hasattr(self.prior, "sample")
            else self.prior
        )
        theta = (
            prior.sample((n,), generator=generator)
            if hasattr(prior, "sample")
            else self.invoke(prior, n, generator=generator, **context)
        )
        return self.specs["theta"].check(theta, "theta") if "theta" in self.specs else theta

    def simulate(
        self,
        theta: Array,
        design: Array,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        **context: Any,
    ) -> Array:
        if "theta" in self.specs:
            self.specs["theta"].check(theta, "theta")
        if "design" in self.specs:
            self.specs["design"].check(design, "design")
        if generator is None:
            device = theta.device if torch.is_tensor(theta) else "cpu"
            generator = torch.Generator(device=device).manual_seed(0 if seed is None else seed)
        obs = self.invoke(self.sim, theta, design, generator=generator, **context)
        return self.specs["obs"].check(obs, "observation") if "obs" in self.specs else obs

    def design(self, history: Batch | None = None, **context: Any) -> Array:
        if self.policy is None:
            raise RuntimeError("No policy is attached; use env.with_components(design=policy)")
        result = self.invoke(self.policy, history, env=self, **context)
        return self.specs["design"].check(result, "design") if "design" in self.specs else result

    def infer(
        self,
        history: Batch | Array | None = None,
        obs: Array | None = None,
        *,
        mask: Array | None = None,
        target: Any = None,
        **context: Any,
    ) -> Any:
        if self.posterior is None:
            raise RuntimeError("No posterior is attached; use env.with_components(infer=posterior)")
        if not isinstance(history, Batch):
            history = Batch(theta=None, design=history, obs=obs, mask=mask, target=target)
        return self.invoke(self.posterior, history, env=self, **context)

    def predict(self, query: Any, history: Batch | None = None, **context: Any) -> Any:
        if self.predictor is None:
            raise RuntimeError("No predictor is attached; use env.with_components(predict=predictor)")
        return self.invoke(self.predictor, query, history, env=self, **context)

    def visualize(self, name: str = "episode", *args: Any, **kwargs: Any) -> Any:
        if name not in self.visualizers:
            raise KeyError(f"Visualizer {name!r} is not registered; choose from {sorted(self.visualizers)}")
        return self.invoke(self.visualizers[name], *args, env=self, **kwargs)

    def rollout(
        self,
        policy: Callable[..., Array] | None = None,
        *,
        episodes: int = 1,
        seed: int = 0,
        epoch: int = 0,
        budget: int | None = None,
    ) -> Any:
        """Convenience wrapper around a reproducible finite ``EpochStream`` rollout."""
        from .data import EpochStream

        return EpochStream(self, episodes, seed=seed, budget=budget).generate(policy, epoch=epoch)

    def dataloader(
        self,
        *,
        episodes: int = 1_024,
        batch_size: int = 128,
        seed: int = 0,
        epoch: int = 0,
        policy: Callable[..., Array] | None = None,
        dataset: Any = None,
        shuffle: bool = True,
        workers: int = 0,
    ) -> Any:
        """Build one deterministic online or offline loader from the environment."""
        from .data import EpisodeDataset, EpochStream, load, loader

        if isinstance(dataset, (str, bytes)) or hasattr(dataset, "__fspath__"):
            dataset = load(dataset)
        if dataset is not None and not isinstance(dataset, EpisodeDataset):
            raise TypeError("dataset must be an EpisodeDataset or a saved dataset path")
        data = dataset if dataset is not None else EpochStream(self, episodes, seed=seed).generate(policy, epoch=epoch)
        return loader(data, batch_size, shuffle=shuffle, seed=seed, epoch=epoch, workers=workers)

    def __repr__(self) -> str:
        rows = [f"BED({self.name!r}, budget={self.budget})"]
        rows.extend(f"  {name:>8}: {spec}" for name, spec in self.specs.items())
        attached = [
            name
            for name, value in (("design", self.policy), ("infer", self.posterior), ("predict", self.predictor))
            if value
        ]
        rows.append(f"  attached: {', '.join(attached) if attached else 'simulator only'}")
        return "\n".join(rows)


REGISTRY: dict[str, Callable[..., BED]] = {}


def register(name: str, factory: Callable[..., BED], *, overwrite: bool = False) -> None:
    """Register a BED factory under a stable versioned name."""
    if name in REGISTRY and not overwrite:
        raise KeyError(f"{name!r} is already registered")
    REGISTRY[name] = factory


def make(name: str, **cfg: Any) -> BED:
    """Construct a registered environment, importing built-ins on first use."""
    if not REGISTRY:
        from . import envs  # noqa: F401
    if name not in REGISTRY:
        raise KeyError(f"Unknown environment {name!r}; available: {available()}")
    return REGISTRY[name](**cfg)


def available() -> tuple[str, ...]:
    if not REGISTRY:
        from . import envs  # noqa: F401
    return tuple(sorted(REGISTRY))
