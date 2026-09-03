"""The small set of objects shared by PyBED environments and methods."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

Array = np.ndarray | torch.Tensor
Result = dict[str, Any]


def result(value: Any = None, *, primary: str | None = None) -> Result:
    """Take a model output and return design, infer, and predict entries."""
    output = {"design": None, "infer": None, "predict": None}
    if isinstance(value, Mapping):
        output.update({name: value.get(name) for name in output})
    elif primary is not None:
        output[primary] = value
    return output


def _map_tree(value: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply ``fn`` to every array in ``value`` and return the same nested shape."""
    if isinstance(value, ParticleCloud):
        return ParticleCloud(
            points=_map_tree(value.points, fn),
            weights=_map_tree(value.weights, fn),
            mask=_map_tree(value.mask, fn),
            particle_dim=value.particle_dim,
        )
    if isinstance(value, PolicySample):
        return PolicySample(
            x=_map_tree(value.x, fn),
            log_prob=_map_tree(value.log_prob, fn),
            entropy=_map_tree(value.entropy, fn),
        )
    if isinstance(value, Mapping):
        return type(value)((key, _map_tree(item, fn)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_map_tree(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_tree(item, fn) for item in value]
    return fn(value)


def _index_tree(value: Any, index: Any, size: int) -> Any:
    """Index arrays whose first axis is the batch axis and return everything else unchanged."""
    if isinstance(value, ParticleCloud):
        batched = value.particle_dim > 0 and value.points.shape[0] == size
        return ParticleCloud(
            points=_index_tree(value.points, index, size),
            weights=_index_tree(value.weights, index, size),
            mask=_index_tree(value.mask, index, size),
            particle_dim=value.particle_dim - 1 if batched else value.particle_dim,
        )

    def take(item: Any) -> Any:
        """Take one tree item and return its indexed value when it has a batch axis."""
        if isinstance(item, (np.ndarray, torch.Tensor)) and item.ndim and item.shape[0] == size:
            return item[index]
        return item

    return _map_tree(value, take)


@runtime_checkable
class Prior(Protocol):
    """A prior takes a sample shape and returns parameter draws."""

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> Array | Batch:
        """Take ``shape`` and a generator and return draws with those leading axes."""
        ...

    def log_prob(self, value: Array) -> Array:
        """Take parameter values and return their log densities."""
        ...


@dataclass(frozen=True)
class Spec:
    """Describe the trailing shape and optional bounds of one event."""

    shape: tuple[int | None, ...]
    dtype: str = "float"
    low: float | None = None
    high: float | None = None
    labels: tuple[str, ...] = ()

    def check(self, value: Any, name: str = "value", *, bounds: bool = False) -> Any:
        """Take an array, check its trailing shape and bounds, and return the same array."""
        if value is None:
            raise ValueError(f"{name} is required by its Spec")
        actual = tuple(value.shape) if hasattr(value, "shape") else tuple(np.asarray(value).shape)
        if len(actual) < len(self.shape):
            raise ValueError(f"{name} has shape {actual}; expected trailing event shape {self.shape}")
        tail = actual[-len(self.shape) :] if self.shape else ()
        if any(want is not None and got != want for got, want in zip(tail, self.shape, strict=True)):
            raise ValueError(f"{name} has shape {actual}; expected trailing event shape {self.shape}")
        if bounds and (self.low is not None or self.high is not None):
            if torch.is_tensor(value):
                too_low = self.low is not None and bool(torch.any(value < self.low))
                too_high = self.high is not None and bool(torch.any(value > self.high))
            else:
                array = np.asarray(value)
                too_low = self.low is not None and bool(np.any(array < self.low))
                too_high = self.high is not None and bool(np.any(array > self.high))
            if too_low or too_high:
                raise ValueError(f"{name} falls outside [{self.low}, {self.high}]")
        return value

    def __repr__(self) -> str:
        """Return a short readable description of the event shape and bounds."""
        interval = "" if self.low is None and self.high is None else f", range=[{self.low}, {self.high}]"
        return f"Spec(shape={self.shape}, dtype={self.dtype}{interval})"


@dataclass(frozen=True)
class Observation:
    """Keep one observation as a design and its measured outcome."""

    design: Any
    outcome: Any

    @property
    def x(self) -> Any:
        """Return the design using its short mathematical name."""
        return self.design

    @property
    def y(self) -> Any:
        """Return the outcome using its short mathematical name."""
        return self.outcome

    def __iter__(self):
        """Yield the design and then the outcome for pair unpacking."""
        yield self.design
        yield self.outcome


@dataclass(frozen=True)
class ParticleCloud:
    """A set of posterior or prior particles, with optional weights and masks."""

    points: Array
    weights: Array | None = None
    mask: Array | None = None
    particle_dim: int = 1

    def __post_init__(self) -> None:
        """Check the particle axis and weights, returning no value."""
        ndim = self.points.ndim
        dim = self.particle_dim if self.particle_dim >= 0 else ndim + self.particle_dim
        if dim < 0 or dim >= ndim:
            raise ValueError(f"particle_dim={self.particle_dim} is invalid for points with {ndim} axes")
        if self.weights is not None and self.weights.shape[-1] != self.points.shape[dim]:
            raise ValueError("The last weights axis must match the number of particles")

    def _parts(self) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, ...], tuple[int, ...]]:
        """Return Torch points, normalized weights, particle axis, batch shape, and event shape."""
        points = torch.as_tensor(self.points)
        dim = self.particle_dim if self.particle_dim >= 0 else points.ndim + self.particle_dim
        batch_shape, event_shape = tuple(points.shape[:dim]), tuple(points.shape[dim + 1 :])
        weight_dtype = points.dtype if points.is_floating_point() else torch.get_default_dtype()
        if self.weights is None:
            weights = torch.ones((*batch_shape, points.shape[dim]), dtype=weight_dtype, device=points.device)
        else:
            weights = torch.as_tensor(self.weights, dtype=weight_dtype, device=points.device)
            if tuple(weights.shape) != (*batch_shape, points.shape[dim]):
                raise ValueError("weights must have shape (*batch_shape, particles)")
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weight_dtype).tiny)
        return points, weights, dim, batch_shape, event_shape

    def mean(self) -> torch.Tensor:
        """Take the weighted particle cloud and return its mean over particles."""
        points, weights, dim, batch_shape, event_shape = self._parts()
        shaped_weights = weights.reshape(*batch_shape, weights.shape[-1], *(1 for _ in event_shape))
        mean = (points * shaped_weights).sum(dim)
        if self.mask is not None:
            mean = torch.where(torch.as_tensor(self.mask, device=mean.device, dtype=torch.bool), mean, 0)
        return mean

    def effective_sample_size(self) -> torch.Tensor:
        """Take normalized particle weights and return their effective sample size."""
        _, weights, _, _, _ = self._parts()
        return weights.square().sum(-1).reciprocal()

    def sample(self, count: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take a sample count and return weighted draws with a new particle axis."""
        points, weights, dim, batch_shape, event_shape = self._parts()
        particles = points.shape[dim]
        flat_batch = int(np.prod(batch_shape)) if batch_shape else 1
        flat_points = points.reshape(flat_batch, particles, *event_shape)
        flat_weights = weights.reshape(flat_batch, particles)
        indices = torch.multinomial(flat_weights, count, replacement=True, generator=generator)
        flat_event = int(np.prod(event_shape)) if event_shape else 1
        gathered = flat_points.reshape(flat_batch, particles, flat_event).gather(
            1, indices[..., None].expand(flat_batch, count, flat_event)
        )
        return gathered.reshape(*batch_shape, count, *event_shape)

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> ParticleCloud:
        """Take a device and optional dtype and return a moved cloud."""
        def move(value: Any) -> Any:
            """Take one cloud field and return it on the requested device."""
            if value is None:
                return None
            tensor = torch.as_tensor(value)
            return tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else None)

        return ParticleCloud(move(self.points), move(self.weights), move(self.mask), self.particle_dim)

    def numpy(self) -> ParticleCloud:
        """Return a cloud whose tensor fields are NumPy arrays on the CPU."""
        return _map_tree(self, lambda value: value.detach().cpu().numpy() if torch.is_tensor(value) else value)

    def detach(self) -> ParticleCloud:
        """Return a cloud detached from autograd and stored on the CPU."""
        return _map_tree(self, lambda value: value.detach().cpu() if torch.is_tensor(value) else value)


@dataclass(frozen=True)
class PolicySample:
    """A sampled design and the optional values needed by a score-function loss."""

    x: Array
    log_prob: Array | None = None
    entropy: Array | None = None

    def to(self, device: str | torch.device) -> PolicySample:
        """Take a device and return the policy sample on that device."""
        return _map_tree(self, lambda value: value.to(device) if torch.is_tensor(value) else value)

    def detach(self) -> PolicySample:
        """Return the policy sample detached from autograd and stored on the CPU."""
        return _map_tree(self, lambda value: value.detach().cpu() if torch.is_tensor(value) else value)


@dataclass(init=False)
class Batch:
    """A batch of truths, observation pairs, masks, targets, and optional particles."""

    theta: Array | None
    x: Array | None
    y: Array | None
    mask: Array | None
    target: Any
    belief: ParticleCloud | None
    context: dict[str, Any]
    meta: dict[str, Any]

    def __init__(
        self,
        parameters: Array | None = None,
        designs: Array | None = None,
        outcomes: Array | None = None,
        mask: Array | None = None,
        target: Any = None,
        belief: ParticleCloud | None = None,
        context: Mapping[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
        *,
        theta: Array | None = None,
        x: Array | None = None,
        y: Array | None = None,
        design: Array | None = None,
        outcome: Array | None = None,
        o: Observation | tuple[Any, Any] | None = None,
        obs: Observation | tuple[Any, Any] | None = None,
    ) -> None:
        """Take parameters and observation arrays and return a new batch."""
        if theta is not None:
            if parameters is not None:
                raise TypeError("Pass parameters or theta, not both")
            parameters = theta
        if designs is not None and (x is not None or design is not None):
            raise TypeError("Pass designs or x/design, not both")
        if outcomes is not None and (y is not None or outcome is not None):
            raise TypeError("Pass outcomes or y/outcome, not both")
        if x is not None and design is not None:
            raise TypeError("Pass x or design, not both")
        if y is not None and outcome is not None:
            raise TypeError("Pass y or outcome, not both")
        designs = designs if designs is not None else x if x is not None else design
        outcomes = outcomes if outcomes is not None else y if y is not None else outcome
        pair = o if o is not None else obs
        if o is not None and obs is not None:
            raise TypeError("Pass o or obs, not both")
        if pair is not None:
            pair_design, pair_outcome = (pair.design, pair.outcome) if isinstance(pair, Observation) else pair
            if designs is not None or outcomes is not None:
                raise TypeError("Pass an observation pair or separate designs/outcomes, not both")
            designs, outcomes = pair_design, pair_outcome
        self.theta, self.x, self.y = parameters, designs, outcomes
        self.mask, self.target, self.belief = mask, target, belief
        self.context, self.meta = dict(context or {}), dict(meta or {})

    @property
    def parameters(self) -> Array | None:
        """Return the parameter values stored in the batch."""
        return self.theta

    @parameters.setter
    def parameters(self, value: Array | None) -> None:
        """Take parameter values and store them in the batch."""
        self.theta = value

    @property
    def thetas(self) -> Array | None:
        """Return the parameter values using their mathematical plural."""
        return self.theta

    @thetas.setter
    def thetas(self, value: Array | None) -> None:
        """Take parameter values and store them as theta values."""
        self.theta = value

    @property
    def designs(self) -> Array | None:
        """Return all designs stored in the batch."""
        return self.x

    @designs.setter
    def designs(self, value: Array | None) -> None:
        """Take design values and store them in the batch."""
        self.x = value

    @property
    def design(self) -> Array | None:
        """Return the design or designs stored in the batch."""
        return self.x

    @design.setter
    def design(self, value: Array | None) -> None:
        """Take design values and store them in the batch."""
        self.x = value

    @property
    def outcomes(self) -> Array | None:
        """Return all measured outcomes stored in the batch."""
        return self.y

    @outcomes.setter
    def outcomes(self, value: Array | None) -> None:
        """Take measured outcomes and store them in the batch."""
        self.y = value

    @property
    def outcome(self) -> Array | None:
        """Return the outcome or outcomes stored in the batch."""
        return self.y

    @outcome.setter
    def outcome(self, value: Array | None) -> None:
        """Take measured outcomes and store them in the batch."""
        self.y = value

    @property
    def o(self) -> Observation:
        """Return the complete observations using their short mathematical name."""
        return Observation(self.x, self.y)

    @property
    def obs(self) -> Observation:
        """Return the complete design-and-outcome observations."""
        return self.o

    @property
    def observations(self) -> Observation:
        """Return all designs and outcomes as one observation pair."""
        return self.o

    def __len__(self) -> int:
        """Return the leading batch size from the first available array."""
        for value in (self.theta, self.x, self.y, self.mask, self.target):
            if getattr(value, "ndim", 0):
                return int(value.shape[0])
        if self.belief is not None and self.belief.particle_dim != 0:
            return int(self.belief.points.shape[0])
        raise ValueError("This Batch has no array from which to read its batch size")

    def history(self, steps: int | None = None) -> Batch:
        """Take a step count and return the corresponding observation prefix."""
        if steps is None:
            return self

        def prefix(value: Any) -> Any:
            """Take one history array and return its first requested steps."""
            return None if value is None else value[:, :steps, ...]

        return replace(
            self,
            x=prefix(self.x),
            y=prefix(self.y),
            mask=None if self.mask is None else self.mask[:, :steps],
        )

    def index(self, index: Any) -> Batch:
        """Take a batch index and return the selected record or smaller batch."""
        size = len(self)
        return Batch(
            parameters=_index_tree(self.parameters, index, size),
            designs=_index_tree(self.designs, index, size),
            outcomes=_index_tree(self.outcomes, index, size),
            mask=_index_tree(self.mask, index, size),
            target=_index_tree(self.target, index, size),
            belief=_index_tree(self.belief, index, size),
            context=_index_tree(self.context, index, size),
            meta=self.meta,
        )

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> Batch:
        """Take a device and optional dtype and return a batch on that device."""
        def move(value: Any) -> Any:
            """Take one batch field and return it on the requested device."""
            if torch.is_tensor(value):
                return value.to(device=device, dtype=dtype if value.is_floating_point() else None)
            if isinstance(value, np.ndarray):
                tensor = torch.as_tensor(value)
                return tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else None)
            return value

        return Batch(
            parameters=_map_tree(self.parameters, move),
            designs=_map_tree(self.designs, move),
            outcomes=_map_tree(self.outcomes, move),
            mask=_map_tree(self.mask, move),
            target=_map_tree(self.target, move),
            belief=_map_tree(self.belief, move),
            context=_map_tree(self.context, move),
            meta=self.meta,
        )

    def numpy(self) -> Batch:
        """Return a batch whose tensor fields are NumPy arrays on the CPU."""
        convert = lambda value: value.detach().cpu().numpy() if torch.is_tensor(value) else value
        return Batch(
            parameters=_map_tree(self.parameters, convert),
            designs=_map_tree(self.designs, convert),
            outcomes=_map_tree(self.outcomes, convert),
            mask=_map_tree(self.mask, convert),
            target=_map_tree(self.target, convert),
            belief=_map_tree(self.belief, convert),
            context=_map_tree(self.context, convert),
            meta=self.meta,
        )

    def detach(self) -> Batch:
        """Return a batch detached from autograd and stored on the CPU."""
        detach = lambda value: value.detach().cpu() if torch.is_tensor(value) else value
        return Batch(
            parameters=_map_tree(self.parameters, detach),
            designs=_map_tree(self.designs, detach),
            outcomes=_map_tree(self.outcomes, detach),
            mask=_map_tree(self.mask, detach),
            target=_map_tree(self.target, detach),
            belief=_map_tree(self.belief, detach),
            context=_map_tree(self.context, detach),
            meta=self.meta,
        )


@dataclass(frozen=True)
class Uniform:
    """A uniform prior between matching lower and upper bounds."""

    low: Array | float
    high: Array | float

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take a leading shape and return uniform draws with the prior event shape."""
        low = torch.as_tensor(self.low, dtype=torch.get_default_dtype())
        high = torch.as_tensor(self.high, dtype=torch.get_default_dtype())
        return low + (high - low) * torch.rand((*shape, *low.shape), generator=generator, device=low.device)

    def log_prob(self, value: Array) -> torch.Tensor:
        """Take values and return their uniform log density, or minus infinity outside the bounds."""
        value = torch.as_tensor(value)
        low, high = torch.as_tensor(self.low, device=value.device), torch.as_tensor(self.high, device=value.device)
        event_dims = tuple(range(value.ndim - low.ndim, value.ndim))
        inside = ((value >= low) & (value <= high)).all(dim=event_dims) if event_dims else (value >= low) & (value <= high)
        log_density = -torch.log(high - low).sum()
        return torch.where(inside, log_density, torch.full_like(inside, -torch.inf, dtype=value.dtype))


@dataclass(frozen=True)
class Normal:
    """An independent normal prior with matching means and standard deviations."""

    mean: Array | float
    std: Array | float

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take a leading shape and return normal draws with the prior event shape."""
        mean = torch.as_tensor(self.mean, dtype=torch.get_default_dtype())
        std = torch.as_tensor(self.std, dtype=torch.get_default_dtype())
        return mean + std * torch.randn((*shape, *mean.shape), generator=generator, device=mean.device)

    def log_prob(self, value: Array) -> torch.Tensor:
        """Take values and return their independent-normal log density."""
        value = torch.as_tensor(value)
        mean, std = torch.as_tensor(self.mean, device=value.device), torch.as_tensor(self.std, device=value.device)
        terms = -0.5 * ((value - mean) / std).square() - torch.log(std) - 0.5 * np.log(2 * np.pi)
        event_dims = tuple(range(value.ndim - mean.ndim, value.ndim))
        return terms.sum(event_dims) if event_dims else terms


@dataclass(frozen=True)
class EmpiricalPrior:
    """A prior that resamples points from a finite cloud."""

    points: Array | ParticleCloud
    weights: Array | None = None
    jitter: float = 0.0

    def _cloud(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the stored points and weights as tensors."""
        if isinstance(self.points, ParticleCloud):
            if self.points.particle_dim != 0:
                raise ValueError("An EmpiricalPrior needs one unbatched cloud with particle_dim=0")
            points = torch.as_tensor(self.points.points, dtype=torch.get_default_dtype())
            weights = self.points.weights if self.weights is None else self.weights
        else:
            points = torch.as_tensor(self.points, dtype=torch.get_default_dtype())
            weights = self.weights
        return points, None if weights is None else torch.as_tensor(weights, dtype=points.dtype)

    def sample(self, shape: tuple[int, ...], *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Take a leading shape and return resampled cloud points, with optional jitter."""
        points, weights = self._cloud()
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
        """Take values and return their Gaussian-kernel mixture log density."""
        if not self.jitter:
            raise NotImplementedError("EmpiricalPrior.log_prob requires jitter > 0")
        value = torch.as_tensor(value)
        points, weights = self._cloud()
        points = points.to(value)
        event_ndim = points.ndim - 1
        difference = value.unsqueeze(-(event_ndim + 1)) - points
        diff = (
            difference.square()
            if event_ndim == 0
            else difference.flatten(-event_ndim).square().sum(-1)
        ) / self.jitter**2
        raw_weights = torch.ones(len(points), device=value.device) if weights is None else weights.to(value)
        log_components = -0.5 * diff + torch.log(raw_weights / raw_weights.sum())
        event_dim = points[0].numel()
        return torch.logsumexp(log_components, -1) - event_dim * np.log(self.jitter * np.sqrt(2 * np.pi))


@dataclass(frozen=True)
class BED:
    """Combine a prior, simulator, spaces, and optional learned components."""

    name: str
    prior: Prior | Callable[..., Array | Batch]
    sim: Callable[..., Array]
    specs: Mapping[str, Spec]
    budget: int = 1
    policy: Callable[..., Array | PolicySample] | None = None
    posterior: Callable[..., Any] | None = None
    predictor: Callable[..., Any] | None = None
    likelihood: Callable[..., Array] | None = None
    candidate_pool: Array | Callable[..., Array] | None = None
    vectorized: bool = True
    visualizers: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    cfg: Mapping[str, Any] = field(default_factory=dict)
    target: Callable[..., Any] | None = None

    def _spec(self, name: str, old_name: str) -> Spec | None:
        """Take a current and old space name and return whichever specification exists."""
        return self.specs.get(name, self.specs.get(old_name))

    def with_components(self, **components: Any) -> BED:
        """Take replacement callables and return a new environment using them."""
        aliases = {
            "simulate": "sim",
            "design": "policy",
            "infer": "posterior",
            "predict": "predictor",
            "log_prob": "likelihood",
            "candidates": "candidate_pool",
        }
        updates = {aliases.get(name, name): value for name, value in components.items()}
        unknown = set(updates) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"Unknown BED component(s): {sorted(unknown)}")
        return replace(self, **updates)

    def invoke(self, component: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Take a component and context, call it, and return its output."""
        signature = inspect.signature(component)
        accepts_context = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        call_kwargs = kwargs if accepts_context else {key: value for key, value in kwargs.items() if key in signature.parameters}
        try:
            return component(*args, **call_kwargs)
        except TypeError as error:
            name = getattr(component, "__qualname__", getattr(component, "__name__", type(component).__name__))
            raise TypeError(f"PyBED could not call component {name}{signature}: {error}") from error

    def sample_prior(
        self, n: int, *, seed: int | None = None, generator: torch.Generator | None = None, **context: Any
    ) -> Array | Batch:
        """Take a sample count and return prior draws, possibly with per-draw context in a Batch."""
        if generator is None:
            generator = torch.Generator().manual_seed(0 if seed is None else seed)
        prior = self.invoke(self.prior, **context) if callable(self.prior) and not hasattr(self.prior, "sample") else self.prior
        draw = (
            prior.sample((n,), generator=generator)
            if hasattr(prior, "sample")
            else self.invoke(prior, n, generator=generator, **context)
        )
        theta = draw.parameters if isinstance(draw, Batch) else draw
        spec = self._spec("theta", "theta")
        if spec is not None:
            spec.check(theta, "theta")
        return draw

    def simulate(
        self,
        theta: Array,
        x: Array | None = None,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        **context: Any,
    ) -> Array:
        """Take parameters and an optional design and return a simulated outcome ``y``."""
        theta_spec, x_spec = self._spec("theta", "theta"), self._spec("x", "design")
        if theta_spec is not None:
            theta_spec.check(theta, "theta")
        if x_spec is not None:
            x_spec.check(x, "x")
        if generator is None:
            device = theta.device if torch.is_tensor(theta) else "cpu"
            generator = torch.Generator(device=device).manual_seed(0 if seed is None else seed)
        y = self.invoke(self.sim, theta, generator=generator, **context) if x is None else self.invoke(
            self.sim, theta, x, generator=generator, **context
        )
        y_spec = self._spec("y", "obs")
        return y_spec.check(y, "y") if y_spec is not None else y

    def candidates(self, history: Batch | None = None, **context: Any) -> Array | None:
        """Take a history and return its available design candidates, or ``None`` for a continuous space."""
        if self.candidate_pool is None:
            return None
        if callable(self.candidate_pool):
            return self.invoke(self.candidate_pool, history, env=self, **context)
        return self.candidate_pool

    def design(
        self,
        history: Batch | None = None,
        *,
        infer: bool = False,
        predict: bool = False,
        query: Any = None,
        **context: Any,
    ) -> Result:
        """Take a history and output flags and return a model-result dictionary."""
        if self.policy is None:
            raise RuntimeError("No policy is attached; use env.with_components(design=policy)")
        if "candidates" not in context:
            context["candidates"] = self.candidates(history, **context)
        output = result(
            self.invoke(
                self.policy,
                history,
                infer=infer,
                predict=predict,
                query=query,
                env=self,
                **context,
            ),
            primary="design",
        )
        chosen = output["design"]
        x = chosen.x if isinstance(chosen, PolicySample) else chosen
        spec = self._spec("x", "design")
        if spec is not None and x is not None:
            spec.check(x, "x")
        if infer and output["infer"] is None and self.posterior is not None:
            output["infer"] = self.infer(history)["infer"]
        if predict and output["predict"] is None and self.predictor is not None:
            output["predict"] = self.predict(x if query is None else query, history)["predict"]
        return output

    def infer(
        self,
        history: Batch | Observation | Array | None = None,
        y: Array | None = None,
        *,
        mask: Array | None = None,
        target: Any = None,
        **context: Any,
    ) -> Result:
        """Take a history or observation arrays and return an inference result dictionary."""
        if self.posterior is None:
            raise RuntimeError("No posterior is attached; use env.with_components(infer=posterior)")
        if isinstance(history, Observation):
            history = Batch(o=history, mask=mask, target=target)
        elif not isinstance(history, Batch):
            history = Batch(theta=None, x=history, y=y, mask=mask, target=target)
        return result(self.invoke(self.posterior, history, env=self, **context), primary="infer")

    def predict(
        self,
        query: Any = None,
        history: Batch | None = None,
        *,
        infer: bool = False,
        **context: Any,
    ) -> Result:
        """Take a query, history, and inference flag and return a model-result dictionary."""
        if self.predictor is None:
            raise RuntimeError("No predictor is attached; use env.with_components(predict=predictor)")
        output = result(
            self.invoke(self.predictor, query, history, infer=infer, env=self, **context),
            primary="predict",
        )
        if infer and output["infer"] is None and self.posterior is not None:
            output["infer"] = self.infer(history)["infer"]
        return output

    def log_prob(self, y: Array, theta: Array, x: Array | None = None, **context: Any) -> Array:
        """Take an outcome, parameters, and optional design and return the observation log density."""
        if self.likelihood is None:
            raise RuntimeError("No likelihood is attached; use env.with_components(log_prob=likelihood)")
        if x is None:
            return self.invoke(self.likelihood, y, theta, env=self, **context)
        return self.invoke(self.likelihood, y, theta, x, env=self, **context)

    def visualize(self, name: str = "episode", *args: Any, **kwargs: Any) -> Any:
        """Take a visualizer name and its data and return the resulting plot objects."""
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
        designs: Array | None = None,
    ) -> Any:
        """Take rollout settings and return a finite generated dataset."""
        from .data import Stream

        return Stream(self, episodes, seed=seed, budget=budget).generate(policy, epoch=epoch, designs=designs)

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
        """Take online or saved-data settings and return one deterministic data loader."""
        from .data import EpisodeDataset, Stream, load, loader

        if isinstance(dataset, (str, bytes)) or hasattr(dataset, "__fspath__"):
            dataset = load(dataset)
        if dataset is not None and not isinstance(dataset, EpisodeDataset):
            raise TypeError("dataset must be an EpisodeDataset or a saved dataset path")
        data = dataset if dataset is not None else Stream(self, episodes, seed=seed).generate(policy, epoch=epoch)
        return loader(data, batch_size, shuffle=shuffle, seed=seed, epoch=epoch, workers=workers)

    def __repr__(self) -> str:
        """Return a readable summary of spaces and attached learned components."""
        rows = [f"BED({self.name!r}, budget={self.budget})"]
        rows.extend(f"  {name:>8}: {spec}" for name, spec in self.specs.items())
        attached = [
            name
            for name, value in (
                ("design", self.policy),
                ("infer", self.posterior),
                ("predict", self.predictor),
            )
            if value
        ]
        rows.append(f"  attached: {', '.join(attached) if attached else 'simulator only'}")
        return "\n".join(rows)


_ENVIRONMENTS: dict[str, Callable[..., BED]] = {}


def _add_environment(name: str, factory: Callable[..., BED]) -> None:
    """Take a name and factory and add one environment to PyBED's built-in list."""
    if name in _ENVIRONMENTS:
        raise KeyError(f"{name!r} is already available")
    _ENVIRONMENTS[name] = factory


def make(name: str, **cfg: Any) -> BED:
    """Take an environment name and settings and return the requested environment."""
    if not _ENVIRONMENTS:
        from . import envs  # noqa: F401
    if name not in _ENVIRONMENTS:
        raise KeyError(f"Unknown environment {name!r}; available: {environments()}")
    return _ENVIRONMENTS[name](**cfg)


def environments() -> tuple[str, ...]:
    """Return the names of PyBED's built-in environments."""
    if not _ENVIRONMENTS:
        from . import envs  # noqa: F401
    return tuple(sorted(_ENVIRONMENTS))


Particles = ParticleCloud
