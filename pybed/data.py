"""Generate, save, reload, and reuse BED observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .core import BED, Batch, ParticleCloud, PolicySample, result


class Seeds:
    """Create repeatable random generators from one seed and a short label."""

    def __init__(self, seed: int = 0):
        """Take one integer seed and store it for later generator requests."""
        self.seed = int(seed)

    def value(self, label: str, *indices: int) -> int:
        """Take a label and integer indices and return their repeatable integer seed."""
        payload = ":".join((str(self.seed), label, *(str(index) for index in indices))).encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)

    def torch(self, label: str, *indices: int, device: str | torch.device = "cpu") -> torch.Generator:
        """Take a label, indices, and device and return a seeded Torch generator."""
        return torch.Generator(device=device).manual_seed(self.value(label, *indices))

    def numpy(self, label: str, *indices: int) -> np.random.Generator:
        """Take a label and indices and return a seeded NumPy generator."""
        return np.random.default_rng(self.value(label, *indices))


def _stack(values: list[Any]) -> Any:
    """Take matching record values and return their batched form."""
    first = values[0]
    if first is None:
        return None
    if isinstance(first, ParticleCloud):
        points = _stack([value.points for value in values])
        weights = None if first.weights is None else _stack([value.weights for value in values])
        mask = None if first.mask is None else _stack([value.mask for value in values])
        return ParticleCloud(points, weights, mask, first.particle_dim + 1)
    if torch.is_tensor(first):
        return torch.stack(values)
    if isinstance(first, np.ndarray):
        return torch.as_tensor(np.stack(values))
    if isinstance(first, Mapping):
        return type(first)((key, _stack([value[key] for value in values])) for key in first)
    if isinstance(first, tuple):
        return tuple(_stack([value[index] for value in values]) for index in range(len(first)))
    if all(value == first for value in values):
        return first
    return values


class EpisodeDataset(Dataset[Batch]):
    """Store a finite batch of paired parameters and outcomes."""

    def __init__(self, batch: Batch):
        """Take a batch with parameters and outcomes and expose its rows as a dataset."""
        if batch.parameters is None or batch.outcomes is None:
            raise ValueError("An episode dataset requires parameters and outcomes; designs are optional")
        self.batch = batch

    def __len__(self) -> int:
        """Return the number of stored records."""
        return len(self.batch)

    def __getitem__(self, index: int) -> Batch:
        """Take a row index and return that Batch record."""
        return self.batch.index(index)

    def save(self, path: str | Path) -> Path:
        """Take a path, save the dataset there, and return the saved path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameters": self.batch.parameters,
            "designs": self.batch.designs,
            "outcomes": self.batch.outcomes,
            "mask": self.batch.mask,
            "target": self.batch.target,
            "belief": self.batch.belief,
            "context": self.batch.context,
            "meta": self.batch.meta,
            "format": "pybed-episodes",
        }
        torch.save(payload, path)
        manifest = {
            "format": "pybed-episodes",
            "episodes": len(self),
            "shapes": {
                name: list(value.shape)
                for name, value in payload.items()
                if isinstance(value, (np.ndarray, torch.Tensor))
            },
            "meta": self.batch.meta,
        }
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> EpisodeDataset:
        """Take a saved dataset path and return the restored EpisodeDataset."""
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("format") != "pybed-episodes":
            raise ValueError("Not a supported PyBED episode file")
        return cls(
            Batch(
                parameters=payload["parameters"],
                designs=payload.get("designs"),
                outcomes=payload["outcomes"],
                mask=payload.get("mask"),
                target=payload.get("target"),
                belief=payload.get("belief"),
                context=payload.get("context", {}),
                meta=payload.get("meta", {}),
            )
        )


def collate(items: list[Batch]) -> Batch:
    """Take individual Batch records and return one stacked Batch."""
    return Batch(
        parameters=_stack([item.parameters for item in items]),
        designs=_stack([item.designs for item in items]),
        outcomes=_stack([item.outcomes for item in items]),
        mask=_stack([item.mask for item in items]),
        target=_stack([item.target for item in items]),
        belief=_stack([item.belief for item in items]),
        context=_stack([item.context for item in items]),
        meta=items[0].meta,
    )


def loader(
    dataset: Dataset[Batch],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    epoch: int = 0,
    workers: int = 0,
    drop_last: bool = False,
) -> DataLoader[Batch]:
    """Take a dataset and loading settings and return a repeatable Torch DataLoader."""
    seeds = Seeds(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=seeds.torch("loader", epoch),
        num_workers=workers,
        drop_last=drop_last,
        collate_fn=collate,
    )


def _pool_sample(
    pool: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
    event_ndim: int,
) -> PolicySample:
    """Take a candidate pool and return one uniformly sampled candidate per batch item."""
    if pool.ndim < event_ndim + 1:
        raise ValueError("A candidate pool needs one candidate axis before its design shape")
    if pool.ndim == event_ndim + 2 and pool.shape[0] == batch_size:
        count = pool.shape[1]
        index = torch.randint(count, (batch_size,), generator=generator, device=pool.device)
        x = pool[torch.arange(batch_size, device=pool.device), index]
    else:
        count = pool.shape[0]
        index = torch.randint(count, (batch_size,), generator=generator, device=pool.device)
        x = pool[index]
    log_prob = torch.full((batch_size,), -float(np.log(count)), dtype=x.dtype, device=x.device)
    return PolicySample(x=x, log_prob=log_prob, entropy=-log_prob)


def random_policy(
    history: Batch | None,
    *,
    env: BED,
    generator: torch.Generator | None = None,
    batch_size: int | None = None,
    candidates: Any = None,
    **context: Any,
) -> torch.Tensor | PolicySample:
    """Take a history and environment and return random bounded or candidate-set designs."""
    if batch_size is None:
        batch_size = len(history) if history is not None else int(context.get("n", 1))
    if candidates is None:
        candidates = env.candidates(history, **context)
    if candidates is not None:
        generator = generator or torch.Generator()
        spec = env._spec("x", "design")
        return _pool_sample(torch.as_tensor(candidates), batch_size, generator, len(spec.shape) if spec else 1)
    spec = env._spec("x", "design")
    if spec is None:
        raise ValueError("This environment has no design space")
    if spec.low is None or spec.high is None:
        raise ValueError("Random designs need finite low/high bounds in env.specs['x']")
    if any(size is None for size in spec.shape):
        raise ValueError("Random designs need a fixed event shape")
    return spec.low + (spec.high - spec.low) * torch.rand(
        (batch_size, *spec.shape), generator=generator
    )


def _expand_time(value: Any, episodes: int, budget: int) -> Any:
    """Take per-episode context and return context with a matching time axis."""
    if isinstance(value, Mapping):
        return type(value)((key, _expand_time(item, episodes, budget)) for key, item in value.items())
    if isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim and value.shape[0] == episodes:
        tensor = torch.as_tensor(value)
        return tensor[:, None, ...].expand(episodes, budget, *tensor.shape[1:])
    return value


def _fixed_designs(designs: Any, episodes: int, budget: int, event_shape: tuple[int | None, ...]) -> torch.Tensor:
    """Take shared or per-episode designs and return one batched design sequence."""
    value = torch.as_tensor(designs)
    event_ndim = len(event_shape)
    if value.ndim == event_ndim + 1:
        value = value[None, ...].expand(episodes, *value.shape)
    if value.ndim != event_ndim + 2 or value.shape[0] != episodes:
        raise ValueError("designs must have shape (steps, *design_shape) or (episodes, steps, *design_shape)")
    if value.shape[1] != budget:
        raise ValueError(f"x contains {value.shape[1]} steps but this Stream has budget={budget}")
    return value


class Stream:
    """Generate comparable finite batches of simulated observations."""

    def __init__(self, env: BED, episodes: int, *, seed: int = 0, budget: int | None = None):
        """Take an environment and rollout size and prepare repeatable episode generation."""
        self.env, self.episodes, self.seed = env, int(episodes), int(seed)
        self.budget = env.budget if budget is None else int(budget)
        self.seeds = Seeds(seed)
        self._draw_cache: dict[int, Batch] = {}

    def draw(self, epoch: int = 0) -> Batch:
        """Take an epoch number and return its prior draws with any per-draw context."""
        if epoch not in self._draw_cache:
            generator = self.seeds.torch(f"{self.env.name}:theta", epoch)
            draw = self.env.sample_prior(self.episodes, generator=generator, epoch=epoch)
            batch = draw if isinstance(draw, Batch) else Batch(parameters=torch.as_tensor(draw))
            self._draw_cache[epoch] = batch.detach()
        cached = self._draw_cache[epoch]
        return Batch(
            parameters=cached.parameters.clone(),
            context=cached.context,
            target=cached.target,
            belief=cached.belief,
        )

    def parameters(self, epoch: int = 0) -> torch.Tensor:
        """Take an epoch number and return a copy of its latent parameters."""
        return torch.as_tensor(self.draw(epoch).parameters).clone()

    def _random_sequence(self, epoch: int) -> torch.Tensor:
        """Take an epoch number and return all random non-adaptive designs at once."""
        spec = self.env._spec("x", "design")
        if spec is None:
            raise ValueError("This environment has no design space")
        pool = self.env.candidates(None, epoch=epoch)
        if pool is not None and not callable(self.env.candidate_pool):
            pool = torch.as_tensor(pool)
            batched = pool.ndim == len(spec.shape) + 2 and pool.shape[0] == self.episodes
            count = pool.shape[1] if batched else pool.shape[0]
            indices = torch.randint(count, (self.episodes, self.budget), generator=self.seeds.torch("design", epoch))
            if batched:
                rows = torch.arange(self.episodes)[:, None].expand_as(indices)
                return pool[rows, indices]
            return pool[indices]
        if spec.low is None or spec.high is None or any(size is None for size in spec.shape):
            raise ValueError("Random designs need a fixed bounded x space")
        return spec.low + (spec.high - spec.low) * torch.rand(
            (self.episodes, self.budget, *spec.shape), generator=self.seeds.torch("design", epoch)
        )

    def _vectorized(self, draw: Batch, x: torch.Tensor | None, epoch: int, common_noise: bool) -> Batch:
        """Take prior draws and fixed designs and return one vectorized simulator batch."""
        theta = torch.as_tensor(draw.parameters)
        context = draw.context
        if x is None:
            y = self.env.simulate(
                theta,
                generator=self.seeds.torch("sim-common" if common_noise else "sim-random", epoch),
                epoch=epoch,
                **context,
            )
            mask = None
        else:
            theta_time = theta[:, None, ...].expand(self.episodes, self.budget, *theta.shape[1:])
            sim_context = _expand_time(context, self.episodes, self.budget)
            y = self.env.simulate(
                theta_time,
                x,
                generator=self.seeds.torch("sim-common" if common_noise else "sim-random", epoch),
                epoch=epoch,
                **sim_context,
            )
            mask = torch.ones((self.episodes, self.budget), dtype=torch.bool)
        target = (
            self.env.invoke(self.env.target, theta, **context)
            if self.env.target is not None
            else draw.target if draw.target is not None else theta
        )
        return Batch(
            parameters=theta,
            designs=x,
            outcomes=torch.as_tensor(y),
            mask=mask,
            target=target,
            belief=draw.belief,
            context=context,
        )

    def generate(
        self,
        policy: Callable[..., Any] | None = None,
        *,
        epoch: int = 0,
        common_noise: bool = True,
        save: str | Path | None = None,
        designs: Any = None,
        infer: bool = False,
        predict: bool = False,
    ) -> EpisodeDataset:
        """Take rollout choices and return a generated episode dataset."""
        draw = self.draw(epoch)
        x_spec = self.env._spec("x", "design")
        can_vectorize = (
            policy is None
            and self.env.vectorized
            and self.env.policy is None
            and not callable(self.env.candidate_pool)
        )
        x = designs
        if designs is not None:
            if x_spec is None:
                raise ValueError("Fixed designs were supplied to an environment without a design space")
            x = _fixed_designs(designs, self.episodes, self.budget, x_spec.shape)
            can_vectorize = self.env.vectorized
        elif can_vectorize and x_spec is not None:
            x = self._random_sequence(epoch)
        if can_vectorize or x_spec is None:
            batch = self._vectorized(draw, x, epoch, common_noise)
        else:
            batch = self._sequential(draw, policy, epoch, common_noise, infer, predict)
        batch.meta.update(epoch=epoch, seed=self.seed, budget=self.budget, episodes=self.episodes)
        dataset = EpisodeDataset(batch.detach())
        if save is not None:
            dataset.save(save)
        return dataset

    def _sequential(
        self,
        draw: Batch,
        policy: Callable[..., Any] | None,
        epoch: int,
        common_noise: bool,
        infer: bool,
        predict: bool,
    ) -> Batch:
        """Take prior draws and an adaptive policy and return their sequential rollout."""
        theta = torch.as_tensor(draw.parameters)
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        belief = draw.belief
        inference: Any = None
        prediction: Any = None
        mask = torch.ones((self.episodes, self.budget), dtype=torch.bool)
        for step in range(self.budget):
            history = Batch(
                parameters=theta,
                designs=None if not xs else torch.stack(xs, 1),
                outcomes=None if not ys else torch.stack(ys, 1),
                mask=mask[:, :step],
                target=draw.target,
                belief=belief,
                context=draw.context,
            )
            call_context = dict(
                generator=self.seeds.torch("design", epoch, step),
                epoch=epoch,
                step=step,
                batch_size=self.episodes,
            )
            if policy is not None:
                output = result(
                    self.env.invoke(
                        policy,
                        history,
                        infer=infer,
                        predict=predict,
                        env=self.env,
                        candidates=self.env.candidates(history, epoch=epoch, step=step),
                        **call_context,
                    ),
                    primary="design",
                )
            elif self.env.policy is not None:
                output = self.env.design(history, infer=infer, predict=predict, **call_context)
            else:
                output = result(random_policy(history, env=self.env, **call_context), primary="design")
            if output["infer"] is not None:
                inference = output["infer"]
            if output["predict"] is not None:
                prediction = output["predict"]
            if isinstance(inference, ParticleCloud):
                belief = inference
            chosen = output["design"]
            if chosen is None:
                raise RuntimeError("The design call did not return a design")
            sample = chosen if isinstance(chosen, PolicySample) else PolicySample(torch.as_tensor(chosen))
            current_x = torch.as_tensor(sample.x)
            if sample.log_prob is not None:
                log_probs.append(torch.as_tensor(sample.log_prob))
            if sample.entropy is not None:
                entropies.append(torch.as_tensor(sample.entropy))
            noise_label = "sim-common" if common_noise else f"sim-{getattr(policy, '__name__', 'policy')}"
            current_y = self.env.simulate(
                theta,
                current_x,
                generator=self.seeds.torch(noise_label, epoch, step),
                epoch=epoch,
                step=step,
                **draw.context,
            )
            xs.append(current_x)
            ys.append(torch.as_tensor(current_y))
        target = (
            self.env.invoke(self.env.target, theta, **draw.context)
            if self.env.target is not None
            else draw.target if draw.target is not None else theta
        )
        context = dict(draw.context)
        if log_probs:
            context["log_prob"] = torch.stack(log_probs, 1)
        if entropies:
            context["entropy"] = torch.stack(entropies, 1)
        if prediction is not None:
            context["predict"] = prediction
        if inference is not None and not isinstance(inference, ParticleCloud):
            context["infer"] = inference
        return Batch(
            parameters=theta,
            designs=torch.stack(xs, 1),
            outcomes=torch.stack(ys, 1),
            mask=mask,
            target=target,
            belief=belief,
            context=context,
        )

    def loaders(
        self,
        policy: Callable[..., Any] | None,
        *,
        epochs: int,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[DataLoader[Batch]]:
        """Take training settings and yield one newly generated loader per epoch."""
        for epoch in range(epochs):
            yield loader(self.generate(policy, epoch=epoch), batch_size, shuffle=shuffle, seed=self.seed, epoch=epoch)


class ReplayBuffer:
    """Keep a fixed number of recent Batch records and sample them again."""

    def __init__(self, capacity: int):
        """Take a positive capacity and create an empty replay buffer."""
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: list[Batch] = []

    def __len__(self) -> int:
        """Return the number of records currently stored."""
        return len(self._items)

    def add(self, batch: Batch) -> None:
        """Take a Batch, detach its rows, and add them to the buffer."""
        detached = batch.detach()
        self._items.extend(detached.index(index) for index in range(len(detached)))
        if len(self._items) > self.capacity:
            del self._items[: len(self._items) - self.capacity]

    def sample(
        self, count: int, *, seed: int | None = None, generator: torch.Generator | None = None
    ) -> Batch:
        """Take a sample count and return that many randomly selected stored records."""
        if not self._items:
            raise RuntimeError("Cannot sample an empty replay buffer")
        if generator is None:
            generator = torch.Generator().manual_seed(0 if seed is None else seed)
        indices = torch.randint(len(self._items), (count,), generator=generator)
        return collate([self._items[int(index)] for index in indices])


def load(path: str | Path, **kwargs: Any) -> EpisodeDataset:
    """Take a saved dataset path and return the restored EpisodeDataset."""
    return EpisodeDataset.load(path, **kwargs)
