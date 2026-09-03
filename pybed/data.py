"""Reproducible online and offline data for fair BED comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .core import BED, Batch


@dataclass(frozen=True)
class SeedBank:
    """Order-independent random streams derived from human-readable namespaces."""

    seed: int = 0

    def value(self, namespace: str, *indices: int) -> int:
        payload = ":".join((str(self.seed), namespace, *(str(i) for i in indices))).encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)

    def torch(self, namespace: str, *indices: int, device: str | torch.device = "cpu") -> torch.Generator:
        return torch.Generator(device=device).manual_seed(self.value(namespace, *indices))

    def numpy(self, namespace: str, *indices: int) -> np.random.Generator:
        return np.random.default_rng(self.value(namespace, *indices))


class EpisodeDataset(Dataset[Batch]):
    """Finite, serializable episodes used for posterior-only or policy training."""

    def __init__(self, batch: Batch):
        if batch.theta is None or batch.design is None or batch.obs is None:
            raise ValueError("An episode dataset requires theta, design, and obs")
        self.batch = batch

    def __len__(self) -> int:
        return len(self.batch)

    def __getitem__(self, index: int) -> Batch:
        values: dict[str, Any] = {}
        for name in ("theta", "design", "obs", "mask", "target"):
            value = getattr(self.batch, name)
            values[name] = value[index] if hasattr(value, "__getitem__") and value is not None else value
        return Batch(**values, context=self.batch.context, meta=self.batch.meta)

    def save(self, path: str | Path) -> Path:
        """Save tensors and a readable sidecar manifest without extra dependencies."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: getattr(self.batch, name) for name in ("theta", "design", "obs", "mask", "target")}
        payload.update(context=self.batch.context, meta=self.batch.meta, format="pybed-episodes-v1")
        torch.save(payload, path)
        manifest = {
            "format": "pybed-episodes-v1",
            "episodes": len(self),
            "shapes": {name: list(value.shape) for name, value in payload.items() if torch.is_tensor(value)},
            "meta": self.batch.meta,
        }
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> EpisodeDataset:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("format") != "pybed-episodes-v1":
            raise ValueError("Not a supported PyBED episode file")
        return cls(
            Batch(
                theta=payload["theta"],
                design=payload["design"],
                obs=payload["obs"],
                mask=payload.get("mask"),
                target=payload.get("target"),
                context=payload.get("context", {}),
                meta=payload.get("meta", {}),
            )
        )


def collate(items: list[Batch]) -> Batch:
    """Collate fixed or padded BED records while preserving experiment metadata."""
    values: dict[str, Any] = {}
    for name in ("theta", "design", "obs", "mask", "target"):
        field = [getattr(item, name) for item in items]
        if field[0] is None:
            values[name] = None
        elif torch.is_tensor(field[0]):
            values[name] = torch.stack(field)
        elif isinstance(field[0], np.ndarray):
            values[name] = torch.as_tensor(np.stack(field))
        else:
            values[name] = field
    return Batch(**values, context=items[0].context, meta=items[0].meta)


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
    """Create a deterministic loader; a complete pass is one offline epoch."""
    seeds = SeedBank(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=seeds.torch("loader", epoch),
        num_workers=workers,
        drop_last=drop_last,
        collate_fn=collate,
    )


def random_policy(
    history: Batch | None,
    *,
    env: BED,
    generator: torch.Generator | None = None,
    batch_size: int | None = None,
    **context: Any,
) -> torch.Tensor:
    """Uniform baseline over the environment's bounded design event shape."""
    spec = env.specs["design"]
    if spec.low is None or spec.high is None:
        raise ValueError("RandomPolicy needs finite low/high bounds in env.specs['design']")
    if batch_size is None:
        batch_size = len(history) if history is not None else int(context.get("n", 1))
    return spec.low + (spec.high - spec.low) * torch.rand((batch_size, *spec.shape), generator=generator)


class EpochStream:
    """Materialize comparable synthetic epochs for one or many policies.

    ``theta(epoch)`` depends only on the root seed, environment and epoch.  Policy
    rollouts use separate design and simulator streams, so two methods always see
    the same ordered truths within an epoch.  Saving a generated epoch turns the
    exact same object into an ordinary finite dataset for posterior-only training.
    """

    def __init__(self, env: BED, episodes: int, *, seed: int = 0, budget: int | None = None):
        self.env, self.episodes, self.seed = env, episodes, seed
        self.budget = env.budget if budget is None else budget
        self.seeds = SeedBank(seed)
        self.theta_cache: dict[int, torch.Tensor] = {}

    def theta(self, epoch: int = 0) -> torch.Tensor:
        if epoch not in self.theta_cache:
            generator = self.seeds.torch(f"{self.env.name}:theta", epoch)
            self.theta_cache[epoch] = torch.as_tensor(
                self.env.sample_prior(self.episodes, generator=generator, epoch=epoch)
            )
        return self.theta_cache[epoch].clone()

    def generate(
        self,
        policy: Callable[..., torch.Tensor] | None = None,
        *,
        epoch: int = 0,
        common_noise: bool = True,
        save: str | Path | None = None,
    ) -> EpisodeDataset:
        theta = self.theta(epoch)
        designs: list[torch.Tensor] = []
        observations: list[torch.Tensor] = []
        mask = torch.ones((self.episodes, self.budget), dtype=torch.bool)
        selected_policy = random_policy if policy is None else policy

        for step in range(self.budget):
            history = Batch(
                theta=theta,
                design=None if not designs else torch.stack(designs, 1),
                obs=None if not observations else torch.stack(observations, 1),
                mask=mask[:, :step],
                context={"epoch": epoch, "step": step},
            )
            design_generator = self.seeds.torch("design", epoch, step)
            if selected_policy is random_policy:
                design = random_policy(history, env=self.env, generator=design_generator, batch_size=self.episodes)
            else:
                design = self.env.invoke(
                    selected_policy,
                    history,
                    env=self.env,
                    generator=design_generator,
                    epoch=epoch,
                    step=step,
                    batch_size=self.episodes,
                )
            noise_namespace = (
                "sim-common" if common_noise else f"sim-{getattr(policy, '__name__', type(policy).__name__)}"
            )
            observation = self.env.simulate(
                theta,
                design,
                generator=self.seeds.torch(noise_namespace, epoch, step),
                epoch=epoch,
                step=step,
            )
            designs.append(torch.as_tensor(design).detach().cpu())
            observations.append(torch.as_tensor(observation).detach().cpu())

        target = self.env.invoke(self.env.target, theta) if self.env.target is not None else theta
        dataset = EpisodeDataset(
            Batch(
                theta=theta.detach().cpu(),
                design=torch.stack(designs, 1),
                obs=torch.stack(observations, 1),
                mask=mask,
                target=torch.as_tensor(target).detach().cpu() if torch.is_tensor(target) else target,
                context={"env": self.env.name},
                meta={"epoch": epoch, "seed": self.seed, "budget": self.budget, "episodes": self.episodes},
            )
        )
        if save is not None:
            dataset.save(save)
        return dataset

    def loaders(
        self,
        policy: Callable[..., torch.Tensor] | None,
        *,
        epochs: int,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[DataLoader[Batch]]:
        """Yield finite online epochs; every epoch contains exactly ``episodes`` fresh truths."""
        for epoch in range(epochs):
            yield loader(self.generate(policy, epoch=epoch), batch_size, shuffle=shuffle, seed=self.seed, epoch=epoch)


def load(path: str | Path, **kwargs: Any) -> EpisodeDataset:
    """Short public alias for loading a saved synthetic BED dataset."""
    return EpisodeDataset.load(path, **kwargs)
