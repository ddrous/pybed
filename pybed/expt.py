"""Small experiment records, checkpoints, training, and policy comparison."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import random
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .core import BED, Batch
from .data import EpisodeDataset, Stream


def _plain(value: Any) -> Any:
    """Take a configuration value and return plain data that YAML can save."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_config(path: str | Path) -> dict[str, Any]:
    """Take a YAML path and return its configuration dictionary."""
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("A configuration file must contain a YAML mapping")
    return dict(loaded)


def save_config(config: Mapping[str, Any] | Any, path: str | Path) -> Path:
    """Take a configuration and path, save readable YAML, and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_plain(config), sort_keys=False), encoding="utf-8")
    return path


def snapshot(files: Sequence[str | Path], folder: str | Path) -> Path:
    """Take source files and a run folder and return a small code snapshot folder."""
    destination = Path(folder) / "code"
    destination.mkdir(parents=True, exist_ok=True)
    for source in map(Path, files):
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    project = Path(__file__).resolve().parents[1]
    try:
        changes = subprocess.run(
            ["git", "-C", str(project), "diff", "--binary"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        if changes:
            (destination / "working-tree.patch").write_text(changes, encoding="utf-8")
    except (OSError, subprocess.SubprocessError):
        pass
    return destination


def seed(seed: int, *, deterministic: bool = True) -> None:
    """Take one seed and apply it to Python, NumPy, and Torch, returning nothing."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@dataclass
class Run:
    """A transparent local run folder with JSONL logs and resumable checkpoints."""

    path: Path
    cfg: dict[str, Any]
    started: float

    @classmethod
    def create(
        cls,
        root: str | Path,
        name: str,
        cfg: Mapping[str, Any] | Any,
        *,
        run_id: str | None = None,
        seed_value: int = 0,
        source_files: Sequence[str | Path] = (),
    ) -> Run:
        """Take run settings and source files and return a new local experiment record."""
        root = Path(root)
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        normalized = _plain(cfg)
        if not isinstance(normalized, Mapping):
            raise TypeError("Run configuration must be a mapping or dataclass")
        normalized = dict(normalized)
        digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode()).hexdigest()[:8]
        run_id = run_id or f"{timestamp}-{digest}"
        path = root / name / run_id
        if path.exists():
            raise FileExistsError(f"Run already exists: {path}")
        for folder in (path, path / "checkpoints", path / "figures", path / "data", path / "code"):
            folder.mkdir(parents=True, exist_ok=True)
        seed(seed_value)
        git_commit = None
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        metadata = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "seed": seed_value,
            "config_sha256": digest,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git_commit": git_commit,
        }
        save_config(normalized, path / "config.yaml")
        (path / "meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
        snapshot(source_files, path)
        return cls(path=path, cfg=normalized, started=time.time())

    def log(self, step: int, **metrics: float) -> dict[str, float]:
        """Take a step and scalar metrics, save them, and return the saved row."""
        row = {"step": int(step), "elapsed_s": time.time() - self.started}
        row.update({name: float(value) for name, value in metrics.items()})
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def checkpoint(
        self,
        name: str,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        """Take model state and a name, save a checkpoint, and return its path."""
        path = self.path / "checkpoints" / f"{name}.pt"
        payload = {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "step": step,
            "config": self.cfg,
            "extra": dict(extra or {}),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(payload, path)
        return path

    def restore(
        self,
        checkpoint: str | Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Take a checkpoint and live objects, restore their state, and return its payload."""
        payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload["optimizer"] is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if restore_rng:
            random.setstate(payload["rng"]["python"])
            np.random.set_state(payload["rng"]["numpy"])
            torch.set_rng_state(payload["rng"]["torch"])
            if torch.cuda.is_available() and payload["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        return payload

    def save_figure(self, figure: Any, name: str, *, dpi: int = 180) -> Path:
        """Take a figure and name, save the image, and return its path."""
        path = self.path / "figures" / f"{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        return path

    def save_data(self, dataset: EpisodeDataset, name: str) -> Path:
        """Take an episode dataset and name, save it, and return its path."""
        return dataset.save(self.path / "data" / f"{name}.pt")

    def finish(self, **summary: Any) -> Path:
        """Take final summary values, save them, and return the summary path."""
        summary = {"elapsed_s": time.time() - self.started, **summary}
        path = self.path / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
        return path


def train(
    model: torch.nn.Module,
    loaders: Iterable[Iterable[Batch]],
    optimizer: torch.optim.Optimizer,
    loss: Callable[[torch.nn.Module, Batch], torch.Tensor | tuple[torch.Tensor, Mapping[str, float]]],
    *,
    run: Run | None = None,
    device: str | torch.device = "cpu",
    grad_clip: float | None = None,
    start_epoch: int = 0,
) -> list[dict[str, float]]:
    """Take a model, loaders, optimizer, and loss and return per-epoch training records.

    Each yielded loader defines one epoch.  It can be a pass over a saved dataset
    or one freshly materialized ``Stream`` epoch; the optimization semantics
    remain identical.
    """
    model.to(device)
    records: list[dict[str, float]] = []
    global_step = 0
    for epoch, batches in enumerate(loaders, start=start_epoch):
        model.train()
        totals: dict[str, float] = {"loss": 0.0}
        count = 0
        for batch in batches:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = loss(model, batch)
            value, diagnostics = output if isinstance(output, tuple) else (output, {})
            if not torch.isfinite(value):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {count}")
            value.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            totals["loss"] += float(value.detach())
            for name, metric in diagnostics.items():
                totals[name] = totals.get(name, 0.0) + float(metric)
            count += 1
            global_step += 1
        averaged = {name: value / max(1, count) for name, value in totals.items()}
        row = {"epoch": float(epoch), "step": float(global_step), **averaged}
        records.append(row)
        if run is not None:
            run.log(global_step, epoch=epoch, **averaged)
    return records


def compare(
    env: BED,
    policies: Mapping[str, Callable[..., torch.Tensor] | None],
    score: Callable[[BED, Batch], Mapping[str, float] | float],
    *,
    episodes: int = 256,
    seed_value: int = 0,
    epoch: int = 0,
    budget: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, EpisodeDataset]]:
    """Take named policies and a score and return comparable results and trajectories."""
    stream = Stream(env, episodes, seed=seed_value, budget=budget)
    results: dict[str, dict[str, float]] = {}
    datasets: dict[str, EpisodeDataset] = {}
    reference_parameters = stream.parameters(epoch)
    for name, policy in policies.items():
        dataset = stream.generate(policy, epoch=epoch, common_noise=True)
        if not torch.equal(reference_parameters, dataset.batch.parameters):
            raise RuntimeError(f"Common-random-number invariant failed for policy {name}")
        value = score(env, dataset.batch)
        results[name] = (
            {"score": float(value)}
            if not isinstance(value, Mapping)
            else {key: float(metric) for key, metric in value.items()}
        )
        datasets[name] = dataset
    return results, datasets
