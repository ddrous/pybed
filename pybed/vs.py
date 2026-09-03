"""Camera-ready, dependency-light visualizers for PyBED objects."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import Batch


def style(context: str = "paper", font_scale: float = 1.25) -> None:
    """Apply PyBED's white-grid publication style without running at import time."""
    sns.set_theme(
        context=context,
        style="whitegrid",
        font_scale=font_scale,
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.labelpad": 7,
            "axes.titlepad": 10,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.frameon": True,
        },
    )
    plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "DejaVu Sans"})


def prior(
    *,
    env: Any,
    n: int = 2_000,
    seed: int = 0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Visualize low-dimensional marginals or representative field samples."""
    style()
    theta = torch.as_tensor(env.sample_prior(n, seed=seed)).detach().cpu()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
    else:
        fig = ax.figure
    if theta.ndim >= 4:
        ax.imshow(theta[0, 0], cmap="mako", origin="lower")
        ax.set(title=f"{env.name}: prior draw", xlabel="grid $x_1$", ylabel="grid $x_2$")
    else:
        flat = theta.reshape(n, -1)
        if flat.shape[1] == 1:
            sns.histplot(x=flat[:, 0].numpy(), bins=40, stat="density", ax=ax, color="#35618f")
            ax.set(xlabel=r"$\theta_1$", ylabel="density", title=f"{env.name}: prior")
        else:
            ax.scatter(flat[:, 0], flat[:, 1], s=10, alpha=0.25, color="#35618f", edgecolors="none")
            ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$", title=f"{env.name}: prior")
            ax.set_aspect("equal", adjustable="box")
    return fig, ax


def location(
    episode: Batch,
    *,
    env: Any,
    index: int = 0,
    posterior: torch.Tensor | np.ndarray | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot source truth, sequential sensor trajectory, and optional posterior cloud."""
    style()
    batch = episode.numpy()
    theta = np.asarray(batch.theta[index] if np.asarray(batch.theta).ndim >= 3 else batch.theta)
    design = np.asarray(
        batch.design[index] if batch.design is not None and np.asarray(batch.design).ndim >= 3 else batch.design
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 5.8))
    else:
        fig = ax.figure
    dims = theta.shape[-1]
    if dims == 1:
        steps = np.arange(len(design))
        ax.scatter(design[:, 0], steps, c=steps, cmap="viridis", s=65, label="design")
        for source in theta[:, 0]:
            ax.axvline(source, color="#d1495b", lw=2.5, ls="--")
        ax.set(xlabel="location", ylabel="design step", title="Location-finding trajectory")
    elif dims == 2:
        steps = np.arange(len(design))
        if posterior is not None:
            cloud = np.asarray(torch.as_tensor(posterior).detach().cpu()).reshape(-1, 2)
            ax.scatter(cloud[:, 0], cloud[:, 1], s=15, alpha=0.18, color="#8f6bb3", label="posterior")
        path = ax.scatter(design[:, 0], design[:, 1], c=steps, cmap="viridis", s=60, zorder=3, label="design")
        ax.plot(design[:, 0], design[:, 1], color="#4f5d75", alpha=0.35, lw=1.5)
        ax.scatter(
            theta[:, 0],
            theta[:, 1],
            marker="*",
            s=260,
            color="#d1495b",
            edgecolor="white",
            lw=1,
            label="truth",
            zorder=5,
        )
        fig.colorbar(path, ax=ax, label="design step")
        ax.set(xlabel=r"design $\xi_1$", ylabel=r"design $\xi_2$", title="Location-finding trajectory")
        ax.set_xlim(env.cfg["low"], env.cfg["high"])
        ax.set_ylim(env.cfg["low"], env.cfg["high"])
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
    else:
        raise ValueError(
            "The built-in location plot supports one or two dimensions; supply a custom visualizer for more"
        )
    return fig, ax


def image_discovery(
    episode: Batch,
    *,
    env: Any,
    index: int = 0,
    posterior: torch.Tensor | np.ndarray | None = None,
) -> tuple[Figure, np.ndarray]:
    """Show ground truth, cumulative measurements, posterior, and design centers."""
    style()
    batch = episode.numpy()
    theta = np.asarray(batch.theta[index])[0]
    obs = np.asarray(batch.obs[index])[:, 0]
    design = np.asarray(batch.design[index])
    cumulative = obs.max(0)
    estimate = cumulative if posterior is None else np.asarray(torch.as_tensor(posterior).detach().cpu()).squeeze()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    for axis, image, title in zip(
        axes,
        (theta, cumulative, estimate),
        ("ground truth", "cumulative measurement", "posterior / reconstruction"),
        strict=True,
    ):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set(xticks=[], yticks=[])
    axes[1].scatter(
        design[:, 1] * (theta.shape[1] - 1),
        design[:, 0] * (theta.shape[0] - 1),
        c=np.arange(len(design)),
        cmap="viridis",
        s=45,
        edgecolor="white",
    )
    return fig, axes


def pde(
    episode: Batch,
    *,
    env: Any,
    index: int = 0,
) -> tuple[Figure, np.ndarray]:
    """Plot an initial field, selected sensors, and their observation time series."""
    style()
    batch = episode.numpy()
    field = np.asarray(batch.theta[index]).squeeze()
    designs = np.asarray(batch.design[index])
    observations = np.asarray(batch.obs[index])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    image = axes[0].imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="mako")
    points = designs.reshape(-1, 2)
    axes[0].scatter(points[:, 1], points[:, 0], c=np.arange(len(points)), cmap="flare", s=55, edgecolor="white")
    axes[0].set(title="initial field and sensor designs", xlabel=r"$x_1$", ylabel=r"$x_2$")
    fig.colorbar(image, ax=axes[0], label="field value")
    times = np.asarray(env.cfg["times"])
    for step in range(observations.shape[0]):
        for sensor in range(observations.shape[1]):
            axes[1].plot(times, observations[step, sensor], alpha=0.65, lw=1.6)
    axes[1].set(title="sensor observations", xlabel="time", ylabel="concentration")
    return fig, axes


def timeseries(
    episode: Batch,
    *,
    env: Any,
    index: int = 0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Generic trajectory view for scalar designs or vector time-series outcomes."""
    style()
    batch = episode.numpy()
    obs = np.asarray(batch.obs[index])
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    else:
        fig = ax.figure
    if obs.shape[-1] == 1:
        ax.plot(np.arange(len(obs)), obs[:, 0], marker="o", lw=2, color="#35618f")
        ax.set(xlabel="design step", ylabel="observation")
    else:
        for step, values in enumerate(obs):
            ax.plot(np.arange(values.size), values.reshape(-1), lw=1.8, alpha=0.72, label=f"step {step + 1}")
        ax.set(xlabel="observation time", ylabel="value")
        if len(obs) <= 8:
            ax.legend(ncol=2)
    ax.set_title(env.name)
    return fig, ax


def training(
    records: Sequence[Mapping[str, float]] | str | Path,
    *,
    keys: Sequence[str] | None = None,
    logy: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot JSONL or in-memory training diagnostics; loss uses log scale by default."""
    style()
    if isinstance(records, (str, Path)):
        records = [json.loads(line) for line in Path(records).read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("No training records to plot")
    if keys is None:
        keys = [key for key in records[0] if key not in {"step", "epoch", "time"}]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    else:
        fig = ax.figure
    xkey = "step" if "step" in records[0] else "epoch"
    x = [row.get(xkey, i) for i, row in enumerate(records)]
    for key in keys:
        y = [row.get(key, np.nan) for row in records]
        ax.plot(x, y, lw=2, label=key.replace("_", " "))
    ax.set(xlabel=xkey, ylabel="metric", title="Training diagnostics")
    if logy:
        ax.set_yscale("log")
    ax.legend()
    return fig, ax


def benchmark(
    results: Mapping[str, Mapping[str, float]],
    *,
    metric: str,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Compact policy comparison plot from ``exp.compare`` results."""
    style()
    names = list(results)
    values = [results[name][metric] for name in names]
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(names)), 4.8))
    else:
        fig = ax.figure
    sns.barplot(x=names, y=values, hue=names, legend=False, palette="crest", ax=ax)
    ax.set(xlabel="policy", ylabel=metric.replace("_", " "), title=f"Policy comparison: {metric.replace('_', ' ')}")
    ax.tick_params(axis="x", rotation=20)
    return fig, ax
