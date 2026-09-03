"""Clear visual checks and paper figures for PyBED experiments."""

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

from .core import Batch, ParticleCloud
from .metrics import class_probabilities


def style(context: str = "paper", font_scale: float = 1.55) -> None:
    """Take plot scale settings and apply PyBED's publication style, returning nothing."""
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
            "grid.color": "#8d99a6",
            "grid.alpha": 0.24,
            "grid.linewidth": 0.8,
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
    """Take an environment and draw count and return a prior figure and axes."""
    style()
    draw = env.sample_prior(n, seed=seed)
    theta = torch.as_tensor(draw.parameters if isinstance(draw, Batch) else draw).detach().cpu()
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
    """Take a location episode and optional posterior cloud and return its figure and axes."""
    style()
    batch = episode.numpy()
    theta = np.asarray(
        batch.parameters[index] if np.asarray(batch.parameters).ndim >= 3 else batch.parameters
    )
    x = np.asarray(
        batch.designs[index]
        if batch.designs is not None and np.asarray(batch.designs).ndim >= 3
        else batch.designs
    )
    if "source_dim" in batch.context:
        source_dim = np.asarray(batch.context["source_dim"])
        dims = int(source_dim[index] if source_dim.ndim else source_dim)
        theta, x = theta[..., :dims], x[..., :dims]
    else:
        dims = theta.shape[-1]
    if "source_mask" in batch.context:
        source_mask = np.asarray(batch.context["source_mask"])
        active = source_mask[index] if source_mask.ndim > 1 else source_mask
        theta = theta[np.asarray(active, dtype=bool)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.8, 5.8))
    else:
        fig = ax.figure
    if dims == 1:
        steps = np.arange(len(x))
        ax.scatter(x[:, 0], steps, c=steps, cmap="viridis", s=65, label="design")
        for source in theta[:, 0]:
            ax.axvline(source, color="#d1495b", lw=2.5, ls="--")
        ax.set(xlabel="location", ylabel="design step", title="Location-finding trajectory")
    elif dims == 2:
        steps = np.arange(len(x))
        if posterior is not None:
            cloud = np.asarray(torch.as_tensor(posterior).detach().cpu()).reshape(-1, 2)
            ax.scatter(cloud[:, 0], cloud[:, 1], s=15, alpha=0.18, color="#8f6bb3", label="posterior")

        path = ax.scatter(x[:, 0], x[:, 1], c=steps, cmap="viridis", s=68, edgecolor="white", linewidth=0.65, zorder=3, label="design")

        ax.scatter(theta[:, 0], theta[:, 1], marker="*", s=260, color="#d1495b", edgecolor="white", lw=1, label="truth", zorder=5)

        fig.colorbar(path, ax=ax, label="design step")
        ax.set(
            xlabel=r"design $x_1$",
            ylabel=r"design $x_2$",
            title="Location-finding trajectory",
        )
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
    """Take an image-discovery episode and return a three-panel figure and axes."""
    style()
    batch = episode.numpy()
    theta = np.asarray(batch.parameters[index])[0]
    y = np.asarray(batch.outcomes[index])[:, 0]
    x = np.asarray(batch.designs[index])
    cumulative = y.max(0)
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
        x[:, 1] * (theta.shape[1] - 1),
        x[:, 0] * (theta.shape[0] - 1),
        c=np.arange(len(x)),
        cmap="viridis",
        s=45,
        edgecolor="white",
    )
    return fig, axes


def image_classification(
    episode: Batch,
    *,
    env: Any,
    index: int = 0,
) -> tuple[Figure, np.ndarray]:
    """Take a masked-MNIST episode and return its image, chosen patches, and outcomes."""
    style()
    batch = episode.numpy()
    image = np.asarray(batch.parameters[index]).squeeze()
    x = np.asarray(batch.designs[index])
    y = np.asarray(batch.outcomes[index])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].scatter(
        x[:, 1] * (image.shape[1] - 1),
        x[:, 0] * (image.shape[0] - 1),
        c=np.arange(len(x)),
        cmap="viridis",
        s=45,
    )
    axes[0].set(title=f"digit {int(np.asarray(batch.target[index]))}", xticks=[], yticks=[])
    axes[1].imshow(y[:, 0].mean(0), cmap="gray")
    axes[1].set(title="mean observed patch", xticks=[], yticks=[])
    axes[2].plot(np.arange(len(x)), np.linalg.norm(y.reshape(len(y), -1), axis=1), marker="o")
    axes[2].set(title="patch signal", xlabel="design step", ylabel="norm")
    return fig, axes


def pde(
    episode: Batch,
    *,
    env: Any,
    index: int = 0,
) -> tuple[Figure, np.ndarray]:
    """Take a PDE episode and return its field/sensor and time-series panels."""
    style()
    batch = episode.numpy()
    field = np.asarray(batch.parameters[index]).squeeze()
    designs = np.asarray(batch.designs[index])
    observations = np.asarray(batch.outcomes[index])
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
    """Take a trajectory episode and return a time-series figure and axes."""
    style()
    batch = episode.numpy()
    obs = np.asarray(batch.outcomes[index])
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
    """Take training records and return a diagnostic figure and axes."""
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
    """Take policy results and a metric name and return a comparison figure and axes."""
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


def particles(
    cloud: ParticleCloud | torch.Tensor | np.ndarray,
    *,
    truth: torch.Tensor | np.ndarray | None = None,
    index: int = 0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Take posterior particles and optional truth and return a one- or two-dimensional view."""
    style()
    weights = None
    if isinstance(cloud, ParticleCloud):
        values = np.asarray(torch.as_tensor(cloud.points).detach().cpu())
        if cloud.particle_dim == 1:
            values = values[index]
            if cloud.weights is not None:
                weights = np.asarray(torch.as_tensor(cloud.weights).detach().cpu())[index]
        elif cloud.particle_dim != 0:
            raise ValueError("The particle plot expects an unbatched cloud or batch-by-particle cloud")
    else:
        values = np.asarray(torch.as_tensor(cloud).detach().cpu())
        values = values[index] if values.ndim > 2 else values
    values = values.reshape(len(values), -1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
    else:
        fig = ax.figure
    truth_value = None if truth is None else np.asarray(torch.as_tensor(truth).detach().cpu())
    if truth_value is not None and truth_value.ndim > 1:
        truth_value = truth_value[index]
    if values.shape[1] == 1:
        ax.hist(values[:, 0], bins=35, weights=weights, density=True, color="#35618f", alpha=0.8)
        if truth_value is not None:
            ax.axvline(float(np.ravel(truth_value)[0]), color="#d1495b", lw=2.5, label="truth")
        ax.set(xlabel=r"$\theta$", ylabel="density", title="Posterior particles")
    else:
        colors = weights if weights is not None else "#35618f"
        points = ax.scatter(values[:, 0], values[:, 1], c=colors, cmap="viridis", s=24, alpha=0.65)
        if weights is not None:
            fig.colorbar(points, ax=ax, label="particle weight")
        if truth_value is not None:
            target = np.ravel(truth_value)
            ax.scatter(target[0], target[1], marker="*", s=240, color="#d1495b", label="truth")
        ax.set(xlabel=r"$\theta_1$", ylabel=r"$\theta_2$", title="Posterior particles")
    if truth_value is not None:
        ax.legend()
    return fig, ax


def class_belief(
    posterior: ParticleCloud | torch.Tensor,
    *,
    truth: int | torch.Tensor | None = None,
    index: int = 0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Take class logits, probabilities, or particles and return a probability bar chart."""
    probabilities = class_probabilities(posterior).detach().cpu()
    if probabilities.ndim > 1:
        probabilities = probabilities[index]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
    else:
        fig = ax.figure
    labels = np.arange(len(probabilities))
    colors = ["#d1495b" if truth is not None and label == int(torch.as_tensor(truth)) else "#35618f" for label in labels]
    ax.bar(labels, probabilities.numpy(), color=colors)
    ax.set(xticks=labels, xlabel="class", ylabel="probability", ylim=(0, 1), title="Class belief")
    return fig, ax


def acquisition(
    candidates: torch.Tensor | np.ndarray,
    scores: torch.Tensor | np.ndarray,
    *,
    chosen: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Take candidate designs and their scores and return an acquisition plot."""
    x = np.asarray(torch.as_tensor(candidates).detach().cpu()).reshape(len(candidates), -1)
    values = np.asarray(torch.as_tensor(scores).detach().cpu()).reshape(-1)
    if len(x) != len(values):
        raise ValueError("candidates and scores must contain the same number of entries")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
    else:
        fig = ax.figure
    if x.shape[1] == 1:
        ax.scatter(x[:, 0], values, c=values, cmap="viridis", s=55)
        if chosen is not None:
            ax.scatter(x[chosen, 0], values[chosen], marker="*", s=240, color="#d1495b")
        ax.set(xlabel=r"candidate $x$", ylabel="score")
    elif x.shape[1] == 2:
        points = ax.scatter(x[:, 0], x[:, 1], c=values, cmap="viridis", s=65)
        fig.colorbar(points, ax=ax, label="score")
        if chosen is not None:
            ax.scatter(*x[chosen], marker="*", s=240, color="#d1495b")
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    else:
        ax.plot(values, marker="o")
        if chosen is not None:
            ax.scatter(chosen, values[chosen], marker="*", s=220, color="#d1495b")
        ax.set(xlabel="candidate", ylabel="score")
    ax.set_title("Candidate designs")
    return fig, ax


def calibration(
    samples: torch.Tensor | np.ndarray,
    truth: torch.Tensor | np.ndarray,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Take posterior samples and truths and return their empirical rank-calibration curve."""
    draws = torch.as_tensor(samples)
    targets = torch.as_tensor(truth, device=draws.device)
    ranks = (draws < targets[:, None]).float().mean(1).flatten().sort().values.cpu().numpy()
    observed = np.arange(1, len(ranks) + 1) / len(ranks)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
    else:
        fig = ax.figure
    ax.plot([0, 1], [0, 1], ls="--", color="#6c757d", label="calibrated")
    ax.step(ranks, observed, where="post", color="#35618f", lw=2.2, label="posterior")
    ax.set(xlabel="posterior rank", ylabel="empirical frequency", title="Calibration", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    return fig, ax


def observation_history(
    batch: Batch,
    *,
    index: int = 0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Take an episode batch and return a heatmap of its outcomes over design steps."""
    if batch.outcomes is None:
        raise ValueError("The batch has no outcomes to plot")
    values = np.asarray(batch.outcomes.detach().cpu() if torch.is_tensor(batch.outcomes) else batch.outcomes)
    values = values[index] if values.ndim > 1 else values
    matrix = values.reshape(values.shape[0], -1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
    else:
        fig = ax.figure
    image = ax.imshow(matrix, aspect="auto", cmap="mako")
    fig.colorbar(image, ax=ax, label="outcome")
    ax.set(xlabel="outcome component", ylabel="design step", title="Observation history")
    return fig, ax
