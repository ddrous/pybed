"""Lightweight metrics for posterior, predictive, and design quality."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

from .core import ParticleCloud


def mse(estimate: torch.Tensor, truth: torch.Tensor, *, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Take estimates and truths and return their mean squared error."""
    estimate, truth = torch.as_tensor(estimate), torch.as_tensor(truth, device=estimate.device)
    return (estimate - truth).square().mean(dim=dim)


def rmse(estimate: torch.Tensor, truth: torch.Tensor, *, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Take estimates and truths and return their root mean squared error."""
    return mse(estimate, truth, dim=dim).sqrt()


def log_mse(
    estimate: torch.Tensor,
    truth: torch.Tensor,
    *,
    dim: int | tuple[int, ...] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Take estimates and truths and return the natural logarithm of their MSE."""
    return torch.log(mse(estimate, truth, dim=dim) + eps)


def nll(posterior: Any, truth: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    """Take a posterior and truths and return their negative log score."""
    score = -posterior.log_prob(truth) if hasattr(posterior, "log_prob") else -posterior(truth)
    if score.ndim > 1:
        score = score.flatten(1).sum(-1)
    if reduction == "none":
        return score
    if reduction == "sum":
        return score.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    return score.mean()


def energy_score(
    samples: torch.Tensor | ParticleCloud,
    truth: torch.Tensor,
    *,
    beta: float = 1.0,
    unbiased: bool = False,
) -> torch.Tensor:
    """Take posterior particles and truths and return their mean multivariate energy score.

    ``samples`` is ``(batch, particles, ...)`` and ``truth`` is ``(batch, ...)``.
    The returned scalar is lower-is-better and remains differentiable.
    """
    cloud = samples if isinstance(samples, ParticleCloud) else None
    samples = torch.as_tensor(cloud.points if cloud is not None else samples)
    if cloud is not None and cloud.particle_dim != 1:
        raise ValueError("Batched energy_score expects particle_dim=1")
    truth = torch.as_tensor(truth, device=samples.device)
    if cloud is not None and cloud.mask is not None:
        mask = torch.as_tensor(cloud.mask, dtype=samples.dtype, device=samples.device)
        samples = samples * mask[:, None, ...]
        truth = truth * mask
    flat = samples.flatten(2)
    target = truth.flatten(1)
    if cloud is None or cloud.weights is None:
        weights = torch.full(
            flat.shape[:2], 1 / flat.shape[1], dtype=flat.dtype, device=flat.device
        )
    else:
        weights = torch.as_tensor(cloud.weights, dtype=flat.dtype, device=flat.device)
        weights = weights / weights.sum(-1, keepdim=True)
    attraction = (
        torch.linalg.vector_norm(flat - target[:, None], dim=-1).pow(beta) * weights
    ).sum(-1)
    distances = torch.linalg.vector_norm(flat[:, :, None] - flat[:, None, :], dim=-1).pow(beta)
    pair_weights = weights[:, :, None] * weights[:, None, :]
    repulsion = (distances * pair_weights).sum((-1, -2))
    if unbiased:
        if flat.shape[1] < 2:
            raise ValueError("unbiased energy_score needs at least two particles")
        repulsion = repulsion / (1 - weights.square().sum(-1)).clamp_min(torch.finfo(flat.dtype).eps)
    return (attraction - 0.5 * repulsion).mean()


def coverage(
    samples: torch.Tensor,
    truth: torch.Tensor,
    *,
    mass: float = 0.9,
    dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Take posterior samples and truths and return equal-tailed interval coverage."""
    samples, truth = torch.as_tensor(samples), torch.as_tensor(truth, device=samples.device)
    lower = torch.quantile(samples, (1 - mass) / 2, dim=1)
    upper = torch.quantile(samples, 1 - (1 - mass) / 2, dim=1)
    covered = ((truth >= lower) & (truth <= upper)).to(samples.dtype)
    return covered.mean() if dim is None else covered.mean(dim=dim)


def calibration_error(samples: torch.Tensor, truth: torch.Tensor, *, bins: int = 10) -> torch.Tensor:
    """Take posterior samples and truths and return a rank calibration error."""
    samples, truth = torch.as_tensor(samples), torch.as_tensor(truth, device=samples.device)
    ranks = (samples < truth[:, None]).to(samples.dtype).mean(1).flatten()
    observed = torch.stack([(ranks <= level).to(samples.dtype).mean() for level in torch.linspace(0.1, 0.9, bins - 1)])
    expected = torch.linspace(0.1, 0.9, bins - 1, device=samples.device)
    return (observed - expected).abs().mean()


def set_mse(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Take estimated and true source sets and return permutation-invariant MSE.

    Shapes are ``(..., sources, dimensions)``.  Exact permutation matching keeps
    the implementation dependency-free and is intended for the small source
    counts used in BED benchmarks.
    """
    estimate, truth = torch.as_tensor(estimate), torch.as_tensor(truth, device=estimate.device)
    sources = estimate.shape[-2]
    if sources > 8:
        raise ValueError("Exact set_mse supports at most eight sources; use an assignment-based custom metric")
    costs = torch.stack(
        [
            (estimate - truth[..., list(order), :]).square().mean((-1, -2))
            for order in itertools.permutations(range(sources))
        ],
        -1,
    )
    return costs.min(-1).values.mean()


def eig(log_posterior: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
    """Take posterior and prior log densities and return Monte Carlo information gain."""
    return (torch.as_tensor(log_posterior) - torch.as_tensor(log_prior)).mean()


def spce(log_likelihoods: torch.Tensor) -> torch.Tensor:
    """Take true and contrastive history log likelihoods and return the sPCE bound.

    The first column must be the history log-likelihood under the generating
    parameter; remaining columns are prior contrastives.  Pass the summed
    log-likelihood across the whole sequential history, shape ``(B, L+1)``.
    """
    values = torch.as_tensor(log_likelihoods)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("spce expects shape (batch, truth+contrastives) with at least one contrastive")
    return (values[:, 0] - torch.logsumexp(values, 1) + np.log(values.shape[1])).mean()


def class_probabilities(
    posterior: torch.Tensor | ParticleCloud, *, classes: int | None = None
) -> torch.Tensor:
    """Take logits, probabilities, class particles, or one-hot particles and return class probabilities."""
    if isinstance(posterior, ParticleCloud):
        points = torch.as_tensor(posterior.points)
        if posterior.particle_dim != 1:
            raise ValueError("Batched class particles need particle_dim=1")
        if points.ndim == 2:
            classes = 10 if classes is None else classes
            points = torch.nn.functional.one_hot(points.long(), classes).to(torch.get_default_dtype())
        elif classes is not None and points.shape[-1] != classes:
            raise ValueError("One-hot class particles must have one value per class")
        if posterior.weights is None:
            weights = torch.full(points.shape[:2], 1 / points.shape[1], dtype=points.dtype, device=points.device)
        else:
            weights = torch.as_tensor(posterior.weights, dtype=points.dtype, device=points.device)
            weights = weights / weights.sum(-1, keepdim=True)
        return (points * weights[..., None]).sum(1)
    values = torch.as_tensor(posterior)
    if classes is not None and values.shape[-1] != classes:
        raise ValueError("Class predictions must have one value per class")
    is_probability = bool(torch.all(values >= 0)) and torch.allclose(
        values.sum(-1), torch.ones_like(values.sum(-1)), atol=1e-5
    )
    return values if is_probability else values.softmax(-1)


def classification_accuracy(
    posterior: torch.Tensor | ParticleCloud, truth: torch.Tensor, *, classes: int | None = None
) -> torch.Tensor:
    """Take class predictions and true labels and return their mean accuracy."""
    probabilities = class_probabilities(posterior, classes=classes)
    return (probabilities.argmax(-1) == torch.as_tensor(truth, device=probabilities.device)).float().mean()


def cross_entropy(
    posterior: torch.Tensor | ParticleCloud, truth: torch.Tensor, *, classes: int | None = None
) -> torch.Tensor:
    """Take class predictions and true labels and return their mean cross entropy."""
    if isinstance(posterior, ParticleCloud):
        probabilities = class_probabilities(posterior, classes=classes)
        return torch.nn.functional.nll_loss(probabilities.clamp_min(1e-12).log(), truth.long())
    return torch.nn.functional.cross_entropy(torch.as_tensor(posterior), truth.long())


def posterior_summary(
    samples: torch.Tensor | ParticleCloud,
    truth: torch.Tensor,
    *,
    log_density: Callable[[torch.Tensor], torch.Tensor] | None = None,
    interval: float = 0.9,
) -> dict[str, float]:
    """Take posterior samples and truths and return common scalar diagnostics."""
    cloud = samples if isinstance(samples, ParticleCloud) else None
    sample_values = torch.as_tensor(cloud.points if cloud is not None else samples)
    truth = torch.as_tensor(truth, device=sample_values.device)
    mean = cloud.mean() if cloud is not None else sample_values.mean(1)
    result = {
        "mse": float(mse(mean, truth)),
        "rmse": float(rmse(mean, truth)),
        "log_mse": float(log_mse(mean, truth)),
        "energy_score": float(energy_score(samples, truth)),
        f"coverage_{interval:g}": float(coverage(sample_values, truth, mass=interval)),
        "calibration_error": float(calibration_error(sample_values, truth)),
        "spread": float(sample_values.flatten(2).std(1).mean()),
    }
    if log_density is not None:
        result["nll"] = float(-log_density(truth).mean())
    return result


METRICS: Mapping[str, Callable[..., torch.Tensor]] = {
    "mse": mse,
    "rmse": rmse,
    "log_mse": log_mse,
    "nll": nll,
    "energy_score": energy_score,
    "coverage": coverage,
    "calibration_error": calibration_error,
    "set_mse": set_mse,
    "eig": eig,
    "spce": spce,
    "classification_accuracy": classification_accuracy,
    "cross_entropy": cross_entropy,
}


def get(name: str) -> Callable[..., torch.Tensor]:
    """Take a metric name and return its scoring function."""
    if name not in METRICS:
        raise KeyError(f"Unknown metric {name!r}; available: {sorted(METRICS)}")
    return METRICS[name]
