"""Lightweight metrics for posterior, predictive, and design quality."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch


def mse(estimate: torch.Tensor, truth: torch.Tensor, *, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Mean squared error, reduced over all elements unless ``dim`` is supplied."""
    estimate, truth = torch.as_tensor(estimate), torch.as_tensor(truth, device=estimate.device)
    return (estimate - truth).square().mean(dim=dim)


def rmse(estimate: torch.Tensor, truth: torch.Tensor, *, dim: int | tuple[int, ...] | None = None) -> torch.Tensor:
    return mse(estimate, truth, dim=dim).sqrt()


def log_mse(
    estimate: torch.Tensor,
    truth: torch.Tensor,
    *,
    dim: int | tuple[int, ...] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Natural logarithm of MSE, useful across orders of magnitude."""
    return torch.log(mse(estimate, truth, dim=dim) + eps)


def nll(posterior: Any, truth: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    """Negative log score for a distribution or a callable log-density."""
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


def energy_score(samples: torch.Tensor, truth: torch.Tensor, *, beta: float = 1.0) -> torch.Tensor:
    """Multivariate proper score for posterior samples.

    ``samples`` is ``(batch, particles, ...)`` and ``truth`` is ``(batch, ...)``.
    The returned scalar is lower-is-better and remains differentiable.
    """
    samples, truth = torch.as_tensor(samples), torch.as_tensor(truth, device=samples.device)
    flat = samples.flatten(2)
    target = truth.flatten(1)
    attraction = torch.linalg.vector_norm(flat - target[:, None], dim=-1).pow(beta).mean(-1)
    repulsion = torch.linalg.vector_norm(flat[:, :, None] - flat[:, None, :], dim=-1).pow(beta).mean((-1, -2))
    return (attraction - 0.5 * repulsion).mean()


def coverage(
    samples: torch.Tensor,
    truth: torch.Tensor,
    *,
    mass: float = 0.9,
    dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Empirical equal-tailed credible-interval coverage."""
    samples, truth = torch.as_tensor(samples), torch.as_tensor(truth, device=samples.device)
    lower = torch.quantile(samples, (1 - mass) / 2, dim=1)
    upper = torch.quantile(samples, 1 - (1 - mass) / 2, dim=1)
    covered = ((truth >= lower) & (truth <= upper)).to(samples.dtype)
    return covered.mean() if dim is None else covered.mean(dim=dim)


def calibration_error(samples: torch.Tensor, truth: torch.Tensor, *, bins: int = 10) -> torch.Tensor:
    """Simulation-based calibration error from posterior ranks."""
    samples, truth = torch.as_tensor(samples), torch.as_tensor(truth, device=samples.device)
    ranks = (samples < truth[:, None]).to(samples.dtype).mean(1).flatten()
    observed = torch.stack([(ranks <= level).to(samples.dtype).mean() for level in torch.linspace(0.1, 0.9, bins - 1)])
    expected = torch.linspace(0.1, 0.9, bins - 1, device=samples.device)
    return (observed - expected).abs().mean()


def set_mse(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant MSE for exchangeable source sets.

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
    """Monte Carlo expected information gain from posterior and prior log density."""
    return (torch.as_tensor(log_posterior) - torch.as_tensor(log_prior)).mean()


def spce(log_likelihoods: torch.Tensor) -> torch.Tensor:
    """Sequential prior-contrastive EIG lower bound.

    The first column must be the history log-likelihood under the generating
    parameter; remaining columns are prior contrastives.  Pass the summed
    log-likelihood across the whole sequential history, shape ``(B, L+1)``.
    """
    values = torch.as_tensor(log_likelihoods)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("spce expects shape (batch, truth+contrastives) with at least one contrastive")
    return (values[:, 0] - torch.logsumexp(values, 1) + np.log(values.shape[1])).mean()


def posterior_summary(
    samples: torch.Tensor,
    truth: torch.Tensor,
    *,
    log_density: Callable[[torch.Tensor], torch.Tensor] | None = None,
    interval: float = 0.9,
) -> dict[str, float]:
    """Common paper-table posterior diagnostics with consistent reductions."""
    samples, truth = torch.as_tensor(samples), torch.as_tensor(truth, device=samples.device)
    mean = samples.mean(1)
    result = {
        "mse": float(mse(mean, truth)),
        "rmse": float(rmse(mean, truth)),
        "log_mse": float(log_mse(mean, truth)),
        "energy_score": float(energy_score(samples, truth)),
        f"coverage_{interval:g}": float(coverage(samples, truth, mass=interval)),
        "calibration_error": float(calibration_error(samples, truth)),
        "spread": float(samples.flatten(2).std(1).mean()),
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
}


def get(name: str) -> Callable[..., torch.Tensor]:
    """Resolve a built-in metric by its stable short name."""
    if name not in METRICS:
        raise KeyError(f"Unknown metric {name!r}; available: {sorted(METRICS)}")
    return METRICS[name]
