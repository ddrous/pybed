"""PyBED: small, composable building blocks for Bayesian experimental design.

The public surface is intentionally narrow. Most projects need only ``make`` or
``BED`` plus the ``data``, ``expt``, ``metrics``, ``sim`` and ``vis`` modules.
"""

from . import backends, data, envs, expt, metrics, sim, vis
from .core import (
    BED,
    Batch,
    EmpiricalPrior,
    Normal,
    Observation,
    ParticleCloud,
    Particles,
    PolicySample,
    Spec,
    Uniform,
    environments,
    make,
)

__all__ = [
    "BED",
    "Batch",
    "EmpiricalPrior",
    "Normal",
    "Observation",
    "ParticleCloud",
    "Particles",
    "PolicySample",
    "Spec",
    "Uniform",
    "environments",
    "make",
    "backends",
    "data",
    "envs",
    "expt",
    "metrics",
    "sim",
    "vis",
]

__version__ = "0.2.0"
