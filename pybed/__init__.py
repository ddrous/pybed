"""PyBED: small, composable building blocks for Bayesian experimental design.

The public surface is intentionally narrow.  Most projects need only ``make`` or
``BED`` plus the ``data``, ``exp``, ``metrics``, ``sim`` and ``vs`` modules.
"""

from . import backends, data, envs, exp, metrics, sim, vs
from .core import BED, Batch, EmpiricalPrior, Normal, Spec, Uniform, available, make, register

__all__ = [
    "BED",
    "BEDEnv",
    "BEDEnvironment",
    "Batch",
    "EmpiricalPrior",
    "Normal",
    "Spec",
    "Uniform",
    "available",
    "make",
    "register",
    "backends",
    "data",
    "envs",
    "exp",
    "metrics",
    "sim",
    "vs",
    "visualisers",
]

__version__ = "0.1.0"

# Friendly compatibility aliases; ``BED`` and ``vs`` remain the canonical short names.
BEDEnv = BED
BEDEnvironment = BED
visualisers = vs
