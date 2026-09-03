"""Convert PyBED data between Torch, NumPy, and optional JAX arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .core import Batch, ParticleCloud, PolicySample


def convert(value: Any, backend: str = "torch", *, device: str | torch.device = "cpu") -> Any:
    """Take arrays or PyBED containers and return them in the requested array backend.

    DataLoaders remain standard PyTorch loaders.  A JAX method can convert each
    yielded batch here (DLPack is used when both frameworks support it) while a
    custom pure-JAX simulator can still be attached to ``BED`` as an ordinary
    callable.  PyBED does not silently translate numerical solvers between array
    frameworks because doing so would compromise reproducibility.
    """
    if isinstance(value, Batch):
        fields: dict[str, Any] = {}
        for name in ("theta", "x", "y", "mask", "target", "belief"):
            item = getattr(value, name)
            fields[name] = None if item is None else convert(item, backend, device=device)
        fields["context"] = convert(value.context, backend, device=device)
        return Batch(**fields, meta=value.meta)
    if isinstance(value, ParticleCloud):
        return ParticleCloud(
            convert(value.points, backend, device=device),
            None if value.weights is None else convert(value.weights, backend, device=device),
            None if value.mask is None else convert(value.mask, backend, device=device),
            value.particle_dim,
        )
    if isinstance(value, PolicySample):
        return PolicySample(
            convert(value.x, backend, device=device),
            None if value.log_prob is None else convert(value.log_prob, backend, device=device),
            None if value.entropy is None else convert(value.entropy, backend, device=device),
        )
    if isinstance(value, dict):
        return {key: convert(item, backend, device=device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(convert(item, backend, device=device) for item in value)
    if isinstance(value, list):
        return [convert(item, backend, device=device) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if backend == "torch":
        return torch.as_tensor(value, device=device)
    if backend == "numpy":
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if backend == "jax":
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError("Install the optional backend with `pip install pybed[jax]`") from error
        if torch.is_tensor(value):
            tensor = value.detach().contiguous()
            try:
                return jax.dlpack.from_dlpack(tensor)
            except (BufferError, RuntimeError, TypeError):
                return jnp.asarray(tensor.cpu().numpy())
        return jnp.asarray(value)
    raise ValueError("backend must be 'torch', 'numpy', or 'jax'")
