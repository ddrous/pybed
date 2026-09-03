"""Array conversion at the framework boundary, including optional JAX support."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .core import Batch


def convert(value: Any, backend: str = "torch", *, device: str | torch.device = "cpu") -> Any:
    """Convert arrays or a ``Batch`` to torch, NumPy, or optional JAX arrays.

    DataLoaders remain standard PyTorch loaders.  A JAX method can convert each
    yielded batch here (DLPack is used when both frameworks support it) while a
    custom pure-JAX simulator can still be attached to ``BED`` as an ordinary
    callable.  PyBED does not silently translate numerical solvers between array
    frameworks because doing so would compromise reproducibility.
    """
    if isinstance(value, Batch):
        fields = {}
        for name in ("theta", "design", "obs", "mask", "target"):
            item = getattr(value, name)
            fields[name] = None if item is None else convert(item, backend, device=device)
        return Batch(**fields, context=value.context, meta=value.meta)
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
