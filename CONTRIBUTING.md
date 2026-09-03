# Contributing to PyBED

PyBED aims to make BED comparisons easier to audit. Contributions should keep scientific choices visible and avoid coupling environments to a particular neural architecture or trainer.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check pybed examples tests
MPLBACKEND=Agg pytest
python -m build
```

## Adding an environment

1. Put reusable numerical code in `pybed/sim.py`.
2. Add a configurable factory in `pybed/envs.py`.
3. Declare event shapes and design bounds with `Spec`, using `x` for designs and `y` for outcomes.
4. Register a versioned name such as `problem-v0`.
5. Test batched shapes, explicit-seed reproducibility, declared gradients, and serialization.
6. Add a short table row and scientific caveat to the README.

Do not hide paper-specific constants in a model or dataloader. Put them in factory arguments and `env.cfg`. A paper-exact preset should cite the source and distinguish the exact elements from numerical approximations.

## Adding a metric

Metrics belong in `pybed/metrics.py` and should document tensor axes, reduction, direction (higher or lower is better), and estimator assumptions. Add it to `METRICS` only when the name and behavior are stable. Information estimators must state whether they require a tractable likelihood.

## Pull requests

Keep changes focused, include tests, and update the changelog. New third-party dependencies should be optional unless every environment needs them. Never commit generated run folders, datasets, credentials, or private tracking links.

Keep docstrings short and concrete: say what a function takes and what it returns. Prefer ordinary research language over framework terminology.
