# PyBED

**Composable, extensible, reproducible environments for Bayesian experimental design.**

PyBED gives BED and simulation-based inference methods one small shared interface for collecting data, training posterior estimators or design policies, and comparing them on exactly the same latent parameters. It is intentionally a toolkit rather than a training framework: PyTorch, JAX, Lightning, a custom loop, or no model at all are equally valid consumers.

> Status: research-quality alpha. The contracts are tested; publication-scale reference numbers for every paper method are not bundled in v0.1.

## Why PyBED?

Deep Adaptive Design (DAD), implicit DAD, ALINE, JADAI, PSST-style particle transport, and classical BED share the same objects but usually reimplement them: a prior, designs, observations, histories, simulators, policy rollouts, metrics, and experiment logging. Small discrepancies in these layers can dominate a comparison.

PyBED standardizes those shared layers and leaves the scientific method replaceable.

```python
import pybed as pb

env = pb.make("location-v0", sources=2, dims=2, budget=30)
print(env)

episodes = pb.data.EpochStream(env, episodes=4096, seed=2030).generate(policy=None)
for batch in pb.data.loader(episodes, batch_size=128, seed=2030, epoch=0):
    posterior = your_model.infer(batch)
```

The core verbs are direct:

```python
y = env.simulate(theta, design, seed=7)
x_next = env.design(history)          # when a policy is attached
posterior = env.infer(history)        # when an inference component is attached
y_pred = env.predict(query, history)  # when a predictor is attached

custom = env.with_components(
    simulate=my_simulator,
    design=my_policy.design,
    infer=my_posterior.infer,
    predict=my_predictor.predict,
)
```

No subclass is required. Components are ordinary callables; PyBED passes only the context their signatures accept.

## Install

```bash
pip install -e .

# Optional integrations
pip install -e '.[mnist]'
pip install -e '.[jax]'

# Contributor setup
pip install -e '.[dev]'
pytest
```

The base dependency set is NumPy, SciPy, PyTorch, Matplotlib, and seaborn. MNIST loading and JAX conversion are opt-in.

## Environments

| Environment | Parameter | Design | Observation | Default budget |
|---|---|---|---|---:|
| `location-v0` | one or more source positions | continuous sensor location | noisy log inverse-square intensity | 30 |
| `mnist-discovery-v0` | a `C x 28 x 28` image | continuous mask center | smooth masked noisy image | 6 |
| `advdiff-v0` | initial spatial field | continuous sensor location(s) | sensor time series | 8 |
| `pendulum-v0` | gravity and damping | initial angle and velocity | stochastic angle time series | 6 |
| `ces-v0` | CES preference parameters | two baskets of goods | noisy preference | 20 |
| `death-v0` | infection/death rate | observation time | affected population count | 4 |

Built-ins use batched PyTorch and explicit random generators. `pybed.sim` also exposes the inverse-square model, a reusable differentiable advection-diffusion-reaction solver, Euler-Maruyama integration, image masks, CES, and death-process simulators directly.

Adding an environment is a factory plus a stable registry name:

```python
from pybed import BED, Spec, Uniform, register

def oscillator(**cfg):
    env = BED(
        name="oscillator-v0",
        prior=Uniform([-1.0, 0.1], [1.0, 2.0]),
        sim=cfg["sim"],
        specs={
            "theta": Spec((2,)),
            "design": Spec((1,), low=0.0, high=10.0),
            "obs": Spec((1,)),
        },
        budget=10,
        cfg=cfg,
    )
    return env

register("oscillator-v0", oscillator)
```

## Histories, masks, and changing targets

`pb.Batch` is the interchange object:

```python
pb.Batch(
    theta=theta,       # (B, ...parameter event shape)
    design=x,          # (B, T, ...design event shape)
    obs=y,             # (B, T, ...observation event shape)
    mask=valid,        # (B, T), for padding / variable horizons
    target=target,     # parameter mask, prediction target, label, image, ...
    context={"time": physical_time},
)
```

This is sufficient for ALINE-like attention masks and target subsets without teaching the environment about a particular Transformer. A non-stationary prior is simply a callable component that returns the prior for the received `epoch`, `time`, or other context. Parameter projections can likewise live in `target` or in an attached posterior.

## Reproducible online epochs

Synthetic training does not naturally have epochs. PyBED defines an online epoch as a finite materialized sequence of `episodes` latent truths:

```python
stream = pb.data.EpochStream(env, episodes=8192, seed=42)

learned = stream.generate(policy, epoch=3)
random = stream.generate(None, epoch=3)
assert (learned.batch.theta == random.batch.theta).all()
```

The seed tree derives independent streams from `(root seed, namespace, epoch, step)`. Consequently:

- every policy receives the same ordered `theta*` values within an epoch;
- policies can share simulator noise (common random numbers) without sharing actions;
- changing dataloader iteration order does not change the generated experiment;
- a generated epoch can be saved and reloaded as a finite posterior-training dataset;
- a full pass over a saved dataset has the ordinary, unambiguous meaning of an epoch.

```python
stream.generate(policy, epoch=3, save="data/epoch-0003.pt")
offline = pb.data.load("data/epoch-0003.pt")

for epoch in range(100):
    for batch in pb.data.loader(offline, 256, seed=42, epoch=epoch):
        train_posterior(batch)
```

The `.pt.json` sidecar records format, tensor shapes, and provenance metadata.

## Point-cloud priors and other frameworks

PSST-style models can supply particles directly:

```python
prior = pb.EmpiricalPrior(points=cloud, weights=weights, jitter=0.01)
env = pb.make("location-v0", prior=prior)
```

JAX methods can consume ordinary Torch data loaders:

```python
for batch in pb.data.loader(dataset, 128):
    jax_batch = pb.backends.convert(batch, "jax")
    update(jax_batch)
```

Numerical solvers are not silently translated across frameworks: built-ins are PyTorch, and a user may attach a pure NumPy/JAX simulator using `with_components(simulate=...)`. This makes backend choice explicit and avoids claiming identical numerics where they do not exist.

## Metrics and comparison

`pybed.metrics` provides MSE, RMSE, log-MSE, negative log score, energy score, interval coverage, simulation-based calibration error, permutation-invariant source-set MSE, Monte Carlo EIG, and sequential prior contrastive estimation (sPCE).

```python
def score(env, batch):
    samples = posterior.sample(batch)
    return pb.metrics.posterior_summary(samples, batch.theta)

results, trajectories = pb.exp.compare(
    env,
    {"DAD": dad.design, "PSST": psst.design, "random": None},
    score,
    episodes=1024,
    seed_value=2030,
)

fig, ax = pb.vs.benchmark(results, metric="mse")
```

Likelihood-based EIG and likelihood-free posterior scores remain separate because not every simulator exposes a tractable likelihood. `spce` accepts the accumulated history log-likelihood under one true parameter and its prior contrastives; it does not infer a likelihood from simulator samples.

## Experiments without a tracking service

`pb.exp.Run` creates a self-contained folder with:

- normalized JSON configuration and environment metadata;
- append-only JSONL metrics;
- model, optimizer, and Python/NumPy/Torch RNG checkpoints;
- figures and reusable datasets;
- a final summary.

`pb.exp.train` accepts any sequence of loaders, so offline data and freshly generated finite epochs share one loop. Nothing prevents using Lightning or another runner instead.

## Examples

The examples are notebook-style Python files with `#%%` cells and no `main()` wrapper.

```bash
python examples/random_location.py
python examples/actionbed/actionbed.py

# fast CI/smoke version
PYBED_QUICK=1 MPLBACKEND=Agg python examples/actionbed/actionbed.py
```

`examples/actionbed/actionbed.py` is the complete ActionBED example in one file. It first visualizes the environment, then trains a masked history encoder, Gaussian posterior, and reparameterized continuous action policy jointly through the differentiable location simulator. It compares ActionBED with random design on shared truths/noise and saves camera-ready trajectories, losses (log scale), datasets, and checkpoints.

## Module map

| Module | Responsibility |
|---|---|
| `core` | `BED`, `Batch`, `Spec`, priors, registry, component replacement |
| `sim` | batched simulators, PDE and SDE numerics |
| `envs` | configurable benchmark factories |
| `data` | seed tree, finite synthetic epochs, save/load, deterministic loaders |
| `metrics` | posterior, predictive, and design scores |
| `vs` | publication-style visual diagnostics |
| `exp` | runs, JSONL logs, checkpoints, generic training and comparison |
| `backends` | explicit Torch/NumPy/JAX boundary conversion |

## Scope and scientific cautions

- The PDE reference environment implements the same initial-condition-to-sensor-time-series abstraction as Alexanderian et al. (2014), using a compact differentiable grid solver rather than reproducing their finite-element geometry and Navier-Stokes velocity field.
- `mnist-discovery-v0` implements the smooth-mask image-discovery task. Pass official MNIST arrays or install the optional loader; the package never downloads data unexpectedly.
- The included ActionBED model is an executable integration example and baseline, not a claimed reproduction of DAD, iDAD, ALINE, JADAI, or PSST.
- Exact paper reproduction still requires the original architecture, objective, hyperparameters, seeds, and reporting protocol. PyBED supplies the shared environment and evaluation layer so those differences stay visible.

## References

- Foster et al., [Deep Adaptive Design](https://proceedings.mlr.press/v139/foster21a.html), ICML 2021.
- Ivanova et al., [Implicit Deep Adaptive Design](https://proceedings.neurips.cc/paper/2021/hash/357a6fdf7642bf815a88822c447d9dc4-Abstract.html), NeurIPS 2021.
- Huang et al., [ALINE](https://arxiv.org/abs/2506.07259), 2025.
- Bracher et al., [JADAI](https://arxiv.org/abs/2512.22999), 2025.
- Alexanderian et al., [A-optimal design of experiments for infinite-dimensional Bayesian linear inverse problems with regularized l0-sparsification](https://doi.org/10.1137/130933381), SIAM J. Sci. Comput. 2014.

## License

MIT. See [LICENSE](LICENSE).

