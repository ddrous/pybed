# PyBED

**Small, composable, reproducible building blocks for Bayesian experimental design.**

PyBED supplies environments, simulators, observation batches, reusable datasets, metrics, and local experiment records. Models and losses remain ordinary PyTorch, NumPy, or JAX code. DAD, iDAD, ALINE, JADAI, Action-BED, particle transports, and ordinary inverse problems can therefore share the same experimental layer without sharing a training framework.

> Status: research-quality alpha. Version 0.2 makes a deliberate notation change described below.

## Install

```bash
pip install -e .

# Optional data and framework integrations
pip install -e '.[mnist]'
pip install -e '.[jax]'
pip install -e '.[lightning]'  # Lightning and W&B

# Contributor setup
pip install -e '.[dev]'
```

The base package does not import Lightning, W&B, torchvision, or JAX.

## The notation

At step $t$, the design is $x_t$, its measured outcome is $y_t$, and the complete observation is

$$
o_t=(x_t,y_t).
$$

A batch uses the same names:

```python
batch = pb.Batch(theta=theta, x=x, y=y, mask=valid)

batch.x          # designs
batch.design     # alias for x
batch.y          # outcomes
batch.outcome    # alias for y
batch.o          # Observation(x, y)
batch.obs        # alias for the complete Observation pair
```

For a non-design inverse problem, `x` is `None`, so `batch.o` is `Observation(None, y)`. In version 0.1, `Batch.obs` meant only the outcome. It now means the complete pair; use `Batch.y` or `Batch.outcome` for the measured value.

## First experiment

```python
import pybed as pb

env = pb.make("location-v0", sources=2, dims=2, budget=10)
data = pb.data.Stream(env, episodes=4096, seed=2030).generate()

for batch in pb.data.loader(data, batch_size=128, seed=2030):
    result = your_model(batch)
```

The direct environment verbs are:

```python
y = env.simulate(theta, x, seed=7)
x_next = env.design(history)
inference = env.infer(history)
prediction = env.predict(query, history)
```

An inference result may be a point, distribution, categorical logits, posterior samples, or a `ParticleCloud`.

## Models with shared inference, design, and prediction

Models such as ALINE and PSST can share one backbone and produce several outputs in one pass:

```python
method = env.with_components(
    infer_design=model.infer_design,
    design_predict=model.design_predict,
    design_infer_predict=model.design_infer_predict,
)

inference, x = method.infer_design(history)
x, prediction = method.design_predict(history)
x, inference, prediction = method.design_infer_predict(history)
```

The verb order is also the return order. A model only needs to supply the combined calls it truly computes jointly. An all-in-one `design_infer_predict` model also serves the two-output calls by returning the requested members. Otherwise, PyBED composes the separate `design`, `infer`, and `predict` components.

During `Stream.generate()`, an attached joint component is called once per adaptive step. A returned `ParticleCloud` becomes the next step's `history.belief`.

## Environments

| Environment | Parameter | Design (x) | Outcome (y) | Budget |
|---|---|---|---|---:|
| `location-v0` | one or more source locations | sensor location | noisy log intensity | 30 |
| `mnist-discovery-v0` | image | smooth mask centre | masked image | 6 |
| `mnist-classification-v0` | image; label is target | continuous patch corner | noisy 5 × 5 patch | 5 |
| `advdiff-v0` | initial field | sensor location(s) | sensor time series | 8 |
| `pendulum-v0` | gravity and damping | initial state | angle time series | 6 |
| `ces-v0` | preference parameters | two baskets | noisy preference | 20 |
| `death-v0` | event rate | observation time | affected count | 4 |

Factories accept numerical settings directly. Components remain replaceable:

```python
custom = env.with_components(
    simulate=my_simulator,
    design=my_policy,
    infer=my_inference,
    predict=my_predictor,
    log_prob=my_likelihood,
    candidates=my_candidate_pool,
)
```

No subclass is needed. PyBED passes a callable only the named context values that it accepts.

## Different source counts and dimensions

Location finding can mix padded shapes within one training batch:

```python
env = pb.make(
    "location-v0",
    sources=(1, 2, 4),
    dims=(1, 2, 3),
)
batch = pb.data.Stream(env, 256, seed=7).generate().batch

batch.context["num_sources"]
batch.context["source_dim"]
batch.context["source_mask"]
batch.context["theta_mask"]
```

Inactive sources and coordinates do not contribute to the simulator. The context is indexed, collated, moved between devices, converted to NumPy/JAX, and saved with the data.

A custom prior may provide the same pattern by returning `Batch(theta=..., context=...)`.

## Particle beliefs

```python
cloud = pb.ParticleCloud(
    points=particles,      # (batch, particles, ...parameter shape)
    weights=weights,       # optional, (batch, particles)
    mask=theta_mask,       # optional
)

mean = cloud.mean()
ess = cloud.effective_sample_size()
resampled = cloud.sample(128)
```

`EmpiricalPrior` resamples an unbatched cloud. `Batch.belief` instead carries a changing per-episode cloud through an adaptive rollout. These are separate ideas: one describes how truths are initially sampled, while the other describes the method's current belief.

For MNIST classification, particles may be integer labels or one-hot vectors of length ten:

```python
probabilities = pb.metrics.class_probabilities(class_cloud)
```

The normalized weighted counts are returned directly. PyBED does not sharpen or recalibrate them.

## Candidate designs and score-function policies

A candidate pool may be shared, batched, or generated from the current history:

```python
method = env.with_components(candidates=candidate_tensor)

def policy(history, candidates):
    distribution = torch.distributions.Categorical(logits=model(history, candidates))
    index = distribution.sample()
    x = candidates[index]
    return pb.PolicySample(
        x=x,
        log_prob=distribution.log_prob(index),
        entropy=distribution.entropy(),
    )
```

Continuous deterministic policies may keep returning an array. PyBED retains `PolicySample.log_prob` in the rollout context so method-specific REINFORCE losses can use it. Baselines, credit assignment, and reward definitions remain in the method.

## Reusing generated work

```python
buffer = pb.data.ReplayBuffer(capacity=20_000)
buffer.add(batch)
replayed = buffer.sample(256, seed=3)
```

The buffer accepts complete detached batches, including masks, nested context, targets, and particle beliefs. It is for recent model-generated work. `EpisodeDataset` is for a fixed, serializable dataset:

```python
data.save("data/train.pt")
same_data = pb.data.load("data/train.pt")
```

Version 2 files store `theta`, `x`, `y`, masks, targets, beliefs, context, and metadata. Version 1 files remain readable.

## Adaptive and non-adaptive simulation

Adaptive policies are simulated step by step. With no policy, PyBED prepares the full design tensor and calls a vectorized simulator once:

```python
random_data = pb.data.Stream(env, 4096, seed=3).generate()
fixed_data = pb.data.Stream(env, 4096, seed=3).generate(x=fixed_x)
```

`fixed_x` may be shared across episodes or supplied per episode. A custom simulator that cannot handle a leading time axis can set `vectorized=False` on its `BED` object.

## Ordinary inverse problems

`pybed.envs.inverse` creates a problem with no design variable:

```python
env = pb.envs.inverse(
    prior=prior,
    simulator=forward_model,
    theta_shape=(64, 64),
    y_shape=(20,),
)
pairs = pb.data.Stream(env, 8192, seed=4).generate()
assert pairs.batch.x is None

posterior = env.with_components(infer=model).infer(pairs.batch.o)
```

An externally supplied joint dataset is just as direct:

```python
pairs = pb.data.EpisodeDataset(pb.Batch(theta=u, y=o))
```

This matches the i.i.d. joint-pair assumption in Baptista, Kaveh, and Stuart's energy-based transport method. A transport produces shape `(batch, particles, ...)`, then trains with:

```python
loss = pb.metrics.energy_score(posterior_particles, batch.theta, unbiased=True)
```

The helper is differentiable, accepts weighted `ParticleCloud` objects, and can exclude self-pairs for the empirical U-statistic. Function-valued parameters remain ordinary trailing tensor axes. Cameron--Martin covariance maps and neural operators belong in the transport model because they depend on the chosen prior and discretization.

Run the complete small example with:

```bash
python examples/inverse/energy_transport.py
```

## MNIST classification

`mnist-classification-v0` follows the masked classification problem in Action-BED:

- the latent parameter is the image;
- the target is its class label;
- each design chooses a continuous patch corner;
- bilinear sampling returns a noisy 5 × 5 patch;
- the default horizon is five.

Images and labels can be passed directly. If omitted, torchvision is loaded only when the factory is called, and data are downloaded only with `download=True`.

Cross entropy and classification accuracy are available in `pybed.metrics`. ALINE may return categorical logits from its inference head; Action-BED may return logits from its downstream action network; particle methods may return a class cloud.

## DAD with Lightning and W&B

The DAD example trains a non-myopic location policy with sPCE, Lightning, and W&B:

```bash
pip install -e '.[lightning]'

# Local W&B files; no upload
WANDB_MODE=offline python examples/dad/dad_lightning_wandb.py

# Send the run to W&B
WANDB_MODE=online WANDB_PROJECT=my-project \
  python examples/dad/dad_lightning_wandb.py

# Four-GPU DDP
PYBED_DEVICES=4 WANDB_MODE=online \
  python examples/dad/dad_lightning_wandb.py

# Short smoke run
PYBED_QUICK=1 WANDB_MODE=disabled \
  python examples/dad/dad_lightning_wandb.py
```

Lightning owns optimization, checkpointing, accelerator selection, and DDP. Its `WandbLogger` records the sPCE values; the example also logs an evaluation trajectory. Neither framework enters PyBED's core.

For a plain PyTorch loop, `pb.exp.Run.log()` returns the dictionary it writes locally, so the same row can also be passed to `wandb_run.log(row)`.

## Method map

- **DAD:** attach a policy and use `env.log_prob(y, theta, x)` to form sPCE.
- **iDAD:** use the same histories with a method-owned InfoNCE or NWJ critic; no likelihood is needed.
- **ALINE:** provide candidates and a joint call such as `infer_design`; inference may be a point, GMM, predictive distribution, or categorical logits.
- **JADAI:** attach joint design and posterior components and differentiate through the simulator.
- **Action-BED:** train the design and downstream action together using `Batch.target`.
- **PSST:** place the current `ParticleCloud` in `Batch.belief`, return an updated cloud from a joint call, use masks for changing parameter shapes, and optionally reuse complete records with `ReplayBuffer`.
- **Energy transport:** train posterior particles from paired `theta,y` data with the energy score.

## Metrics and experiments

PyBED includes MSE, RMSE, log-MSE, negative log score, weighted energy score, interval coverage, calibration error, permutation-invariant source-set MSE, cross entropy, classification accuracy, Monte Carlo EIG, and sPCE.

`pb.exp.Run` creates plain local folders containing JSON configuration, JSONL metrics, checkpoints, figures, data, and a final summary. It has no service dependency.

## Module map

| Module | Contents |
|---|---|
| `core` | `BED`, `Batch`, observations, particles, policy samples, priors, registry |
| `sim` | batched differentiable simulators and numerical solvers |
| `envs` | configurable benchmark factories |
| `data` | `Stream`, `Seeds`, datasets, loaders, replay |
| `metrics` | posterior, design, calibration, and classification scores |
| `vs` | visual checks and paper figures |
| `exp` | local runs, checkpoints, training, policy comparison |
| `backends` | explicit Torch, NumPy, and optional JAX conversion |

## Scientific cautions

- Built-in environments are compact shared benchmarks, not guaranteed reproductions of every paper's numerical setup.
- The PDE environment uses a differentiable grid solver rather than reproducing a particular finite-element mesh.
- Exact paper reproduction still requires the original architecture, objective, optimizer, seeds, and reporting protocol.
- The examples favor readable research code over a large training framework inside PyBED.

## References

- Foster et al., [Deep Adaptive Design](https://proceedings.mlr.press/v139/foster21a.html), 2021.
- Ivanova et al., [Implicit Deep Adaptive Design](https://proceedings.neurips.cc/paper/2021/hash/357a6fdf7642bf815a88822c447d9dc4-Abstract.html), 2021.
- Huang et al., [ALINE](https://arxiv.org/abs/2506.07259), 2025.
- Bracher et al., [JADAI](https://arxiv.org/abs/2512.22999), 2025.
- Rossa, Phillips, and Rainforth, *Task-Driven Bayesian Experimental Design with Singly Intractable Objectives*, 2026.
- Baptista, Kaveh, and Stuart, *Energy-Based Transport for Amortized Bayesian Inference*, 2026.

## License

MIT. See [LICENSE](LICENSE).
