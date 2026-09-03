# %% Imports and camera-ready style
"""Explore location finding with a random policy; no model or training required."""
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch

import pybed as pb

DEBUG = False

@dataclass(frozen=True)
class Config:
    """Environment, sampling, and output settings."""
    seed: int = 2030
    sources: int = 2
    dims: int = 2
    budget: int = 12
    episodes: int = 32
    noise: float = 0.25
    runs: str = "runs"


# %% Start a recorded run
cfg = Config()
if DEBUG:
    cfg = Config(budget=3, episodes=4)
pb.vis.style()
run = pb.expt.Run.create(
    cfg.runs,
    "random-location",
    cfg,
    seed_value=cfg.seed,
    source_files=[__file__],
)


# %% Construct and inspect the environment
env = pb.make(
    "location-v0",
    sources=cfg.sources,
    dims=cfg.dims,
    budget=cfg.budget,
    noise=cfg.noise,
)
print(env)

fig, ax = env.visualize("prior", n=2_000, seed=7)
run.save_figure(fig, "prior")


# %% Generate one reproducible random-policy epoch
stream = pb.data.Stream(env, episodes=cfg.episodes, budget=cfg.budget, seed=cfg.seed)
episodes = stream.generate(None, epoch=0)
print(episodes.batch.parameters.shape, episodes.batch.designs.shape, episodes.batch.outcomes.shape)
run.save_data(episodes, "random-episodes")

fig, ax = env.visualize("episode", episodes.batch, index=0)
run.save_figure(fig, "random-trajectory")


# %% Verify the common-truth contract that policy comparisons rely on
repeated = stream.generate(None, epoch=0)
assert torch.equal(episodes.batch.parameters, repeated.batch.parameters)
assert torch.equal(episodes.batch.designs, repeated.batch.designs)
assert torch.equal(episodes.batch.outcomes, repeated.batch.outcomes)

run.finish(episodes=cfg.episodes)
plt.show()
