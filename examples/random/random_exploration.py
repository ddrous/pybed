# %% Imports and camera-ready style
"""Explore location finding with a random policy; no model or training required."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

import pybed as pb

pb.vs.style()
OUT = Path("runs/random-location")
OUT.mkdir(parents=True, exist_ok=True)


# %% Construct and inspect the environment
env = pb.make("location-v0", sources=2, dims=2, budget=12, noise=0.25)
print(env)

fig, ax = env.visualize("prior", n=2_000, seed=7)
fig.savefig(OUT / "prior.png", dpi=180, bbox_inches="tight")


# %% Generate one reproducible random-policy epoch
stream = pb.data.Stream(env, episodes=32, budget=12, seed=2030)
episodes = stream.generate(None, epoch=0, save=OUT / "random-episodes.pt")
print(episodes.batch.theta.shape, episodes.batch.x.shape, episodes.batch.y.shape)

fig, ax = env.visualize("episode", episodes.batch, index=0)
fig.savefig(OUT / "random-trajectory.png", dpi=180, bbox_inches="tight")


# %% Verify the common-truth contract that policy comparisons rely on
repeated = stream.generate(None, epoch=0)
assert torch.equal(episodes.batch.theta, repeated.batch.theta)
assert torch.equal(episodes.batch.x, repeated.batch.x)
assert torch.equal(episodes.batch.y, repeated.batch.y)

plt.show()
