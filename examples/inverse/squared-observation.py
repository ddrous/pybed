"""Learn a likelihood-free posterior transport from paired inverse-problem data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

import pybed as pb

DEBUG = False

@dataclass(frozen=True)
class Config:
    """Example settings."""
    seed: int = 2030
    pairs: int = 8_192
    batch_size: int = 256
    particles: int = 32
    epochs: int = 200
    learning_rate: float = 3e-4
    runs: str = "runs"
    checkpoint: str | None = None


class Transport(nn.Module):
    """Move reference-prior samples according to the observed inverse data."""

    def __init__(self):
        """Create the small residual transport network and return no separate value."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, z: torch.Tensor, o: pb.Observation) -> torch.Tensor:
        """Take reference particles and observations and return transported posterior particles."""
        inputs = torch.cat((z, o.outcome[:, None, :].expand_as(z)), -1)
        return z + self.net(inputs)


# %% Define the inverse problem
cfg = Config()
if DEBUG:
    cfg = Config(pairs=128, batch_size=32, particles=4, epochs=2)
pb.expt.seed(cfg.seed)


def simulator(theta: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Take scalar parameters and return noisy squared observations."""
    noise = 0.5 * torch.randn(theta.shape, generator=generator, device=theta.device)
    return theta.square() + noise


env = pb.envs.inverse(
    prior=pb.Normal(torch.zeros(1), torch.ones(1)),
    simulator=simulator,
    theta_shape=(1,),
    y_shape=(1,),
    name="squared-observation-v0",
)
run = pb.expt.Run.create(
    cfg.runs,
    "energy-transport",
    cfg,
    seed_value=cfg.seed,
    source_files=[__file__],
)


# %% Generate paired data and train the conditional transport
pairs = pb.data.Stream(env, cfg.pairs, seed=cfg.seed).generate()
model = Transport()
if cfg.checkpoint:
    saved = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("model", saved))
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
seeds = pb.data.Seeds(cfg.seed)
for epoch in range(cfg.epochs):
    for step, batch in enumerate(pb.data.loader(pairs, cfg.batch_size, seed=cfg.seed, epoch=epoch)):
        z = env.sample_prior(
            len(batch) * cfg.particles,
            generator=seeds.torch("reference", epoch, step),
        ).reshape(len(batch), cfg.particles, 1)
        particles = model(z, batch.obs)
        loss = pb.metrics.energy_score(particles, batch.parameters, unbiased=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
run.checkpoint("last", model=model, optimizer=optimizer, step=cfg.epochs)


# %% Draw and visualize a posterior for a new observation
observation = pb.Observation(None, torch.tensor([[4.0]]))
z = env.sample_prior(1_000, seed=cfg.seed + 1).reshape(1, 1_000, 1)
posterior = pb.ParticleCloud(model(z, observation).detach())
figure, _ = pb.vis.particles(posterior)
run.save_figure(figure, "posterior")
summary = {
    "observation": float(observation.outcome.item()),
    "posterior_mean": float(posterior.mean().item()),
}
run.finish(**summary)
print(summary)
