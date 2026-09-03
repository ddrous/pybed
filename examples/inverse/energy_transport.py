"""Learn a likelihood-free posterior transport from paired inverse-problem data."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn

import pybed as pb


QUICK = os.getenv("PYBED_QUICK", "0") == "1"


@dataclass(frozen=True)
class Config:
    """Hold the example settings and produce no output on its own."""

    seed: int = 2030
    pairs: int = 128 if QUICK else 8_192
    batch_size: int = 32 if QUICK else 256
    particles: int = 4 if QUICK else 32
    epochs: int = 2 if QUICK else 200
    learning_rate: float = 3e-4


class Transport(nn.Module):
    """Move reference-prior samples according to the observed inverse data."""

    def __init__(self):
        """Create the small residual transport network and return no separate value."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, z: torch.Tensor, o: pb.Observation) -> torch.Tensor:
        """Take reference particles and observations and return transported posterior particles."""
        inputs = torch.cat((z, o.y[:, None, :].expand_as(z)), -1)
        return z + self.net(inputs)


def main() -> None:
    """Generate joint pairs, train the transport, print a posterior cloud, and return nothing."""
    cfg = Config()
    pb.exp.seed(cfg.seed)

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
    pairs = pb.data.Stream(env, cfg.pairs, seed=cfg.seed).generate()
    model = Transport()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    seeds = pb.data.Seeds(cfg.seed)
    for epoch in range(cfg.epochs):
        for step, batch in enumerate(pb.data.loader(pairs, cfg.batch_size, seed=cfg.seed, epoch=epoch)):
            z = env.sample_prior(
                len(batch) * cfg.particles,
                generator=seeds.torch("reference", epoch, step),
            ).reshape(len(batch), cfg.particles, 1)
            particles = model(z, batch.o)
            loss = pb.metrics.energy_score(particles, batch.theta, unbiased=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    o = pb.Observation(None, torch.tensor([[4.0]]))
    z = env.sample_prior(1_000, seed=cfg.seed + 1).reshape(1, 1_000, 1)
    posterior = pb.ParticleCloud(model(z, o).detach())
    print({"observation": float(o.y.item()), "posterior_mean": float(posterior.mean().item())})


if __name__ == "__main__":
    main()
