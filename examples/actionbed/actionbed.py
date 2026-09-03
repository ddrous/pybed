# %% Imports, configuration, and experiment conventions
"""ActionBED: a compact joint design-and-inference example built entirely on PyBED.

This is intentionally a single notebook-style file with no ``main()``.  It trains
both a reparameterized continuous design policy (the action model) and a Gaussian
posterior estimator, evaluates against a common-random-number random policy, and
writes transparent run artifacts without Weights & Biases.

Set ``PYBED_QUICK=1`` for the smoke-test configuration used in CI.
"""

import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import torch
from torch import nn

import pybed as pb


@dataclass(frozen=True)
class Config:
    seed: int = 2030
    sources: int = 1
    dims: int = 2
    budget: int = 8
    epochs: int = 60
    steps_per_epoch: int = 16
    batch_size: int = 128
    hidden: int = 128
    learning_rate: float = 3e-4
    min_std: float = 0.03
    entropy_weight: float = 1e-3
    eval_episodes: int = 256
    runs: str = "runs"


cfg = Config()
if os.getenv("PYBED_QUICK") == "1":
    cfg = Config(epochs=2, steps_per_epoch=2, batch_size=16, budget=3, eval_episodes=16, hidden=32)

pb.exp.seed(cfg.seed)
pb.vs.style()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Environment intuition comes first
env = pb.make(
    "location-v0",
    sources=cfg.sources,
    dims=cfg.dims,
    budget=cfg.budget,
    low=0.0,
    high=1.0,
    noise=0.35,
)
print(env)

preview = pb.data.EpochStream(env, episodes=16, seed=cfg.seed, budget=cfg.budget).generate(None)
preview_fig, _ = env.visualize("episode", preview.batch, index=0)


# %% One history encoder, one posterior head, and one stochastic action head
class HistoryEncoder(nn.Module):
    """Masked DeepSets encoder; padding never leaks into a history representation."""

    def __init__(self, design_dim: int, obs_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.empty = nn.Parameter(torch.zeros(hidden))
        self.tokens = nn.Sequential(
            nn.Linear(design_dim + obs_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mix = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))

    def forward(self, history: pb.Batch) -> torch.Tensor:
        batch = len(history)
        if history.design is None or history.obs is None or history.design.shape[1] == 0:
            return self.empty.expand(batch, -1)
        steps = history.design.shape[1]
        time = torch.linspace(1 / max(1, steps), 1, steps, device=history.design.device)
        time = time.reshape(1, steps, 1).expand(batch, -1, -1)
        tokens = self.tokens(torch.cat((history.design, history.obs, time), -1))
        mask = torch.ones((batch, steps), device=tokens.device) if history.mask is None else history.mask.to(tokens)
        pooled = (tokens * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        maximum = tokens.masked_fill(~mask.bool()[..., None], -torch.inf).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, self.empty)
        return self.mix(torch.cat((pooled, maximum), -1))


class ActionBED(nn.Module):
    """Jointly trained posterior and bounded stochastic BED policy."""

    def __init__(self, env: pb.BED, hidden: int, min_std: float):
        super().__init__()
        theta_dim = int(torch.tensor(env.specs["theta"].shape).prod())
        design_dim = int(torch.tensor(env.specs["design"].shape).prod())
        obs_dim = int(torch.tensor(env.specs["obs"].shape).prod())
        self.encoder = HistoryEncoder(design_dim, obs_dim, hidden)
        self.posterior_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * theta_dim))
        self.action_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * design_dim))
        self.theta_shape = env.specs["theta"].shape
        self.design_shape = env.specs["design"].shape
        self.low, self.high, self.min_std = env.specs["design"].low, env.specs["design"].high, min_std

    def infer(self, history: pb.Batch, **context: object) -> torch.distributions.Independent:
        mean, raw_std = self.posterior_head(self.encoder(history)).chunk(2, -1)
        std = torch.nn.functional.softplus(raw_std) + self.min_std
        return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)

    def design(
        self,
        history: pb.Batch,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool | None = None,
        **context: object,
    ) -> torch.Tensor:
        mean, raw_std = self.action_head(self.encoder(history)).chunk(2, -1)
        std = torch.nn.functional.softplus(raw_std) + self.min_std
        deterministic = not self.training if deterministic is None else deterministic
        latent = (
            mean if deterministic else mean + std * torch.randn(mean.shape, generator=generator, device=mean.device)
        )
        bounded = self.low + (self.high - self.low) * torch.sigmoid(latent)
        return bounded.reshape(len(history), *self.design_shape)

    def entropy(self, history: pb.Batch) -> torch.Tensor:
        mean, raw_std = self.action_head(self.encoder(history)).chunk(2, -1)
        std = torch.nn.functional.softplus(raw_std) + self.min_std
        return torch.distributions.Normal(mean, std).entropy().sum(-1).mean()


model = ActionBED(env, cfg.hidden, cfg.min_std).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
run = pb.exp.Run.create(cfg.runs, "actionbed", asdict(cfg), seed_value=cfg.seed)
run.save_figure(preview_fig, "environment-random-preview")


# %% Joint training: prefix posterior error differentiates through every action and outcome
records: list[dict[str, float]] = []
truth_stream = pb.data.EpochStream(env, cfg.steps_per_epoch * cfg.batch_size, seed=cfg.seed)
seeds = pb.data.SeedBank(cfg.seed)

for epoch in range(cfg.epochs):
    epoch_truth = truth_stream.theta(epoch).reshape(cfg.steps_per_epoch, cfg.batch_size, *env.specs["theta"].shape)
    epoch_loss = 0.0
    epoch_mse = 0.0
    model.train()

    for step, theta_cpu in enumerate(epoch_truth):
        theta = theta_cpu.to(device)
        history = pb.Batch(theta=theta, design=None, obs=None, mask=None)
        posterior_losses = []
        posterior_mses = []

        for time in range(cfg.budget):
            action_generator = seeds.torch("action", epoch, step, time, device=device)
            noise_generator = seeds.torch("observation", epoch, step, time, device=device)
            design = model.design(history, generator=action_generator)
            obs = env.simulate(theta, design, generator=noise_generator, epoch=epoch, step=time)
            designs = design[:, None] if history.design is None else torch.cat((history.design, design[:, None]), 1)
            outcomes = obs[:, None] if history.obs is None else torch.cat((history.obs, obs[:, None]), 1)
            mask = torch.ones((cfg.batch_size, time + 1), dtype=torch.bool, device=device)
            history = pb.Batch(theta=theta, design=designs, obs=outcomes, mask=mask)
            posterior = model.infer(history)
            truth = theta.flatten(1)
            posterior_losses.append(-posterior.log_prob(truth).mean())
            posterior_mses.append((posterior.mean - truth).square().mean())

        loss = torch.stack(posterior_losses).mean() - cfg.entropy_weight * model.entropy(history)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        epoch_loss += float(loss.detach())
        epoch_mse += float(torch.stack(posterior_mses).mean().detach())

    diagnostics = run.log(
        (epoch + 1) * cfg.steps_per_epoch,
        epoch=epoch,
        loss=epoch_loss / cfg.steps_per_epoch,
        posterior_mse=epoch_mse / cfg.steps_per_epoch,
    )
    records.append(diagnostics)

run.checkpoint("last", model=model, optimizer=optimizer, step=cfg.epochs * cfg.steps_per_epoch)


# %% The same model now supplies PyBED's design, infer, and predict-facing API
def posterior_predict(query: torch.Tensor, history: pb.Batch, *, env: pb.BED) -> torch.Tensor:
    posterior = model.infer(history.to(device))
    theta = posterior.mean.reshape(len(history), *env.specs["theta"].shape)
    return pb.sim.location(
        theta,
        query.to(device),
        strength=env.cfg["strength"],
        background=env.cfg["background"],
        softening=env.cfg["softening"],
        noise=0.0,
        log_signal=env.cfg["log_signal"],
    )


action_env = env.with_components(design=model.design, infer=model.infer, predict=posterior_predict)
print(action_env)


# %% Common-truth evaluation against random designs
def score(candidate_env: pb.BED, batch: pb.Batch) -> dict[str, float]:
    history = batch.to(device)
    with torch.no_grad():
        posterior = model.infer(history)
        truth = history.theta.flatten(1)
        return {
            "mse": float(pb.metrics.mse(posterior.mean, truth)),
            "log_mse": float(pb.metrics.log_mse(posterior.mean, truth)),
            "nll": float(-posterior.log_prob(truth).mean()),
        }


model.eval()


def learned_policy(history: pb.Batch, **context: object) -> torch.Tensor:
    with torch.no_grad():
        return model.design(history.to(device), deterministic=True).cpu()


results, trajectories = pb.exp.compare(
    env,
    {"ActionBED": learned_policy, "random": None},
    score,
    episodes=cfg.eval_episodes,
    seed_value=cfg.seed + 10_000,
    budget=cfg.budget,
)
print(results)


# %% Camera-ready diagnostics and trajectories
loss_fig, _ = pb.vs.training(records, keys=("loss", "posterior_mse"), logy=True)
run.save_figure(loss_fig, "training")

comparison_fig, _ = pb.vs.benchmark(results, metric="mse")
run.save_figure(comparison_fig, "policy-comparison")

learned_fig, _ = env.visualize("episode", trajectories["ActionBED"].batch, index=0)
random_fig, _ = env.visualize("episode", trajectories["random"].batch, index=0)
run.save_figure(learned_fig, "learned-trajectory")
run.save_figure(random_fig, "random-trajectory")

run.save_data(trajectories["ActionBED"], "evaluation-actionbed")
run.save_data(trajectories["random"], "evaluation-random")
run.finish(results=results)

plt.show()
