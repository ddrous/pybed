# %% ACTION-BED for source location finding (Rossa, Phillips & Rainforth, 2026)
"""Paper-faithful, single-file ACTION-BED example.

This learns a design policy and a final point estimate directly from expected
future loss. See config.yaml in the DAD example for paper settings.
"""

from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import torch
from torch import nn

from tqdm.auto import tqdm
import pybed as pb

DEBUG = False

@dataclass(frozen=True)
class Config:
    """Training, evaluation, and saved-model settings."""
    seed: int = 42
    sources: int = 2
    budget: int = 30
    batch_size: int = 256
    warmup_steps: int = 50_00
    joint_steps: int = 150_00
    learning_rate: float = 5e-4
    decay: float = 0.95
    decay_period: int = 2_000
    objective: str = "log_mse"
    eval_episodes: int = 2_048
    log_every: int = 100
    runs: str = "runs"
    checkpoint: str | None = None


# %% Setup
cfg = Config()
pb.expt.seed(cfg.seed, deterministic=False)
torch.use_deterministic_algorithms(False)
pb.vis.style()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prior = pb.Normal(torch.zeros(cfg.sources, 2), torch.ones(cfg.sources, 2))
env = pb.make(
    "location-v0",
    sources=cfg.sources,
    dims=2,
    budget=cfg.budget,
    prior=prior,
    low=-4.0,
    high=4.0,
    background=0.1,
    softening=1e-4,
    strength=1.0,
    noise=0.5,
    log_signal=True,
)
seeds = pb.data.Seeds(cfg.seed)
run = pb.expt.Run.create(
    cfg.runs,
    "actionbed",
    asdict(cfg),
    seed_value=cfg.seed,
    source_files=[__file__],
)


# %% TNP-like deterministic design policy from Appendix E.1
class DesignPolicy(nn.Module):
    def __init__(self, budget: int):
        """Take an experiment budget and create the sequential design network."""
        super().__init__()
        self.design_encoder = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 32))
        self.outcome_encoder = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 32))
        self.pair = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))
        self.adapter = nn.Linear(32, 64)
        self.history_type = nn.Parameter(torch.empty(64).normal_(std=0.02))
        self.decision_type = nn.Parameter(torch.empty(64).normal_(std=0.02))
        self.decision = nn.Parameter(torch.empty(64).normal_(std=0.02))
        self.time = nn.Embedding(budget + 1, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(64)
        self.emitter = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, history: pb.Batch) -> torch.Tensor:
        """Take an observation history and return the next two-dimensional design."""
        batch = len(history)
        steps = 0 if history.designs is None else history.designs.shape[1]
        if steps:
            x = self.design_encoder(history.designs)
            y = self.outcome_encoder(history.outcomes)
            tokens = self.adapter(self.pair(torch.cat((x, y), -1)))
            times = self.time(torch.arange(steps, device=tokens.device))[None]
            tokens = tokens + self.history_type + times
        else:
            tokens = self.decision.new_empty(batch, 0, 64)
        query = self.decision + self.decision_type + self.time.weight[steps]
        tokens = torch.cat((tokens, query.expand(batch, 1, -1)), 1)
        return self.emitter(self.transformer(self.norm(tokens))[:, -1])


# %% Separate terminal downstream action policy from Appendix G.1
class ActionPolicy(nn.Module):
    def __init__(self, budget: int, sources: int):
        """Take the budget and source count and create the final action network."""
        super().__init__()
        self.sources = sources
        self.net = nn.Sequential(
            nn.Linear(3 * budget, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2 * sources),
        )

    def forward(self, history: pb.Batch) -> torch.Tensor:
        """Take a complete history and return estimated source locations."""
        pairs = torch.cat((history.designs, history.outcomes), -1).flatten(1)
        return self.net(pairs).reshape(len(history), self.sources, 2)


class ActionBED(nn.Module):
    def __init__(self):
        """Create the design and downstream action networks and return no separate value."""
        super().__init__()
        self.design_policy = DesignPolicy(cfg.budget)
        self.action_policy = ActionPolicy(cfg.budget, cfg.sources)


model = ActionBED().to(device)
if cfg.checkpoint:
    saved = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved.get("model", saved))


def sample_truth(stage: str, step: int) -> torch.Tensor:
    """Take a training stage and step and return a repeatable batch of source locations."""
    generator = seeds.torch(stage, step, device=device)
    return torch.randn((cfg.batch_size, cfg.sources, 2), generator=generator, device=device)


def rollout(theta: torch.Tensor, step: int, *, random: bool) -> pb.Batch:
    """Take source locations and return their random or learned observation history."""
    history = pb.Batch(parameters=theta, designs=None, outcomes=None, mask=None)
    xs, ys = [], []
    for time in range(cfg.budget):
        if random:
            x = torch.randn((len(theta), 2), generator=seeds.torch("warm-design", step, time, device=device), device=device)
        else:
            x = model.design_policy(history)
        y = env.simulate(
            theta,
            x,
            generator=seeds.torch("noise", random, step, time, device=device),
        )
        xs.append(x)
        ys.append(y)
        history = pb.Batch(
            parameters=theta,
            designs=torch.stack(xs, 1),
            outcomes=torch.stack(ys, 1),
            mask=None,
        )
    return history


def canonical(theta: torch.Tensor) -> torch.Tensor:
    """Take exchangeable source locations and return a repeatable norm-based ordering."""
    order = theta.square().sum(-1).argsort(-1)
    return theta.gather(1, order[..., None].expand_as(theta))


def future_loss(estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Take estimated and true sources and return the configured downstream loss."""
    if cfg.objective == "mse":
        direct = (estimate - truth).square().sum((-1, -2))
        swapped = (estimate - truth.flip(1)).square().sum((-1, -2))
        return torch.minimum(direct, swapped).mean()
    if cfg.objective == "log_mse":
        error = (estimate - canonical(truth)).square().sum((-1, -2))
        return (error + torch.log(error + 1e-6)).mean()
    raise ValueError("objective must be 'mse' or 'log_mse'")


def update(optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ExponentialLR, step: int, *, random: bool) -> float:
    """Take optimizer state and one step setting, update the model, and return its loss."""
    optimizer.zero_grad(set_to_none=True)
    theta = sample_truth("warm-theta" if random else "joint-theta", step)
    history = rollout(theta, step, random=random)
    loss = future_loss(model.action_policy(history), theta)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite {cfg.objective} loss at step {step}")
    loss.backward()
    optimizer.step()
    if (step + 1) % cfg.decay_period == 0:
        scheduler.step()
    return float(loss.detach())


# %% Algorithm 2: downstream warm-up, then joint pathwise optimization
records: list[dict[str, float]] = []
model.train()
warm_optimizer = torch.optim.Adam(model.action_policy.parameters(), lr=cfg.learning_rate, betas=(0.8, 0.998))
warm_scheduler = torch.optim.lr_scheduler.ExponentialLR(warm_optimizer, gamma=cfg.decay)
warm_progress = tqdm(range(cfg.warmup_steps), desc="Action-BED warm-up", unit="step", dynamic_ncols=True)
for step in warm_progress:
    loss = update(warm_optimizer, warm_scheduler, step, random=True)
    warm_progress.set_postfix(loss=f"{loss:.4g}", lr=f"{warm_optimizer.param_groups[0]['lr']:.2e}", refresh=False)
    if step % cfg.log_every == 0 or step + 1 == cfg.warmup_steps:
        records.append(run.log(step + 1, phase=0, loss=loss))

run.checkpoint("warmup", model=model, optimizer=warm_optimizer, step=cfg.warmup_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.8, 0.998))
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.decay)
joint_progress = tqdm(range(cfg.joint_steps), desc="Action-BED joint", unit="step", dynamic_ncols=True)
for step in joint_progress:
    loss = update(optimizer, scheduler, step, random=False)
    joint_progress.set_postfix(loss=f"{loss:.4g}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", refresh=False)
    if step % cfg.log_every == 0 or step + 1 == cfg.joint_steps:
        records.append(run.log(cfg.warmup_steps + step + 1, phase=1, loss=loss))

run.checkpoint("last", model=model, optimizer=optimizer, step=cfg.warmup_steps + cfg.joint_steps)


# %% Terminal downstream evaluation and trajectory visualization
model.eval()


def learned_policy(history: pb.Batch, **_: object) -> torch.Tensor:
    """Take a rollout history and return the trained design on the CPU."""
    with torch.no_grad():
        return model.design_policy(history.to(device)).cpu()


evaluation = pb.data.Stream(env, cfg.eval_episodes, seed=cfg.seed + 1, budget=cfg.budget).generate(learned_policy)
with torch.no_grad():
    batch = evaluation.batch.to(device)
    estimate = model.action_policy(batch)
    direct = (estimate - batch.parameters).square().mean((-1, -2))
    swapped = (estimate - batch.parameters.flip(1)).square().mean((-1, -2))
    set_mse = torch.minimum(direct, swapped)
    ordered_error = (estimate - canonical(batch.parameters)).square().sum((-1, -2))
    results = {
        "set_mse": float(set_mse.mean()),
        "log_mse": float(torch.log(ordered_error + 1e-6).mean()),
        "set_mse_se": float(set_mse.std() / len(set_mse) ** 0.5),
    }

print(results)
training_fig, _ = pb.vis.training(records, keys=("loss",), logy=False)
trajectory_fig, _ = env.visualize("episode", evaluation.batch, index=0)
run.save_figure(training_fig, "training")
run.save_figure(trajectory_fig, "trajectory")
run.save_data(evaluation, "evaluation")
run.finish(results=results)
plt.show()
