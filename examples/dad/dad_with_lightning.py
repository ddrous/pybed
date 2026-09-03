"""Train a small DAD policy with Lightning and record the run with W&B."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import pybed as pb

try:
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
    import wandb
except ImportError as error:
    raise ImportError("Install this example with `pip install -e '.[lightning]'`") from error

# Specify path to config.yaml; if not present, load from script folder
CONFIG_PATH: str | None = None
DEBUG = False

@dataclass(frozen=True)
class Config:
    """Settings for the example."""
    seed: int = 2030


class DesignPolicy(nn.Module):
    """Map past observation pairs to the next bounded design."""
    def __init__(self, low: float, high: float):
        """Create history encoder and output head."""
        super().__init__()
        self.low, self.high = low, high
        self.pair = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.empty = nn.Parameter(torch.zeros(64))
        self.head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, history: pb.Batch) -> torch.Tensor:
        """Return one design per episode."""
        if history.designs is None:
            summary = self.empty.expand(len(history), -1)
        else:
            summary = self.pair(torch.cat((history.designs, history.outcomes), -1)).sum(1)
        unit_x = torch.sigmoid(self.head(summary))
        return self.low + (self.high - self.low) * unit_x


class DAD(L.LightningModule):
    """Train a DAD policy via sequential prior contrastive estimation."""
    def __init__(self, env: pb.BED, cfg: Config):
        """Create the Lightning model."""
        super().__init__()
        self.env, self.cfg = env, cfg
        self.policy = DesignPolicy(float(env.cfg["low"]), float(env.cfg["high"]))
        self.save_hyperparameters(asdict(cfg))

    def rollout_log_likelihoods(self, theta: torch.Tensor) -> torch.Tensor:
        """Return full-history log likelihoods for parameters."""
        batch_size, choices = theta.shape[:2]
        truth = theta[:, 0]
        history = pb.Batch(mask=torch.empty((batch_size, 0), dtype=torch.bool, device=theta.device))
        log_likelihoods = torch.zeros((batch_size, choices), device=theta.device)
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for step in range(self.cfg.budget):
            x = self.policy(history)
            generator = torch.Generator(device=theta.device).manual_seed(
                self.cfg.seed + self.global_step * self.cfg.budget + step
            )
            y = self.env.simulate(truth, x, generator=generator)
            log_likelihoods = log_likelihoods + self.env.log_prob(
                y[:, None, :], theta, x[:, None, :]
            )
            xs.append(x)
            ys.append(y)
            history = pb.Batch(designs=torch.stack(xs, 1), outcomes=torch.stack(ys, 1))
        return log_likelihoods

    def training_step(self, batch: tuple[torch.Tensor], batch_index: int) -> torch.Tensor:
        """Log the DAD bound and return its loss."""
        del batch_index
        log_likelihoods = self.rollout_log_likelihoods(batch[0])
        bound = pb.metrics.spce(log_likelihoods)
        self.log("train/sPCE", bound, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return -bound

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)


def training_data(env: pb.BED, cfg: Config) -> TensorDataset:
    """Return fixed true-plus-contrastive prior draws."""
    draws = env.sample_prior(
        cfg.train_episodes * (cfg.contrastives + 1),
        generator=pb.data.Seeds(cfg.seed).torch("dad-training"),
    )
    theta = torch.as_tensor(draws).reshape(cfg.train_episodes, cfg.contrastives + 1, 1, 2)
    return TensorDataset(theta)


def evaluate(model: DAD, env: pb.BED, cfg: Config, output: Path, logger: WandbLogger) -> None:
    """Evaluate policy and save trajectory locally and in W&B."""
    model.eval()

    def policy(history: pb.Batch, **_: object) -> torch.Tensor:
        """Take a CPU history and return trained designs on the CPU."""
        with torch.no_grad():
            return model.policy(history.to(model.device)).cpu()

    dataset = pb.data.Stream(env, cfg.eval_episodes, seed=cfg.seed + 1).generate(policy)
    figure, _ = env.visualize("episode", dataset.batch, index=0)
    figure.savefig(output / "trajectory.png", dpi=180, bbox_inches="tight")
    dataset.save(output / "evaluation.pt")
    logger.experiment.log(
        {
            "evaluation/mean_outcome": float(dataset.batch.outcomes.mean()),
            "evaluation/trajectory": wandb.Image(figure),
        }
    )
    plt.close(figure)


# %% Read configuration and prepare run
config_path = Path(CONFIG_PATH) if CONFIG_PATH else Path(__file__).with_name("config.yaml")
if not config_path.exists():
    config_path = Path(__file__).with_name("config.yaml")
cfg = Config(**pb.expt.load_config(config_path))
if DEBUG:
    cfg = replace(cfg, budget=3, train_episodes=64, eval_episodes=16, contrastives=3, batch_size=16, epochs=1, wandb_mode="disabled")
L.seed_everything(cfg.seed, workers=True)
env = pb.make("location-v0", sources=1, dims=2, budget=cfg.budget, low=-4.0, high=4.0, noise=0.5)
run = pb.expt.Run.create(cfg.runs, "dad-lightning-wandb", cfg, seed_value=cfg.seed, source_files=[__file__, config_path])
output = run.path


# %% Build data and logger
data = training_data(env, cfg)
train_loader = DataLoader(
    data,
    batch_size=cfg.batch_size,
    shuffle=True,
    generator=pb.data.Seeds(cfg.seed).torch("dad-loader"),
    num_workers=0,
)
logger = WandbLogger(
    project=cfg.wandb_project,
    name="dad-location",
    save_dir=output,
    offline=cfg.wandb_mode == "offline",
    mode=cfg.wandb_mode,
    log_model=False,
)


# %% Train and evaluate
model = DAD(env, cfg)
use_gpu = torch.cuda.is_available()
devices = cfg.devices if use_gpu else 1
trainer = L.Trainer(
    max_epochs=cfg.epochs,
    accelerator="gpu" if use_gpu else "cpu",
    devices=devices,
    strategy="ddp" if devices > 1 else "auto",
    logger=logger,
    default_root_dir=output,
    enable_checkpointing=True,
    deterministic=True,
    log_every_n_steps=1 if DEBUG else 10,
    enable_progress_bar=not DEBUG,
)
trainer.fit(model, train_loader, ckpt_path=cfg.checkpoint)
trainer.save_checkpoint(output / "last.ckpt")
if trainer.is_global_zero:
    evaluate(model, env, cfg, output, logger)
    run.finish(checkpoint="last.ckpt")
    logger.experiment.finish()
