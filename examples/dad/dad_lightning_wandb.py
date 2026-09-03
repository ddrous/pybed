"""Train a small DAD policy with Lightning and record the run with W&B."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
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


QUICK = os.getenv("PYBED_QUICK", "0") == "1"


@dataclass(frozen=True)
class Config:
    """Hold the settings consumed by the example and produce no output on its own."""

    seed: int = 2030
    budget: int = 3 if QUICK else 10
    train_episodes: int = 64 if QUICK else 16_384
    eval_episodes: int = 16 if QUICK else 512
    contrastives: int = 3 if QUICK else 31
    batch_size: int = 16 if QUICK else 256
    epochs: int = 1 if QUICK else 100
    learning_rate: float = 3e-4
    devices: int = int(os.getenv("PYBED_DEVICES", "1"))
    runs: str = os.getenv("PYBED_RUNS", "runs")
    wandb_project: str = os.getenv("WANDB_PROJECT", "pybed-dad")
    wandb_mode: str = os.getenv("WANDB_MODE", "offline")


class DesignPolicy(nn.Module):
    """Map a set of past observation pairs to the next bounded design."""

    def __init__(self, low: float, high: float):
        """Take design bounds and create the history encoder and output head."""
        super().__init__()
        self.low, self.high = low, high
        self.pair = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.empty = nn.Parameter(torch.zeros(64))
        self.head = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, history: pb.Batch) -> torch.Tensor:
        """Take a Batch history and return one design per episode."""
        if history.x is None:
            summary = self.empty.expand(len(history), -1)
        else:
            summary = self.pair(torch.cat((history.x, history.y), -1)).sum(1)
        unit_x = torch.sigmoid(self.head(summary))
        return self.low + (self.high - self.low) * unit_x


class DAD(L.LightningModule):
    """Train a DAD policy by maximizing sequential prior contrastive estimation."""

    def __init__(self, env: pb.BED, cfg: Config):
        """Take a PyBED environment and settings and create the Lightning model."""
        super().__init__()
        self.env, self.cfg = env, cfg
        self.policy = DesignPolicy(float(env.cfg["low"]), float(env.cfg["high"]))
        self.save_hyperparameters(asdict(cfg))

    def rollout_log_likelihoods(self, theta: torch.Tensor) -> torch.Tensor:
        """Take true-plus-contrastive parameters and return their full-history log likelihoods."""
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
            history = pb.Batch(x=torch.stack(xs, 1), y=torch.stack(ys, 1))
        return log_likelihoods

    def training_step(self, batch: tuple[torch.Tensor], batch_index: int) -> torch.Tensor:
        """Take one parameter batch and index, log the DAD bound, and return its loss."""
        del batch_index
        log_likelihoods = self.rollout_log_likelihoods(batch[0])
        bound = pb.metrics.spce(log_likelihoods)
        self.log("train/sPCE", bound, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return -bound

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Take the model parameters and return their Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)


def training_data(env: pb.BED, cfg: Config) -> TensorDataset:
    """Take an environment and settings and return fixed true-plus-contrastive prior draws."""
    draws = env.sample_prior(
        cfg.train_episodes * (cfg.contrastives + 1),
        generator=pb.data.Seeds(cfg.seed).torch("dad-training"),
    )
    theta = torch.as_tensor(draws).reshape(cfg.train_episodes, cfg.contrastives + 1, 1, 2)
    return TensorDataset(theta)


def evaluate(model: DAD, env: pb.BED, cfg: Config, output: Path, logger: WandbLogger) -> None:
    """Take a trained policy, evaluate it, and save its trajectory locally and in W&B."""
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
            "evaluation/mean_outcome": float(dataset.batch.y.mean()),
            "evaluation/trajectory": wandb.Image(figure),
        }
    )
    plt.close(figure)


def main() -> None:
    """Build the data and services, train DAD, evaluate it, and return nothing."""
    cfg = Config()
    L.seed_everything(cfg.seed, workers=True)
    env = pb.make(
        "location-v0",
        sources=1,
        dims=2,
        budget=cfg.budget,
        low=-4.0,
        high=4.0,
        noise=0.5,
    )
    output = Path(cfg.runs) / "dad-lightning-wandb"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_CACHE_DIR", str(output / "wandb-cache"))
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
        log_every_n_steps=1 if QUICK else 10,
        enable_progress_bar=not QUICK,
    )
    trainer.fit(model, train_loader)
    trainer.save_checkpoint(output / "last.ckpt")
    if trainer.is_global_zero:
        evaluate(model, env, cfg, output, logger)
        logger.experiment.finish()


if __name__ == "__main__":
    main()
