import torch

import pybed as pb


def test_location_shape_reproducibility_and_gradient():
    env = pb.make("location-v0", sources=2, dims=2)
    theta = env.sample_prior(5, seed=1)
    design = torch.full((5, 2), 0.4, requires_grad=True)
    first = env.simulate(theta, design, seed=7)
    second = env.simulate(theta, design, seed=7)
    assert first.shape == (5, 1)
    assert torch.equal(first, second)
    first.sum().backward()
    assert design.grad is not None and torch.isfinite(design.grad).all()


def test_image_discovery_with_array_data():
    images = torch.zeros(4, 1, 28, 28)
    images[:, :, 10:18, 12:16] = 1
    labels = torch.arange(4)
    env = pb.envs.mnist(images=images, labels=labels, budget=2, noise=0)
    theta = env.sample_prior(3, seed=3)
    design = torch.full((3, 2), 0.5, requires_grad=True)
    obs = env.simulate(theta, design, seed=8)
    assert obs.shape == (3, 1, 28, 28)
    obs.sum().backward()
    assert design.grad is not None


def test_advdiff_time_series_and_gradient():
    env = pb.make("advdiff-v0", shape=(12, 12), sensors=2, times=(0.05, 0.1), dt=0.01, diffusion=0.001)
    theta = env.sample_prior(2, seed=5).requires_grad_()
    design = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]] * 2, requires_grad=True)
    obs = env.simulate(theta, design, seed=6)
    assert obs.shape == (2, 2, 2)
    obs.mean().backward()
    assert theta.grad is not None and design.grad is not None
    assert torch.isfinite(theta.grad).all() and torch.isfinite(design.grad).all()


def test_pendulum_and_additional_literature_sims():
    pendulum = pb.make("pendulum-v0", steps=20, keep_every=5)
    theta = pendulum.sample_prior(3, seed=1)
    design = torch.zeros(3, 2)
    assert pendulum.simulate(theta, design, seed=2).shape == (3, 5)

    ces = pb.make("ces-v0", goods=2)
    theta = ces.sample_prior(3, seed=1)
    assert ces.simulate(theta, torch.ones(3, 4), seed=2).shape == (3, 1)

    death = pb.make("death-v0")
    theta = death.sample_prior(3, seed=1)
    outcome = death.simulate(theta, torch.ones(3, 1), seed=2)
    assert outcome.shape == (3, 1)
    assert ((outcome >= 0) & (outcome <= death.cfg["population"])).all()
