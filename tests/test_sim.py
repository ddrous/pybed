import torch

import pybed as pb


def test_location_shape_reproducibility_and_gradient():
    """Check location-simulator shapes, repeatability, and design gradients."""
    env = pb.make("location-v0", sources=2, dims=2)
    theta = env.sample_prior(5, seed=1)
    x = torch.full((5, 2), 0.4, requires_grad=True)
    first = env.simulate(theta, x, seed=7)
    second = env.simulate(theta, x, seed=7)
    assert first.shape == (5, 1)
    assert torch.equal(first, second)
    first.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_image_discovery_with_array_data():
    """Check array-backed image discovery and design gradients."""
    images = torch.zeros(4, 1, 28, 28)
    images[:, :, 10:18, 12:16] = 1
    labels = torch.arange(4)
    env = pb.envs.mnist(images=images, labels=labels, budget=2, noise=0)
    theta = env.sample_prior(3, seed=3)
    x = torch.full((3, 2), 0.5, requires_grad=True)
    y = env.simulate(theta, x, seed=8)
    assert y.shape == (3, 1, 28, 28)
    y.sum().backward()
    assert x.grad is not None


def test_mnist_classification_patch_and_labels():
    """Check the Action-BED MNIST patch shape, labels, and design gradients."""
    images = torch.zeros(4, 1, 28, 28)
    images[:, :, 8:20, 8:20] = 1
    labels = torch.tensor([4, 9, 4, 9])
    env = pb.envs.mnist_classification(images=images, labels=labels, noise=0, budget=2)
    dataset = pb.data.Stream(env, episodes=3, seed=2).generate()
    assert dataset.batch.y.shape == (3, 2, 1, 5, 5)
    assert dataset.batch.target.shape == (3,)
    theta = dataset.batch.theta
    x = torch.full((3, 2), 0.25, requires_grad=True)
    env.simulate(theta, x).sum().backward()
    assert x.grad is not None


def test_advdiff_time_series_and_gradient():
    """Check PDE output shapes and gradients with respect to fields and designs."""
    env = pb.make("advdiff-v0", shape=(12, 12), sensors=2, times=(0.05, 0.1), dt=0.01, diffusion=0.001)
    theta = env.sample_prior(2, seed=5).requires_grad_()
    x = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]] * 2, requires_grad=True)
    y = env.simulate(theta, x, seed=6)
    assert y.shape == (2, 2, 2)
    y.mean().backward()
    assert theta.grad is not None and x.grad is not None
    assert torch.isfinite(theta.grad).all() and torch.isfinite(x.grad).all()


def test_pendulum_and_additional_literature_sims():
    """Check the output shapes and bounds of the remaining built-in simulators."""
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
