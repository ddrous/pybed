import math

import torch

import pybed as pb
from pybed import metrics


def test_basic_metrics_and_set_matching():
    """Check basic errors and permutation-invariant source matching."""
    truth = torch.tensor([[[0.0], [1.0]]])
    estimate = torch.tensor([[[1.0], [0.0]]])
    assert metrics.set_mse(estimate, truth) == 0
    assert metrics.mse(torch.ones(2), torch.zeros(2)) == 1
    assert torch.allclose(metrics.log_mse(torch.ones(2), torch.zeros(2)), torch.tensor(0.0))


def test_spce_known_cases():
    """Check sPCE at an uninformative and an informative likelihood table."""
    equal = torch.zeros(8, 5)
    assert torch.allclose(metrics.spce(equal), torch.tensor(0.0))
    separated = torch.tensor([[3.0, 0.0, 0.0]] * 20)
    assert 0 < metrics.spce(separated) < math.log(3)


def test_proper_scores_and_summary():
    """Check sample-based posterior scores and their combined summary."""
    truth = torch.zeros(4, 2)
    samples = torch.zeros(4, 16, 2)
    assert metrics.energy_score(samples, truth) == 0
    assert metrics.coverage(samples, truth, mass=0.9) == 1
    summary = metrics.posterior_summary(samples, truth)
    assert summary["mse"] == 0
    assert summary["energy_score"] == 0


def test_weighted_particles_and_class_metrics():
    """Check weighted energy scores and categorical particle-cloud metrics."""
    truth = torch.tensor([[1.0, 0.0]])
    cloud = pb.ParticleCloud(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]),
        torch.tensor([[2.0, 1.0, 1.0]]),
    )
    assert torch.isfinite(metrics.energy_score(cloud, truth, unbiased=True))
    probabilities = metrics.class_probabilities(cloud, classes=2)
    assert torch.allclose(probabilities, torch.tensor([[0.5, 0.5]]))
    assert metrics.classification_accuracy(cloud, torch.tensor([0])) == 1
