import torch

import pybed as pb


def test_environment_summary_and_component_replacement():
    """Check component replacement and the direct environment verbs."""
    env = pb.make("location-v0", sources=2, dims=2, budget=5)
    assert "location-v0" in repr(env)
    assert "theta" in repr(env)

    def exact(theta, x):
        """Take parameters and designs and return zero outcomes."""
        return torch.zeros((*theta.shape[:-2], 1))

    custom = env.with_components(simulate=exact, infer=lambda history: history.theta)
    theta = custom.sample_prior(4, seed=2)
    x = torch.full((4, 2), 0.5)
    assert torch.equal(custom.simulate(theta, x), torch.zeros(4, 1))
    assert torch.equal(custom.infer(pb.Batch(theta)), theta)
    assert env.posterior is None


def test_spec_and_empirical_prior():
    """Check event validation and repeatable empirical-prior sampling."""
    spec = pb.Spec((2,), low=0, high=1)
    spec.check(torch.ones(3, 2), bounds=True)
    try:
        spec.check(torch.ones(3, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch was not detected")

    points = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    prior = pb.EmpiricalPrior(points, jitter=0.1)
    a = prior.sample((8,), generator=torch.Generator().manual_seed(4))
    b = prior.sample((8,), generator=torch.Generator().manual_seed(4))
    assert torch.equal(a, b)
    assert torch.isfinite(prior.log_prob(a)).all()

    scalar = pb.EmpiricalPrior(torch.tensor([-1.0, 1.0]), jitter=0.2)
    assert scalar.log_prob(torch.tensor([0.0, 0.5])).shape == (2,)


def test_registry_contains_builtins():
    """Check that each built-in environment is discoverable."""
    assert {
        "location-v0",
        "mnist-discovery-v0",
        "mnist-classification-v0",
        "advdiff-v0",
        "pendulum-v0",
    }.issubset(pb.available())


def test_observation_aliases_and_joint_verbs():
    """Check observation aliases and each joint-call return order."""
    x, y = torch.ones(2, 1), torch.zeros(2, 1)
    batch = pb.Batch(theta=torch.zeros(2, 1), design=x, outcome=y)
    assert batch.x is batch.design and batch.y is batch.outcome
    assert batch.obs.x is x and batch.o.y is y

    env = pb.BED(
        "joint",
        pb.Normal(0.0, 1.0),
        lambda theta, x: y,
        {"theta": pb.Spec(()), "x": pb.Spec((1,)), "y": pb.Spec((1,))},
    ).with_components(
        design=lambda history: x,
        infer=lambda history: "inference",
        predict=lambda query, history: "prediction",
    )
    assert env.infer_design(batch) == ("inference", x)
    assert env.design_predict(batch) == (x, "prediction")
    assert env.design_infer_predict(batch) == (x, "inference", "prediction")

    pair_env = env.with_components(infer=lambda history: history.o)
    inferred_o = pair_env.infer(pb.Observation(x, y))
    assert inferred_o.x is x and inferred_o.y is y

    one_pass = env.with_components(
        design_infer_predict=lambda history: (x + 1, "joint inference", "joint prediction")
    )
    result = one_pass.design_infer_predict(batch)
    assert torch.equal(result[0], x + 1)
    assert result[1:] == ("joint inference", "joint prediction")
    inference, design = one_pass.infer_design(batch)
    assert inference == "joint inference" and torch.equal(design, x + 1)
    design, prediction = one_pass.design_predict(batch)
    assert torch.equal(design, x + 1) and prediction == "joint prediction"


def test_particle_cloud_statistics():
    """Check weighted particle means, resampling, and effective sample size."""
    points = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    cloud = pb.ParticleCloud(points, torch.tensor([[1.0, 2.0, 1.0]]))
    assert torch.allclose(cloud.mean(), torch.tensor([[0.25, 0.75]]))
    assert torch.allclose(cloud.effective_sample_size(), torch.tensor([8 / 3]))
    assert cloud.sample(5, generator=torch.Generator().manual_seed(2)).shape == (1, 5, 2)
    labels = pb.ParticleCloud(torch.tensor([[4, 9, 9]]))
    assert labels.sample(2, generator=torch.Generator().manual_seed(2)).shape == (1, 2)
    assert torch.allclose(labels.effective_sample_size(), torch.tensor([3.0]))
