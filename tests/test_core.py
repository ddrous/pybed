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

    custom = env.with_components(simulate=exact, infer=lambda history: history.parameters)
    theta = custom.sample_prior(4, seed=2)
    x = torch.full((4, 2), 0.5)
    assert torch.equal(custom.simulate(theta, x), torch.zeros(4, 1))
    assert torch.equal(custom.infer(pb.Batch(parameters=theta))["infer"], theta)
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
    }.issubset(pb.environments())


def test_observation_aliases_and_model_results():
    """Check the readable batch aliases and shared model-result dictionaries."""
    x, y = torch.ones(2, 1), torch.zeros(2, 1)
    theta = torch.zeros(2, 1)
    batch = pb.Batch(parameters=theta, designs=x, outcomes=y)
    assert batch.parameters is batch.theta is batch.thetas
    assert batch.designs is batch.design is batch.x
    assert batch.outcomes is batch.outcome is batch.y
    assert batch.obs.design is x and batch.o.outcome is y

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
    output = env.design(batch, infer=True, predict=True)
    assert output == {"design": x, "infer": "inference", "predict": "prediction"}
    assert env.infer(batch) == {"design": None, "infer": "inference", "predict": None}
    assert env.predict(x, batch, infer=True) == {
        "design": None,
        "infer": "inference",
        "predict": "prediction",
    }

    pair_env = env.with_components(infer=lambda history: history.obs)
    inferred_o = pair_env.infer(pb.Observation(x, y))
    assert inferred_o["infer"].design is x and inferred_o["infer"].outcome is y

    def shared(history, infer=False, predict=False):
        """Take a history and flags and return all requested outputs in one pass."""
        return {
            "design": x + 1,
            "infer": "shared inference" if infer else None,
            "predict": "shared prediction" if predict else None,
        }

    one_pass = env.with_components(design=shared)
    combined = one_pass.design(batch, infer=True, predict=True)
    assert torch.equal(combined["design"], x + 1)
    assert combined["infer"] == "shared inference"
    assert combined["predict"] == "shared prediction"


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
