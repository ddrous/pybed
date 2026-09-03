import torch

import pybed as pb


def test_environment_summary_and_component_replacement():
    env = pb.make("location-v0", sources=2, dims=2, budget=5)
    assert "location-v0" in repr(env)
    assert "theta" in repr(env)

    def exact(theta, design):
        return torch.zeros((*theta.shape[:-2], 1))

    custom = env.with_components(simulate=exact, infer=lambda history: history.theta)
    theta = custom.sample_prior(4, seed=2)
    design = torch.full((4, 2), 0.5)
    assert torch.equal(custom.simulate(theta, design), torch.zeros(4, 1))
    assert torch.equal(custom.infer(pb.Batch(theta)), theta)
    assert env.posterior is None


def test_spec_and_empirical_prior():
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


def test_registry_contains_builtins():
    assert {"location-v0", "mnist-discovery-v0", "advdiff-v0", "pendulum-v0"}.issubset(pb.available())
