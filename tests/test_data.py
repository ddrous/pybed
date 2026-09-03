import torch

import pybed as pb


def test_epoch_reproducibility_and_common_truths(tmp_path):
    """Check repeatable rollout data, policy comparison, saving, and loading."""
    env = pb.make("location-v0", budget=4)
    stream = pb.data.Stream(env, episodes=24, seed=17)
    first = stream.generate(None, epoch=2)
    second = stream.generate(None, epoch=2)
    assert torch.equal(first.batch.theta, second.batch.theta)
    assert torch.equal(first.batch.x, second.batch.x)
    assert torch.equal(first.batch.y, second.batch.y)

    def center(history, **context):
        """Take a history and return centre designs for every episode."""
        return torch.full((len(history), 2), 0.5)

    other = stream.generate(center, epoch=2)
    assert torch.equal(first.batch.theta, other.batch.theta)
    assert not torch.equal(first.batch.x, other.batch.x)

    path = first.save(tmp_path / "episodes.pt")
    restored = pb.data.load(path)
    assert torch.equal(first.batch.theta, restored.batch.theta)
    assert torch.equal(first.batch.mask, restored.batch.mask)
    assert restored.batch.meta["epoch"] == 2
    assert path.with_suffix(".pt.json").exists()


def test_loader_order_is_epoch_deterministic():
    """Check that loader order repeats within an epoch and changes across epochs."""
    env = pb.make("location-v0", budget=2)
    dataset = pb.data.Stream(env, episodes=20, seed=9).generate(None)
    order_a = torch.cat([batch.theta for batch in pb.data.loader(dataset, 5, seed=11, epoch=3)])
    order_b = torch.cat([batch.theta for batch in pb.data.loader(dataset, 5, seed=11, epoch=3)])
    order_c = torch.cat([batch.theta for batch in pb.data.loader(dataset, 5, seed=11, epoch=4)])
    assert torch.equal(order_a, order_b)
    assert not torch.equal(order_a, order_c)


def test_context_replay_and_inverse_pairs():
    """Check batched context, replayed particles, and inverse data without designs."""
    env = pb.make("location-v0", sources=(1, 2), dims=(1, 3), budget=2)
    dataset = pb.data.Stream(env, episodes=6, seed=4).generate()
    assert dataset.batch.context["theta_mask"].shape == (6, 2, 3)
    loaded = next(iter(pb.data.loader(dataset, 3, shuffle=False)))
    assert loaded.context["source_dim"].shape == (3,)

    loaded.belief = pb.ParticleCloud(torch.randn(3, 8, 2, 3))
    replay = pb.data.ReplayBuffer(4)
    replay.add(loaded)
    sampled = replay.sample(2, seed=2)
    assert sampled.belief.points.shape == (2, 8, 2, 3)

    inverse = pb.envs.inverse(
        prior=pb.Normal(torch.zeros(2), torch.ones(2)),
        simulator=lambda theta: theta.square(),
        theta_shape=(2,),
        y_shape=(2,),
    )
    pairs = pb.data.Stream(inverse, episodes=5, seed=3).generate()
    assert pairs.batch.x is None
    assert pairs.batch.o.x is None
    assert pairs.batch.y.shape == (5, 2)


def test_nonadaptive_stream_uses_one_simulator_call():
    """Check that fixed non-adaptive designs are simulated in one vectorized call."""
    calls = []

    def simulator(theta, x):
        """Take scalar parameters and designs, record one call, and return their sum."""
        calls.append((theta.shape, x.shape))
        return theta + x

    env = pb.BED(
        "vectorized",
        pb.Normal(torch.zeros(1), torch.ones(1)),
        simulator,
        {"theta": pb.Spec((1,)), "x": pb.Spec((1,), low=0, high=1), "y": pb.Spec((1,))},
        budget=3,
    )
    x = torch.tensor([[0.1], [0.2], [0.3]])
    batch = pb.data.Stream(env, episodes=4).generate(x=x).batch
    assert len(calls) == 1
    assert batch.x.shape == batch.y.shape == (4, 3, 1)


def test_joint_particle_rollout_and_candidate_pool():
    """Check one-pass inference/design state updates and random candidate-set designs."""
    calls = []

    class Prior:
        """Return truths together with an initial two-particle belief."""

        def sample(self, shape, generator=None):
            """Take a leading shape and return truth and belief batches."""
            del generator
            return pb.Batch(
                theta=torch.zeros(*shape, 1),
                belief=pb.ParticleCloud(torch.zeros(*shape, 2, 1)),
            )

    def infer_design(history):
        """Take a history and return its updated particles and one next design."""
        calls.append(len(history))
        cloud = pb.ParticleCloud(history.belief.points + 1)
        return cloud, torch.full((len(history), 1), 0.5)

    env = pb.BED(
        "joint-particles",
        Prior(),
        lambda theta, x: theta + x,
        {"theta": pb.Spec((1,)), "x": pb.Spec((1,), low=0, high=1), "y": pb.Spec((1,))},
        budget=3,
        joint_infer_design=infer_design,
    )
    batch = pb.data.Stream(env, episodes=4).generate().batch
    assert calls == [4, 4, 4]
    assert torch.equal(batch.belief.points, torch.full((4, 2, 1), 3.0))

    pool = torch.tensor([[0.1], [0.3], [0.8]])
    candidate_env = env.with_components(infer_design=None, candidates=pool)
    candidate_batch = pb.data.Stream(candidate_env, episodes=5).generate().batch
    assert torch.isin(candidate_batch.x, pool).all()

    scalar_env = pb.BED(
        "scalar-candidates",
        pb.Normal(0.0, 1.0),
        lambda theta, x: theta + x,
        {"theta": pb.Spec(()), "x": pb.Spec(()), "y": pb.Spec(())},
        candidate_pool=torch.tensor([0.1, 0.5, 0.9]),
    )
    scalar_x = pb.data.random_policy(None, env=scalar_env, batch_size=4).x
    assert scalar_x.shape == (4,)

    def sampled_policy(history, **context):
        """Take a history and return sampled designs with their score terms."""
        size = len(history)
        return pb.PolicySample(torch.zeros(size, 1), torch.ones(size), torch.full((size,), 2.0))

    sampled = pb.data.Stream(candidate_env, episodes=5, budget=2).generate(sampled_policy).batch
    assert sampled.context["log_prob"].shape == (5, 2)
    assert torch.equal(sampled.context["entropy"], torch.full((5, 2), 2.0))
