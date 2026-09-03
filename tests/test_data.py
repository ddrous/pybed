import torch

import pybed as pb


def test_epoch_reproducibility_and_common_truths(tmp_path):
    env = pb.make("location-v0", budget=4)
    stream = pb.data.EpochStream(env, episodes=24, seed=17)
    first = stream.generate(None, epoch=2)
    second = stream.generate(None, epoch=2)
    assert torch.equal(first.batch.theta, second.batch.theta)
    assert torch.equal(first.batch.design, second.batch.design)
    assert torch.equal(first.batch.obs, second.batch.obs)

    def center(history, **context):
        return torch.full((len(history), 2), 0.5)

    other = stream.generate(center, epoch=2)
    assert torch.equal(first.batch.theta, other.batch.theta)
    assert not torch.equal(first.batch.design, other.batch.design)

    path = first.save(tmp_path / "episodes.pt")
    restored = pb.data.load(path)
    assert torch.equal(first.batch.theta, restored.batch.theta)
    assert torch.equal(first.batch.mask, restored.batch.mask)
    assert restored.batch.meta["epoch"] == 2
    assert path.with_suffix(".pt.json").exists()


def test_loader_order_is_epoch_deterministic():
    env = pb.make("location-v0", budget=2)
    dataset = pb.data.EpochStream(env, episodes=20, seed=9).generate(None)
    order_a = torch.cat([batch.theta for batch in pb.data.loader(dataset, 5, seed=11, epoch=3)])
    order_b = torch.cat([batch.theta for batch in pb.data.loader(dataset, 5, seed=11, epoch=3)])
    order_c = torch.cat([batch.theta for batch in pb.data.loader(dataset, 5, seed=11, epoch=4)])
    assert torch.equal(order_a, order_b)
    assert not torch.equal(order_a, order_c)
