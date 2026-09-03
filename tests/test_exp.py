import torch

import pybed as pb


def test_run_checkpoint_and_compare(tmp_path):
    run = pb.exp.Run.create(tmp_path, "unit", {"width": 2}, run_id="fixed", seed_value=3)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint = run.checkpoint("last", model=model, optimizer=optimizer, step=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    payload = run.restore(checkpoint, model=model, optimizer=optimizer)
    assert payload["step"] == 4
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert run.log(4, loss=1.2)["loss"] == 1.2
    assert run.finish(done=True).exists()

    env = pb.make("location-v0", budget=2)
    results, datasets = pb.exp.compare(
        env,
        {"a": None, "b": None},
        lambda candidate, batch: {"mean": batch.obs.mean()},
        episodes=8,
        seed_value=2,
    )
    assert results.keys() == datasets.keys()
    assert torch.equal(datasets["a"].batch.theta, datasets["b"].batch.theta)
