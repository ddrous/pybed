import os
import importlib.util
import subprocess
import sys
from pathlib import Path


def test_actionbed_quick_run(tmp_path):
    """Run the short Action-BED example and check that it saves a summary."""
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env.update(
        PYBED_QUICK="1",
        MPLBACKEND="Agg",
        PYTHONPATH=str(root) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "actionbed" / "actionbed.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert list((tmp_path / "runs" / "actionbed").glob("*/summary.json"))


def test_inverse_quick_run(tmp_path):
    """Run the short inverse-transport example and check that it completes."""
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env.update(PYBED_QUICK="1", MPLBACKEND="Agg", PYTHONPATH=str(root))
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "inverse" / "energy_transport.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "posterior_mean" in result.stdout


def test_dad_lightning_wandb_quick_run(tmp_path):
    """Run the short DAD integration when its optional packages are installed."""
    if importlib.util.find_spec("lightning") is None or importlib.util.find_spec("wandb") is None:
        return
    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env.update(
        PYBED_QUICK="1",
        PYBED_RUNS=str(tmp_path / "runs"),
        WANDB_MODE="disabled",
        MPLBACKEND="Agg",
        PYTHONPATH=str(root),
    )
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "dad" / "dad_lightning_wandb.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert (tmp_path / "runs" / "dad-lightning-wandb" / "last.ckpt").exists()
