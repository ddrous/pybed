import os
import subprocess
import sys
from pathlib import Path


def test_actionbed_quick_run(tmp_path):
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
