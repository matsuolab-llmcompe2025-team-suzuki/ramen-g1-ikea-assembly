"""Runtime C++ library selection; no imports of the robot SDK."""

import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("previous", [None, "", "/usr/local/cuda/lib64", "/test/pixi/lib:/custom/lib"])
def test_runtime_activation_preserves_paths_and_is_idempotent(previous):
    script = Path("scripts/activate_runtime.sh").resolve()
    env = dict(os.environ, CONDA_PREFIX="/test/pixi")
    if previous is None:
        env.pop("LD_LIBRARY_PATH", None)
    else:
        env["LD_LIBRARY_PATH"] = previous
    proc = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c",
         'source "$1"; source "$1"; printf "%s" "$LD_LIBRARY_PATH"', "test", str(script)],
        env=env, check=True, capture_output=True, text=True,
    )
    expected = previous if (previous or "").startswith("/test/pixi/lib:") else "/test/pixi/lib" + (":" + previous if previous else "")
    assert proc.stdout == expected
