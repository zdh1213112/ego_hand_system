"""Exercise the public launcher and the package boundary without model assets."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_public_launcher_help_does_not_require_conda_or_models():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_hand_inference.sh"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--input", "--session", "--mcap", "--shared-steps"):
        assert option in result.stdout


def test_stage_module_resolution_works_without_inherited_pythonpath():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    # Argument help exercises package execution without optional dependencies.
    code = (
        "import sys; sys.path.insert(0, 'scripts'); "
        "from ego_hand_inference.run_hand_inference import run_stage; "
        "run_stage('run_hand_inference.py', ['--help'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--max-delta-us" in result.stdout
