"""The public entry must pass explicit overrides through and preserve signed defaults."""
from pathlib import Path
import json
import sys
from unittest.mock import patch

import pytest
import yaml

from test_signed_label_export import REPO, paired, public_cli
from label_export_settings import merge, sha


def test_configuration_overrides_are_persisted_and_forwarded(tmp_path, monkeypatch, capsys):
    root = paired(tmp_path, ["left", "right"])
    baseline = REPO / "configs/native_nokov_v4.json"
    config = json.loads(baseline.read_text())
    profile = config["label_pipeline"]
    for key in ("seed_alignment", "marker_placement", "single_hand_config"):
        profile[key] = str(baseline.parent / profile[key])
    profile["overrides"] = {
        "parameters": {"train_ratio": 0.8, "v4": {"fit": {"max_tip_displacement_mm": 2.5}}}
    }
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps(config))
    obs = tmp_path / "obs.npz"
    obs.touch()
    out = tmp_path / "result"
    with patch("signed_label_export.subprocess.run") as launch:
        public_cli(monkeypatch, root, out, "--optimization-config", custom, "--wilor-observations", obs)
    cfg = json.loads((out / "settings.json").read_text())
    assert cfg["effective_parameters"]["v4"]["fit"]["max_tip_displacement_mm"] == 2.5
    assert cfg["signed_parameter_differences"]["parameters.v4.fit.max_tip_displacement_mm"] == {
        "signed": 3.0, "effective": 2.5
    }
    assert "parameters.v4.fit.max_tip_displacement_mm" in capsys.readouterr().out
    assert cfg["defaults"]["sha256"] == sha(Path(cfg["defaults"]["path"]))
    assert yaml.safe_load((out / "recording.yaml").read_text())["parameters"]["train_ratio"] == 0.8
    command = launch.call_args_list[-1].args[0]
    assert command[command.index("--settings") + 1] == str(out / "settings.json")


@pytest.mark.parametrize("value", [True, float("nan"), "3"])
def test_numeric_legacy_overrides_reject_ambiguous_types(value):
    with pytest.raises(ValueError):
        merge({"limit": 3.0}, {"limit": value})


def test_unknown_config_field_and_cli_fit_override_fail(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Unknown"):
        merge({"fit": {"limit": 3.0}}, {"fit": {"typo": 3.0}})
    with pytest.raises(SystemExit):
        public_cli(monkeypatch, paired(tmp_path, ["right"]), tmp_path/"out", "--sample-stride", "2")
