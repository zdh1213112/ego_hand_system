import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from export_ego_nokov_labels import main


def paired(tmp_path, hands):
    root = tmp_path / "recording"
    ego = root / "EGO-pair"
    nokov = root / "session1/nokov"
    (ego / "part0001").mkdir(parents=True)
    nokov.mkdir(parents=True)
    (ego / "part0001/test.mcap").touch()
    (ego / "test_calibration_camera.yaml").touch()
    for filename in ("nokov_rigid_bodies.csv", "nokov_markers.csv", "nokov_skeleton_segments.csv"):
        (nokov / filename).touch()
    (nokov / "asset_descriptions.json").write_text(
        json.dumps({"marker_sets": {s + "_hand_4mm": [] for s in hands}})
    )
    return root


def public_cli(monkeypatch, root, output, *options):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_ego_nokov_labels.py",
            "--recording",
            str(root),
            "--output",
            str(output),
            "--optimization-config",
            str(REPO / "configs/native_nokov_v4.json"),
            *map(str, options),
        ],
    )
    main()


@pytest.mark.parametrize("side", ["left", "right"])
def test_single_hand_dispatches_to_native_temporal(tmp_path, monkeypatch, side):
    root = paired(tmp_path, [side])
    out = tmp_path / "output"
    with patch("signed_label_export.subprocess.run") as launch:
        public_cli(monkeypatch, root, out)
    convert, fit = launch.call_args_list
    assert Path(convert.args[0][2]).name == "convert.py"
    command = fit.args[0]
    assert command[command.index("--optimization-config") + 1] == str(
        REPO / "label_export/configs/native_nokov_temporal.json"
    )
    assert json.loads((out / "settings.json").read_text())["mode"] == "single_hand_temporal"


def test_dual_hand_respects_cached_observations_and_output(tmp_path, monkeypatch):
    root = paired(tmp_path, ["left", "right"])
    out = tmp_path / "output"
    existing = tmp_path / "observed.npz"
    existing.touch()
    with patch("signed_label_export.subprocess.run") as launch:
        public_cli(monkeypatch, root, out, "--dry-run", "--wilor-observations", existing)
        assert not out.exists()
        launch.assert_not_called()
        public_cli(monkeypatch, root, out, "--wilor-observations", existing)
        fit, materialize = launch.call_args_list
        assert Path(fit.args[0][2]).name == "export_glove_labels.py"
        assert Path(materialize.args[0][2]) == REPO / "label_export/pipeline/v4/export.py"
        assert Path(materialize.args[0][-1]) == out / "optimized"
        cfg = yaml.safe_load((out / "recording.yaml").read_text())
        assert cfg["native_wilor"] == str(existing) and cfg["wrist_policy"] == "adaptive_measured"
        with pytest.raises(SystemExit):
            public_cli(monkeypatch, root, out, "--wilor-observations", existing)
        assert launch.call_count == 2
