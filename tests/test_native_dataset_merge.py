import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from merge_wilor_training_datasets import main


def source(root, name, focal, space="native_raw"):
    root = root / name
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "images/0.jpg").write_bytes(b"image-" + name.encode())
    (root / "labels/0.npy").write_bytes(b"label-" + name.encode())
    summary = dict(schema_version=6, sample_count=1, image_space=space,
        geometry_space="physical_side", mano_parameter_space="right_canonical",
        side_semantics="physical_identity", mano_pose_representation="rotation_matrix_full_mean",
        mano_assets={"model": "same-hash"}, mano_model_by_side={"left": "MANO_RIGHT.pkl", "right": "MANO_RIGHT.pkl"},
        camera_calibrations={"camera0": {"K": [[focal,0,10],[0,focal,10],[0,0,1]]}},
        image_size=[20,20])
    (root / "summary.json").write_text(json.dumps(summary))
    row=dict(index=0, image="images/0.jpg", label="labels/0.npy", side=0,
        camera="camera0", source_group=name, stored_image_space=space)
    (root / "index.jsonl").write_text(json.dumps(row) + "\n")
    return root


def test_merge_keeps_different_calibrations_for_same_camera_name(tmp_path, monkeypatch):
    first, second = source(tmp_path,"recording_a",300), source(tmp_path,"recording_b",500)
    dest=tmp_path/"merged"
    monkeypatch.setattr(sys,"argv",["merge","--inputs",str(first),str(second),"--output",str(dest)])
    assert main()==0
    summary=json.loads((dest/"summary.json").read_text())
    assert summary["schema_version"]==6
    assert summary["camera_calibrations_by_source"]["0"]["camera0"]["K"][0][0]==300
    assert summary["camera_calibrations_by_source"]["1"]["camera0"]["K"][0][0]==500
    rows=[json.loads(s) for s in (dest/"index.jsonl").read_text().splitlines()]
    assert [r["calibration_group"] for r in rows]==["0","1"]
    assert [r["source_group"] for r in rows]==["recording_a","recording_b"]
    assert (dest/rows[1]["label"]).read_bytes()==(second/"labels/0.npy").read_bytes()


def test_merge_rejects_native_and_rectified_mixture(tmp_path, monkeypatch):
    first=source(tmp_path,"raw",300)
    second=source(tmp_path,"rect",300,"physical_rectified")
    dest=tmp_path/"merged"
    monkeypatch.setattr(sys,"argv",["merge","--inputs",str(first),str(second),"--output",str(dest)])
    with pytest.raises(ValueError,match="image_space"):
        main()
    assert not dest.exists()
