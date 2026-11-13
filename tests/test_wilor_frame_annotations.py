from pathlib import Path
import json
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from wilor_frame_annotations import assign_image_paths, require_both_hands, write_frame_index


def test_two_hands_share_image_but_keep_two_labels(tmp_path):
    items=[dict(camera="camera0",sync_index=10,side=s) for s in (0,1)]
    assign_image_paths(items,True)
    assert items[0]["image_stem"]==items[1]["image_stem"]
    rows=[dict(item,image="images/"+item["image_stem"]+".jpg",label=f"labels/{i}.npy",index=i) for i,item in enumerate(items)]
    write_frame_index(tmp_path/"frames.jsonl",rows)
    frames=[json.loads(x) for x in (tmp_path/"frames.jsonl").read_text().splitlines()]
    assert len(frames)==1 and frames[0]["hand_count"]==2
    assert [h["side"] for h in frames[0]["hands"]]==[0,1]
    assert frames[0]["hands"][0]["label"]!=frames[0]["hands"][1]["label"]


def test_missing_hand_drops_camera_frame_only():
    items=[dict(camera="camera0",sync_index=10,side=s) for s in (0,1)]
    items.append(dict(camera="camera1",sync_index=10,side=0))
    accepted,rejected=require_both_hands(items)
    assert len(accepted)==2
    assert rejected[0]["camera"]=="camera1"
    assert rejected[0]["reason"]=="both_hands_required"


def test_duplicate_identity_is_rejected():
    items=[dict(camera="camera0",sync_index=10,side=0)]*2
    with pytest.raises(ValueError,match="duplicate"):
        require_both_hands(items)


def test_different_camera_frames_keep_different_images():
    items=[dict(camera=c,sync_index=10,side=0) for c in ("camera0","camera1")]
    assign_image_paths(items,True)
    assert items[0]["image_stem"]!=items[1]["image_stem"]


def test_dual_pipeline_dry_run_selects_both_tracks(tmp_path, monkeypatch, capsys):
    import numpy as np
    import run_nokov_native_labels as runner
    session=tmp_path/"session"
    dataset=session/"ego/normalized_multiview"
    dataset.mkdir(parents=True)
    (dataset/"multiview_frames.csv").write_text("reference_timestamp_ns\n1\n")
    aligned=tmp_path/"synthetic_plan_fixture.npz"
    np.savez(aligned,valid=np.ones((2,1,21),dtype=bool))
    output=tmp_path/"output"
    monkeypatch.setattr(sys,"argv",["run","--processed-session",str(session),"--aligned-nokov",str(aligned),"--output",str(output),"--dry-run"])
    runner.main()
    commands=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    fit=next(c["command"] for c in commands if c["stage"]=="fit")
    assert [fit[i+1] for i,v in enumerate(fit) if v=="--track-id"]==["0","1"]
    assert "--require-both-hands" in commands[-1]["command"]
    assert any("prepare_native_nokov_mano_input.py" in part for part in commands[0]["command"])
    assert not any("rectification" in part for command in commands for part in command["command"])
    assert not output.exists()


def test_dual_pipeline_rejects_one_observed_hand(tmp_path, monkeypatch):
    import numpy as np
    import run_nokov_native_labels as runner
    aligned=tmp_path/"one_hand.npz"
    valid=np.zeros((2,1,21),dtype=bool);valid[0]=True
    np.savez(aligned,valid=valid)
    monkeypatch.setattr(sys,"argv",["run","--processed-session",str(tmp_path),"--aligned-nokov",str(aligned),"--output",str(tmp_path/"output"),"--dry-run"])
    with pytest.raises(ValueError,match="Both physical hand tracks"):
        runner.main()
