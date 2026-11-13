import importlib.util
from pathlib import Path
import subprocess
import sys
import pytest

SPEC = importlib.util.spec_from_file_location('raw_label_entry', Path(__file__).resolve().parents[1]/'scripts/export_ego_nokov_labels.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def inputs(tmp_path):
    pair=tmp_path/'pair';ego=pair/'EGO_take';part=ego/'part0001';part.mkdir(parents=True)
    for stream in ['left','right','right_side','left_side']:
        for ext in ['.mp4','_pts.csv']:
            (part/f'take_camera_{stream}_part0001{ext}').touch()
    (part/'take_imu_part0001.csv').touch()
    (ego/'take_calibration_camera.yaml').touch();(ego/'meta_take.json').touch()
    nokov=pair/'session_take/nokov';nokov.mkdir(parents=True);(nokov/'nokov_rigid_bodies.csv').touch()
    seed=tmp_path/'seed.json';seed.write_text('{}')
    return pair,seed


def test_discovery_rejects_ambiguity(tmp_path):
    pair,seed=inputs(tmp_path)
    assert m.resolve_sources(pair)[0].name=='EGO_take'
    (pair/'EGO_second').mkdir()
    with pytest.raises(ValueError,match='Expected one'):
        m.resolve_sources(pair)


def test_dry_run_does_not_write_or_execute(tmp_path,monkeypatch):
    pair,seed=inputs(tmp_path);out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['entry','--recording',str(pair),'--seed-alignment',str(seed),'--output',str(out),'--dry-run','--joint'])
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:pytest.fail('dry run executed a process'))
    m.main();assert not out.exists()


def test_baseline_sequence_and_failure_stop(tmp_path,monkeypatch):
    pair,seed=inputs(tmp_path);calls=[]
    monkeypatch.setattr(sys,'argv',['entry','--recording',str(pair),'--seed-alignment',str(seed),'--output',str(tmp_path/'out')])
    def fake(cmd,**kwargs):
        calls.append(Path(cmd[2]).name)
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(m.subprocess,'run',fake);m.main()
    assert calls==['normalize_orbbec_ego4_session.py','synchronize_ego_imu_nokov.py','align_native_dual_nokov.py','native_fourview_alignment.py','run_nokov_native_labels.py','render_wilor_training_videos.py']
    calls.clear();sys.argv[sys.argv.index('--output')+1]=str(tmp_path/'failed')
    def fail(cmd,**kwargs):
        calls.append(cmd);return subprocess.CompletedProcess(cmd,1)
    monkeypatch.setattr(m.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='normalize failed'):m.main()
    assert len(calls)==1


def test_joint_builds_zero_offset_bundle_and_both_videos(tmp_path,monkeypatch):
    import json
    import numpy as np
    pair,seed=inputs(tmp_path);out=tmp_path/'joint_run';calls=[]
    monkeypatch.setattr(sys,'argv',['entry','--recording',str(pair),'--seed-alignment',str(seed),'--output',str(out),'--joint'])
    def fake(cmd,**kwargs):
        calls.append(Path(cmd[2]).name)
        if calls[-1]=='prepare_multiview_mano_bundle.py':
            np.savez(Path(cmd[cmd.index('--output')+1]),offset_m=np.full((2,21),.002),pixels=np.arange(10))
        return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(m.subprocess,'run',fake);m.main()
    with np.load(out/'constraints.npz') as b:
        assert np.all(b['offset_m']==0)
        np.testing.assert_equal(b['pixels'],np.arange(10))
    assert calls.count('run_nokov_native_labels.py')==2
    assert calls.count('render_wilor_training_videos.py')==2
    assert json.loads((out/'joint_config.json').read_text())['multiview_mano_optimization']['enabled'] is True


def test_observation_only_stops_before_mano(tmp_path,monkeypatch):
    pair,seed=inputs(tmp_path);calls=[]
    monkeypatch.setattr(sys,'argv',['entry','--recording',str(pair),'--seed-alignment',str(seed),'--output',str(tmp_path/'observations'),'--observations-only'])
    def fake(cmd,**kwargs):
        calls.append(Path(cmd[2]).name);return subprocess.CompletedProcess(cmd,0)
    monkeypatch.setattr(m.subprocess,'run',fake);m.main()
    assert calls[-1]=='audit_native_multiview_observations.py'
    assert 'run_nokov_native_labels.py' not in calls
