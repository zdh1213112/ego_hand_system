import json,subprocess,sys
from pathlib import Path
import pytest
PROJECT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(PROJECT/'scripts'))
from personal_marker24_config import load_personal_config
PAIR=PROJECT.parent/'data/20261111/20261111_nokov_hand/pair_20261111_192916'
ENTRY=PROJECT/'label_export/export_ego_nokov_labels.py'

def write_config(tmp_path,enabled=True):
 p=tmp_path/'config.json';placement=tmp_path/'placement.json';placement.write_text('{}');p.write_text(json.dumps({'schema_version':1,'personal_marker24_optimization':{'enabled':enabled,'placement':'placement.json','dorsal_reference':{'left':{'camera':'camera1','sync':616},'right':{'camera':'camera2','sync':584}}}}));return p

def dry(tmp_path,config=None,extras=()):
 out=tmp_path/'never_created';cmd=[sys.executable,str(ENTRY),'--recording',str(PAIR),'--output',str(out),'--dry-run',*extras]
 if config:cmd+=['--optimization-config',str(config)]
 r=subprocess.run(cmd,text=True,capture_output=True);assert not out.exists();return r

def test_disabled_is_identical_to_baseline(tmp_path):
 a=dry(tmp_path);b=dry(tmp_path,write_config(tmp_path,False));assert a.returncode==b.returncode==0
 assert a.stdout==b.stdout
 assert 'personal_marker24' not in a.stdout

def test_enabled_plans_connected_stages_without_writes(tmp_path):
 r=dry(tmp_path,write_config(tmp_path));assert r.returncode==0,r.stderr
 stages=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')]
 by={x['stage']:x['command'] for x in stages}
 assert '--source' in by['native_contours'] and str(tmp_path/'never_created/optimized/marker24') in by['native_contours']
 assert str(tmp_path/'never_created/optimized/marker24') in by['contour_shape_refinement']
 assert str(tmp_path/'never_created/optimized/contour_fit') in by['materialize_optimized_fit']
 assert 'export_check_preview' in by and 'videos' in by
 assert not any(x['stage'].startswith('diagnostic_') for x in stages)

@pytest.mark.parametrize('flag',['--joint','--observations-only'])
def test_old_and_new_modes_cannot_mix(tmp_path,flag):
 r=dry(tmp_path,write_config(tmp_path),[flag]);assert r.returncode!=0 and 'cannot be combined' in r.stderr

@pytest.mark.parametrize('key,value',[('enabled','true'),('calibration_steps',0),('diagnostics',1),('holdout',[704,512]),('pose_scope','everything')])
def test_invalid_settings_rejected(tmp_path,key,value):
 p=write_config(tmp_path);d=json.loads(p.read_text());d['personal_marker24_optimization'][key]=value;p.write_text(json.dumps(d))
 with pytest.raises(ValueError):load_personal_config(p)

def test_paths_and_per_hand_shape_contract(tmp_path):
 p=write_config(tmp_path);cfg=load_personal_config(p);assert cfg['placement']==str(tmp_path/'placement.json');assert cfg['pose_scope']=='sampled_sequence';assert cfg['shape_sharing']=='one_beta_vector_per_physical_hand_across_frames_and_cameras'
