"""Per-recording data locations for the private legacy solver process."""
import json
import os
from pathlib import Path
import sys
import yaml
assets=Path(__file__).resolve().parents[3]/'assets'
ego=assets.parents[2]
project=ego.parent
with Path(os.environ['LABEL_PIPELINE_CASE']).open() as stream:
    config=json.load(stream)
recording=config['id'];root=Path(config['recording_output'])
optimized=root/'native/optimized_stride4'
params=yaml.safe_load((assets.parent/'defaults.yaml').read_text())
overrides=params.get('legacy_recording_overrides',{}).get(recording,{})
for name,value in overrides.items():
    params[name]={**params[name],**value} if isinstance(value,dict) else value
for name,value in config.get('parameters',{}).items():
    params[name]={**params[name],**value} if isinstance(value,dict) else value
placement=(project/config['marker_placement']).resolve()
sys.path[:0]=[str(ego/'scripts/joint_refit'),str(ego/'scripts')]
def stage(name):
    return root/'legacy_stages'/name
