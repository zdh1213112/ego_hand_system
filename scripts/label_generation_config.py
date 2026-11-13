"""Explicit opt-in configuration for native multiview MANO refinement."""
import json,math
from pathlib import Path


def load_label_generation_config(path=None):
    default={'schema_version':1,'multiview_mano_optimization':{'enabled':False}}
    if path is None:return default
    path=Path(path).resolve();data=json.loads(path.read_text())
    if not isinstance(data,dict) or set(data)-{'schema_version','multiview_mano_optimization'}:raise ValueError('Unknown label generation configuration field')
    if type(data.get('schema_version')) is not int or data['schema_version']!=1:raise ValueError('Expected schema_version=1')
    section=data.get('multiview_mano_optimization',{})
    allowed={'enabled','bundle','initial_fit','rgb_weight','sigma_px','pose_iterations'}
    if not isinstance(section,dict) or set(section)-allowed:raise ValueError('Unknown multiview MANO configuration field')
    enabled=section.get('enabled',False)
    if type(enabled) is not bool:raise ValueError('enabled must be a JSON boolean, not a string or number')
    if not enabled:return default
    resolved={'enabled':True}
    for name in ['bundle','initial_fit']:
        value=section.get(name)
        if not isinstance(value,str) or not value.strip():raise ValueError(f'enabled=true requires {name}')
        value=Path(value).expanduser();resolved[name]=str((path.parent/value).resolve() if not value.is_absolute() else value.resolve())
    for name,default_value in [('rgb_weight',.25),('sigma_px',4.)]:
        value=section.get(name,default_value)
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError(f'{name} must be positive and finite')
        resolved[name]=float(value)
    value=section.get('pose_iterations',200)
    if type(value) is not int or value<1:raise ValueError('pose_iterations must be a positive integer')
    resolved['pose_iterations']=value
    return {'schema_version':1,'multiview_mano_optimization':resolved}


def require_available_mode(config):
    section=config['multiview_mano_optimization']
    if not section['enabled']:return
    import numpy as np
    from multiview_mano_constraints import sha256
    path=Path(section['bundle'])
    if not path.is_file():raise FileNotFoundError(f'Missing independent RGB/marker attachment bundle: {path}')
    with np.load(path,allow_pickle=False) as bundle:
        if str(bundle['schema'])!='native_multiview_mano_v1' or str(bundle['observation_source'])!='independent_rgb_marker_detections':raise ValueError('Require independent RGB marker bundle')
        expected=json.loads(str(bundle['baseline_fit_sha256_json']))
    for side in (0,1):
        track=Path(section['initial_fit'])/f'track_{side}.npz'
        if not track.is_file():raise FileNotFoundError(track)
        if sha256(track)!=expected[str(side)]:raise ValueError('Initial MANO fit differs from attachment calibration baseline')
