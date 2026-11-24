"""Resolve the signed export configuration and fingerprint the code and assets actually used."""
import copy
import hashlib
import json
import math
from pathlib import Path

import yaml


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def merge(base, overrides, prefix=''):
    result = copy.deepcopy(base)
    changes = {}
    if not isinstance(overrides, dict):
        raise ValueError(f'{prefix} expects a JSON object')
    for key, value in overrides.items():
        path = f'{prefix}.{key}' if prefix else key
        if key not in base:
            raise ValueError(f'Unknown configuration field {path}')
        old = base[key]
        if isinstance(old, dict):
            result[key], nested = merge(old, value, path)
            changes.update(nested)
            continue
        def typed(a,b):
            if type(a) is float:
                return type(b) in (int,float) and math.isfinite(b)
            if isinstance(a,list):
                return isinstance(b,list) and len(b)==len(a) and all(typed(x,y) for x,y in zip(a,b))
            return type(a) is type(b)
        if not typed(old,value):
            raise ValueError(f'Invalid type or dimensions for {path}')
        if old!=value:
            changes[path] = {'signed':old,'effective':value}
        result[key]=value
    return result, changes


def provenance(repo, cfg, name, hands, weights, observation, seed, placement, single):
    pipe=repo/'tools/label_pipeline'
    default_path=pipe/'defaults.yaml'
    defaults=yaml.safe_load(default_path.read_text())
    baseline=json.loads((pipe/'assets/signed_cli_profile.json').read_text())
    raw={k:v for k,v in cfg.items() if k!='overrides'}
    _, changes=merge(baseline['label_pipeline'],raw,'label_pipeline')
    overrides=cfg.get('overrides',{})
    if not isinstance(overrides,dict) or set(overrides)-{'parameters','single_hand','runtime'}:
        raise ValueError('overrides accepts parameters, single_hand, and runtime')
    per_rec=defaults.get('legacy_recording_overrides',{}).get(name,{})
    effective,_=merge(defaults,per_rec)
    editable={k:v for k,v in effective.items() if k not in ('fit_method','legacy_recording_overrides','native_tool_roots','training_mano_directory','holdout_by_time')}
    params,delta=merge(editable,overrides.get('parameters',{}),'parameters')
    changes.update(delta)
    effective.update(params)
    signed_single=baseline['single_hand_config']
    actual=json.loads(single.read_text())['joint_refit']
    temporal,sd=merge(signed_single,actual,'single_hand')
    changes.update(sd)
    temporal,sd=merge(temporal,overrides.get('single_hand',{}),'single_hand')
    changes.update(sd)
    runtime,rd=merge(baseline['runtime'],overrides.get('runtime',{}),'runtime')
    changes.update(rd)
    if len(hands)==1 and overrides.get('parameters'):
        raise ValueError('Use overrides.single_hand for a single-hand fit')
    if len(hands)==2 and overrides.get('single_hand'):
        raise ValueError('single_hand overrides do not apply to a two-hand recording')
    if not 0 < params['train_ratio'] < 1 or not 1 <= params['v4']['optics']['min_views'] <= 4:
        raise ValueError('Invalid training ratio or optical camera count')
    numeric=params['v4']['fit']; opt=params['v4']['optics']
    if min(numeric['batch_size'],numeric['finite_difference_deg'],opt['workers'],opt['denominator_floor'])<=0:
        raise ValueError('Batch size, finite difference, worker count and denominator floor must be positive')
    if min(runtime.values())<=0:
        raise ValueError('Runtime parameters must be positive')
    manifests={}
    roots=[pipe,repo/'scripts',repo/'third_party/MANO']
    for key in defaults['native_tool_roots'][:4]:
        roots.append(repo.parent/key)
    for base in roots:
        for path in sorted(base.rglob('*')):
            if path.suffix in ('.py','.cpp','.sh') and path.is_file():
                manifests[str(path.relative_to(repo.parent))]=sha(path)
    payload=json.dumps(manifests,sort_keys=True,separators=(',',':')).encode()
    active=len(hands)==2 and observation is None
    model_files={k:dict(path=str(p),sha256=sha(p) if p.is_file() else None,used=active) for k,p in weights.items()}
    return dict(defaults=dict(path=str(default_path),sha256=sha(default_path),values=defaults),
                effective_parameters=effective,single_hand_parameters=temporal,runtime_parameters=runtime,
                signed_parameter_differences=changes,observation_weights=model_files,
                external_observation_sha256=sha(observation) if observation else None,
                asset_sha256={str(p):sha(p) for p in (seed,placement,single,repo/'models/mano/MANO_RIGHT.pkl')},
                code_sha256=hashlib.sha256(payload).hexdigest(),code_files=manifests)
