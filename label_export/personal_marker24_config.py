"""Strict opt-in config for the shared personal-shape/24-marker pipeline."""
import json
import math
from pathlib import Path

def load_personal_config(path=None):
 if path is None:return {'enabled':False}
 p=Path(path).expanduser().resolve();data=json.loads(p.read_text())
 if not isinstance(data,dict) or set(data)-{'schema_version','personal_marker24_optimization','joint_refit'} or type(data.get('schema_version')) is not int or data['schema_version']!=1:raise ValueError('Expected schema_version=1 and personal_marker24_optimization or joint_refit')
 s=data.get('personal_marker24_optimization',{})
 allowed={'enabled','placement','dorsal_reference','holdout','calibration_frames','calibration_steps','contour_steps','pose_scope','diagnostics','diagnostic_steps','export_optimized','joint_pose_attachments'}
 if not isinstance(s,dict) or set(s)-allowed:raise ValueError('Unknown personal_marker24_optimization setting')
 enabled=s.get('enabled',False)
 if type(enabled) is not bool:raise ValueError('enabled must be a JSON boolean')
 if not enabled:return {'enabled':False}
 result={'enabled':True,'shape_sharing':'one_beta_vector_per_physical_hand_across_frames_and_cameras'}
 value=s.get('placement')
 if not isinstance(value,str) or not value.strip():raise ValueError('enabled=true requires placement JSON')
 q=Path(value).expanduser();result['placement']=str((p.parent/q).resolve() if not q.is_absolute() else q.resolve())
 if not Path(result['placement']).is_file():raise FileNotFoundError(result['placement'])
 for name,default,minimum in [('calibration_frames',48,8),('calibration_steps',120,10),('contour_steps',120,1),('diagnostic_steps',100,1)]:
  value=s.get(name,default)
  if type(value) is not int or value<minimum:raise ValueError(f'{name} must be an integer >= {minimum}')
  result[name]=value
 for name,default in [('diagnostics',False),('export_optimized',True)]:
  value=s.get(name,default)
  if type(value) is not bool:raise ValueError(f'{name} must be a JSON boolean')
  result[name]=value
 scope=s.get('pose_scope','sampled_sequence')
 if scope not in ['holdout','sampled_sequence']:raise ValueError('Invalid pose_scope')
 result['pose_scope']=scope
 hold=s.get('holdout',[512,704])
 if not isinstance(hold,list) or len(hold)!=2 or any(type(n) is not int for n in hold) or not 0<=hold[0]<hold[1]:raise ValueError('holdout must be [first_sync, end_exclusive]')
 result['holdout']=hold
 refs=s.get('dorsal_reference')
 if not isinstance(refs,dict) or set(refs)!={'left','right'}:raise ValueError('Provide dorsal_reference for left and right in this recording')
 for h,r in refs.items():
  if not isinstance(r,dict) or set(r)!={'camera','sync'} or r['camera'] not in ['camera0','camera1','camera2','camera3'] or type(r['sync']) is not int or r['sync']<0:raise ValueError('Invalid dorsal reference')
 j=s.get('joint_pose_attachments',{'enabled':False})
 if not isinstance(j,dict) or set(j)-{'enabled','steps','view_policy'} or type(j.get('enabled',False)) is not bool:raise ValueError('Invalid joint_pose_attachments configuration')
 steps=j.get('steps',160);policy=j.get('view_policy','all')
 if type(steps) is not int or steps<1 or policy not in ['all','leave_one_out']:raise ValueError('Invalid joint pose/attachment steps or view_policy')
 result['joint_pose_attachments']={'enabled':j.get('enabled',False),'steps':steps,'view_policy':policy}
 result['dorsal_reference']=refs
 return result


def load_joint_refit_config(path=None):
 """Opt-in per-recording joint refit. Absent or enabled=false leaves the baseline untouched."""
 if path is None: return {'enabled': False}
 p = Path(path).expanduser().resolve(); data = json.loads(p.read_text())
 section = data.get('joint_refit')
 if section is None: return {'enabled': False}
 if not isinstance(section, dict): raise ValueError('joint_refit must be an object')
 boolean_options = ('sdk_marker_gaps', 'dorsal_ring_attachment', 'bounded_articulation', 'preserve_visible_contours', 'temporal_articulation')
 allowed = {'enabled', 'placement', 'sample_stride', 'joint_limit_weight', 'min_markers_per_finger', *boolean_options}
 if set(section) - allowed: raise ValueError('Unknown joint_refit setting: ' + ', '.join(sorted(set(section) - allowed)))
 enabled = section.get('enabled', False)
 if type(enabled) is not bool: raise ValueError('joint_refit.enabled must be a JSON boolean')
 if not enabled: return {'enabled': False}
 value = section.get('placement')
 if not isinstance(value, str) or not value.strip(): raise ValueError('joint_refit.enabled requires a placement JSON')
 q = Path(value).expanduser(); placement = (p.parent / q).resolve() if not q.is_absolute() else q.resolve()
 if not placement.is_file(): raise FileNotFoundError(placement)
 stride = section.get('sample_stride', 8)
 if type(stride) is not int or stride < 1: raise ValueError('joint_refit.sample_stride must be an integer >= 1')
 result = {'enabled': True, 'placement': str(placement), 'sample_stride': stride}
 if 'joint_limit_weight' in section:
  weight = section['joint_limit_weight']
  if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
   raise ValueError('joint_refit.joint_limit_weight must be finite and nonnegative')
  result['joint_limit_weight'] = weight
 if 'min_markers_per_finger' in section:
  minimum = section['min_markers_per_finger']
  if type(minimum) is not int or not 0 <= minimum <= 4:
   raise ValueError('joint_refit.min_markers_per_finger must be an integer from 0 to 4')
  result['min_markers_per_finger'] = minimum
 for name in boolean_options:
  if name in section:
   if type(section[name]) is not bool:
    raise ValueError(f'joint_refit.{name} must be a JSON boolean')
   result[name] = section[name]
 if result.get('temporal_articulation') and not result.get('bounded_articulation'):
  raise ValueError('joint_refit.temporal_articulation requires bounded_articulation')
 if result.get('preserve_visible_contours') and not result.get('bounded_articulation'):
  raise ValueError('joint_refit.preserve_visible_contours requires bounded_articulation')
 return result
