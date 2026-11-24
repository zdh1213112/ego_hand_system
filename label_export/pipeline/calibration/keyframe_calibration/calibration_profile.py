"""Fixed personal calibration identity and asset checks, without frame-dependent parameters."""
import hashlib
import json
from pathlib import Path

import numpy as np


def placement_signature(placement):
    # Review status and notes do not change physical placement. User-edited region,
    # surface and normalized photo coordinates do invalidate a saved calibration.
    data = {'marker_diameter_mm': placement.get('marker_diameter_mm'),
            'mount': placement.get('mount'), 'hands': {}}
    for hand in ['left', 'right']:
        data['hands'][hand] = {}
        for name, marker in placement['hands'][hand].items():
            if not isinstance(marker, dict) or 'region' not in marker:
                continue
            point = marker.get('point', {})
            data['hands'][hand][name] = {'region': marker['region'], 'surface': marker.get('surface'),
                                         'point': {k: point[k] for k in ['view', 'x', 'y'] if k in point}}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_profile(data, model_faces=None, placement=None, model_path=None):
    if data.get('schema') != 'personal_glove_calibration_v1' or data.get('geometry_space') != 'right_canonical':
        raise ValueError('Expected right-canonical personal_glove_calibration_v1')
    if data.get('approved_for_training') is not False:
        raise ValueError('Calibration must retain its explicit experimental status')
    if set(data.get('hands', {})) != {'left', 'right'}:
        raise ValueError('Calibration must contain both physical hands')
    for hand, values in data['hands'].items():
        for key, shape in [('betas', (10,)), ('attachment_faces', (21,)), ('barycentric', (21, 3)),
                           ('normal_offsets_m', (21,)), ('wrist_local_m', (3, 3))]:
            x = np.asarray(values[key])
            if x.shape != shape or not np.isfinite(x).all():
                raise ValueError(f'{hand}: invalid {key}')
        faces = np.asarray(values['attachment_faces'])
        if not np.equal(faces, np.rint(faces)).all() or (faces < 0).any():
            raise ValueError(f'{hand}: invalid face indices')
        if model_faces is not None and (faces >= len(model_faces)).any():
            raise ValueError(f'{hand}: face index outside model')
        bary = np.asarray(values['barycentric'])
        if (bary < 0).any() or not np.allclose(bary.sum(-1), 1, atol=1e-5):
            raise ValueError(f'{hand}: invalid barycentric coordinates')
        offsets = np.asarray(values['normal_offsets_m'])
        if (offsets < 0).any() or (offsets > .004001).any():
            raise ValueError(f'{hand}: effective offsets outside calibrated 0..4mm bounds')
    provenance = data.get('provenance', {})
    if placement is not None:
        if provenance.get('placement_signature') != placement_signature(placement):
            raise ValueError('Marker placement differs from saved calibration; recalibrate for changed placement')
    if model_path is not None:
        if provenance.get('mano_right_sha256') != file_sha256(model_path):
            raise ValueError('MANO model differs from saved calibration')
    return data
