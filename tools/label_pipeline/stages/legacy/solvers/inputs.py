# Source work/glove_trainset_20261120_v2/inputs.py, SHA-256 2b343d59b6912cd873badab5cf733350c4269776bd4f3bfc35e0ba8d324ea93d

import case_settings as case

import json

ROOT = case.stage('glove_trainset_20261120_v2')

WORK = ROOT.parent

COMPAT = WORK / 'glove_joints_20261120/compat'

DEST = COMPAT.parent / 'compat_v2'

HANDS = ('left','right')

from glove8 import take, read, dump

from observations import geometry

from anatomy_common import OPT

def rows(p):
    with p.open() as f:
        yield from map(json.loads,f)
