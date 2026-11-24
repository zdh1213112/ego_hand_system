"""Prove the v2 observations and NOKOV solver use the same source frames and camera clocks."""
import csv
from pathlib import Path
import numpy as np

from stages.signed import write_json

def check_archive(root, archive):
    with (root/'native/processed/ego/normalized_multiview/multiview_frames.csv').open() as f:
        frames=list(csv.DictReader(f))
    with np.load(archive,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['times'],np.arange(len(frames)))
        np.testing.assert_array_equal(z['reference_timestamp_ns'],[int(r['reference_timestamp_ns']) for r in frames])
        np.testing.assert_array_equal(z['camera_indices'],[[int(r[f'camera{c}_frame_index']) for c in range(4)] for r in frames])
        if z['valid'].shape!=(len(frames),4,2) or not np.isfinite(z['joints_2d'][z['valid']]).all():
            raise ValueError('Invalid four-camera WiLoR observations')


def check(root,archive):
    native=root/'native/processed/ego/normalized_multiview/multiview_frames.csv'
    source=root.parents[1]/'inference'/root.name/'multiview_frames.csv'
    cols=['reference_timestamp_ns',*[f'camera{c}_{field}' for c in range(4) for field in ('frame_index','timestamp_ns')]]
    def values(f):
        with f.open() as stream:return np.array([[int(r[k]) for k in cols] for r in csv.DictReader(stream)],np.int64)
    x=values(native);y=values(source);np.testing.assert_array_equal(x,y)
    with np.load(archive) as z:
        np.testing.assert_array_equal(z['times'],np.arange(len(x)))
        np.testing.assert_array_equal(z['camera_indices'],x[:,1::2])
        np.testing.assert_array_equal(z['reference_timestamp_ns'],x[:,0])
        if z['valid'].shape!=(len(x),4,2) or not np.isfinite(z['joints_2d'][z['valid']]).all():raise ValueError('WiLoR 观测无效')
        count=z['valid'].sum(axis=0).tolist()
    f=root/'records/same_exposure_check.json';write_json(f,dict(frames=len(x),compared_fields=cols,
        max_clock_difference_ns=0,source_frame_mismatches=0,observed_cam_left_right=count))
    return f
