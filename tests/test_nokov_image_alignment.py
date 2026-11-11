from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_nokov_image_markers import (Recording,corrected_config,correction_matrix,
                                         fit_mount,project_observations)
from render_nokov_mano_overlay import (digest,interpolation_brackets,interpolate_hand,
                                      project_interpolated_hand,validate_alignment_sources)
from inspect_nokov_image_alignment import image_alignment_gate


def cameras():
    result=[]
    for i in range(2):
        t=np.eye(4);t[0,3]=-.1*i
        result.append(dict(id=f'cam_{i}',model='KB4',image_size=[1920,1200],
                           intrinsics=dict(fx=505.,fy=505.,cx=940.,cy=590.),
                           distortion=[.07,-.014,.002,0.],T_camera_marker=t.tolist()))
    return {'cameras':result}


class ImageMarkerAlignmentTests(unittest.TestCase):
    def test_gate_keeps_camera_counts_and_data_roles_separate(self):
        self.assertEqual(image_alignment_gate([1.]*5,'validation'),'insufficient_observations')
        self.assertEqual(image_alignment_gate([1.]*8,'development'),'development_only')
        self.assertEqual(image_alignment_gate([1.]*8,'fit'),'fit_only')
        self.assertEqual(image_alignment_gate([1.]*8,'final_test'),'passed_sparse_scope')
        self.assertEqual(image_alignment_gate([12.]*8,'final_test'),'failed')
        self.assertEqual(image_alignment_gate([1.]*8,'unknown'),'unknown_role')
        self.assertEqual(image_alignment_gate([1.]*7+[np.nan],'final_test'),'invalid_observations')

    def test_one_mount_preserves_intercamera_calibration(self):
        original=cameras();after=corrected_config(original,[.02,-.04,.01,.015,.02,-.01])
        relative=lambda cfg:np.asarray(cfg['cameras'][1]['T_camera_marker'])@np.linalg.inv(cfg['cameras'][0]['T_camera_marker'])
        np.testing.assert_allclose(relative(after),relative(original),atol=1e-14)
        np.testing.assert_array_equal(original['cameras'][0]['T_camera_marker'],np.eye(4))

    def test_recover_known_mount_from_two_camera_independent_pixels(self):
        rng=np.random.default_rng(11);points=rng.uniform([-.15,-.1,.4],[.2,.2,1.],(60,3))
        obs=[dict(camera=i%2,uncertainty_px=1.) for i in range(60)]
        truth=np.array([.02,-.015,.025,.01,-.02,.03]);config=cameras()
        pixels=project_observations(points,obs,corrected_config(config,truth))
        for o,p in zip(obs,pixels):o['pixel']=p.tolist()
        fit=fit_mount(points,obs,config)
        self.assertTrue(fit.success)
        np.testing.assert_allclose(correction_matrix(fit.x),correction_matrix(truth),atol=1e-8)

    def test_reject_invalid_correspondences_and_behind_camera_solution(self):
        rng=np.random.default_rng(4)
        points=rng.uniform([-.15,-.1,-1.],[.2,.2,-.4],(30,3))
        obs=[dict(camera=i%2,uncertainty_px=1.) for i in range(len(points))]
        pixels=project_observations(points,obs,cameras())
        for o,p in zip(obs,pixels):o['pixel']=p.tolist()
        with self.assertRaisesRegex(ValueError,'behind the camera'):
            fit_mount(points,obs,cameras())
        obs[0]['uncertainty_px']=0.
        with self.assertRaisesRegex(ValueError,'positive'):
            fit_mount(points,obs,cameras())
        obs[0]['uncertainty_px']=1.
        with self.assertRaisesRegex(ValueError,'Collinear'):
            fit_mount(np.zeros_like(points),obs,cameras())

    def test_large_epoch_offset_once_and_no_extrapolation(self):
        origin=1_794_000_000_000_000_000
        times=np.array([origin,origin+10_000_000,origin+20_000_000],np.int64)
        offset=12_651_797
        queries=np.array([origin+offset-1,origin+offset,origin+offset+5_000_000,
                          origin+offset+20_000_000,origin+offset+20_000_001],np.int64)
        left,right,alpha,valid=interpolation_brackets(queries,times,offset)
        np.testing.assert_array_equal(valid,[False,True,True,True,False])
        np.testing.assert_allclose(alpha[valid],[0,.5,1])
        self.assertTrue(np.isnan(alpha[~valid]).all())

    def test_renderer_and_calibration_sample_same_camera_time(self):
        r=Recording.__new__(Recording)
        r.motion_s=np.array([0.,.01,.02]);r.offset_s=.012
        r.camera_s=np.array([[.017],[.017]])
        r.position=np.array([[0.,0.,0.],[.003,0.,0.],[.006,0.,0.]])
        q=Rotation.from_euler('y',[0.,.01,.02]);r.slerp=Slerp(r.motion_s,q)
        points=np.array([[[.05,.1,.5]],[[.06,.1,.5]],[[.07,.1,.5]]])
        r.data={'markers_world_m':points}
        o=[dict(camera=0,video_frame=0,marker_index=0)]
        expected=project_observations(r.local_points(o),o,cameras())[0]
        data={k:points.copy() for k in ['vertices','joints','markers_world_m','predicted_markers']}
        sample=interpolate_hand(data,0,1,.5)
        uv,_,_=project_interpolated_hand(sample,(r.position[0]+r.position[1])*.5,r.slerp(.005).as_quat(),cameras()['cameras'][0])
        np.testing.assert_allclose(uv[-1],expected,atol=1e-10)
        data['markers_world_m'][1,0]=np.nan
        self.assertTrue(np.isnan(interpolate_hand(data,0,1,.5)['markers_world_m']).all())
        with self.assertRaises(ValueError):interpolate_hand(data,0,1,1.001)

    def test_alignment_rejects_changed_source_and_nonzero_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'reconstruction.npz';a.write_bytes(b'a')
            b=Path(directory)/'calibration.json';b.write_text('{}')
            cfg={'image_alignment':{'source_bindings':{str(a):digest(a),str(b):digest(b)},'time_offset_change_ms':0.}}
            validate_alignment_sources(cfg,a,b)
            a.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'):validate_alignment_sources(cfg,a,b)
            a.write_bytes(b'a');cfg['image_alignment']['time_offset_change_ms']=5
            with self.assertRaisesRegex(ValueError,'clock offset'):validate_alignment_sources(cfg,a,b)


if __name__=='__main__':unittest.main()
