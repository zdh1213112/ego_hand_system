from pathlib import Path
import sys
import unittest

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from visualize_ego_nokov_sync import (angular_velocity, motion_comparison,
                                      hand_projection_config, project_world_markers,
                                      skeleton_coordinate_check)


class SynchronizationVisualizationTest(unittest.TestCase):
    def test_skeleton_coordinate_check_rejects_local_joint_positions(self):
        rng=np.random.default_rng(21)
        root=np.column_stack([np.linspace(.3,.6,40),np.full(40,-.4),np.full(40,.9)])
        markers=root[:,None,:]+rng.normal(0,.02,(40,24,3))
        joints=markers[:,np.arange(26)%24,:].copy()
        self.assertTrue(skeleton_coordinate_check(joints,markers)['available'])
        self.assertFalse(skeleton_coordinate_check(joints-root[:,None,:],markers)['available'])

    def test_projection_recovers_optical_points_after_nontrivial_world_and_installation_transforms(self):
        r_cm=Rotation.from_euler('xyz',[.2,-.3,.1])
        t_cm=np.eye(4);t_cm[:3,:3]=r_cm.as_matrix();t_cm[:3,3]=[.04,-.03,.02]
        head_r=Rotation.from_euler('xyz',[-.5,.4,.8]);head_p=np.array([.6,-.4,1.3])
        optical=np.array([[0,0,1],[.2,-.1,.8],[0,0,-1],[1000,0,1],[np.nan,0,1.]])
        local=r_cm.inv().apply(optical-t_cm[:3,3]);world=head_r.apply(local)+head_p
        camera={'id':'cam_0','image_size':[640,480],'intrinsics':{'fx':300,'fy':310,'cx':320,'cy':240},'distortion':[.02,-.003,.0004,-.00001],'T_camera_marker':t_cm.tolist()}
        uv,valid,recovered=project_world_markers(world,head_p,head_r.as_quat(),camera)
        np.testing.assert_allclose(recovered[:4],optical[:4],atol=1e-12)
        expected,_=cv2.fisheye.projectPoints(optical[:2].reshape(-1,1,3),np.zeros(3),np.zeros(3),np.array([[300.,0,320],[0,310,240],[0,0,1]]),np.array(camera['distortion']))
        np.testing.assert_allclose(uv[:2],expected[:,0],atol=1e-10)
        self.assertEqual(valid.tolist(),[True,True,False,False,False])

    def test_factory_and_installation_composition_preserves_direction_and_mm_units(self):
        def camera(index,r,t):
            return {'id':f'cam_{index}','image_width':640,'image_height':480,'distortion_model':'KB','intrinsics':{'fx':300,'fy':300,'cx':320,'cy':240},'distortion':{f'k{i}':0 for i in range(1,5)},'extrinsics':{'rotation':r.tolist(),'translation':t}}
        r_c1_c0=Rotation.from_euler('z',.3).as_matrix()
        cameras={'cameras':[camera(0,np.eye(3),[0,0,0]),camera(1,r_c1_c0,[-100,0,0])]}
        t_c0_i=np.eye(4);t_c0_i[:3,:3]=Rotation.from_euler('y',-.2).as_matrix();t_c0_i[:3,3]=[10,20,30]
        noise={'noise_density':.01,'random_walk':.001}
        imu={'cam0':{'T_cam_imu':t_c0_i.tolist(),'timeshift_cam_imu':0},'imu0':{'update_rate':1000,'accelerometer':noise,'gyroscope':noise}}
        t_m_i=np.eye(4);t_m_i[:3,:3]=Rotation.from_euler('x',.4).as_matrix();t_m_i[:3,3]=[.003,-.004,-.024]
        result=hand_projection_config(cameras,imu,{'T_marker_imu':t_m_i.tolist()})
        t_c0_i[:3,3]*=.001
        t_c1_c0=np.eye(4);t_c1_c0[:3,:3]=r_c1_c0;t_c1_c0[:3,3]=[-.1,0,0]
        np.testing.assert_allclose(result['cameras'][0]['T_camera_marker'],t_c0_i@np.linalg.inv(t_m_i),atol=1e-12)
        np.testing.assert_allclose(result['cameras'][1]['T_camera_marker'],t_c1_c0@t_c0_i@np.linalg.inv(t_m_i),atol=1e-12)

    def test_rotation_derivative_is_body_angular_velocity_without_edge_extrapolation(self):
        t=np.arange(101)*.01
        rotation=Rotation.from_euler('z',t*2)
        indices,omega=angular_velocity(t,rotation)
        self.assertGreater(t[indices[0]],0)
        self.assertLess(t[indices[-1]],t[-1])
        np.testing.assert_allclose(omega,np.tile([0,0,2],(len(indices),1)),atol=1e-12)

    def test_known_capture_delay_is_corrected_before_nearest_matching(self):
        epoch=1793793532731618000
        raw=[epoch+i*10_000_000 for i in range(100)]
        offset=160_000_000
        query=[t+offset for t in raw]
        omega=np.column_stack([np.arange(100)*.01,np.zeros(100),np.zeros(100)])
        before,old=motion_comparison(query,omega,raw,omega,0,2_000_000,query[50])
        after,new=motion_comparison(query,omega,raw,omega,offset,2_000_000,query[50])
        self.assertGreater(old['held_out_after_20s']['speed_norm_rmse_rad_s'],.1)
        self.assertEqual(new['held_out_after_20s']['speed_norm_rmse_rad_s'],0)
        self.assertEqual(new['held_out_after_20s']['matched_queries'],50)
        self.assertEqual(after[0][0],query[0])
        self.assertEqual(after[0][1],0)
        self.assertIsNone(before[-1][1])


if __name__=='__main__':
    unittest.main()
