from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ego_data.fixed_relations import (SCHEMA, FixedReference, check_initialization,
                                      pose_errors, sha256, transform_poses,
                                      validate_bindings)


class FixedRelationsTest(unittest.TestCase):
    def test_nonzero_camera_offset_in_frozen_relations_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'offset must be 0 ms'):
            validate_bindings({'camera_time_policy':{'camera_to_imu_offset_ns':1801936}})

    def test_saved_world_transform_preserves_later_error_instead_of_refitting(self):
        poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[0,1,2.3]
        world=np.eye(4);world[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
        world[:3,3]=[.2,.4,1.2]
        transformed=transform_poses(poses,world)
        reference=np.array([[.2,.4,1.2],[.2,1.4,1.2],[.2,2.4,1.2]])
        pe,re=pose_errors(transformed,reference,Rotation.from_matrix(transformed[:,:3,:3]),np.ones(3,dtype=bool))
        np.testing.assert_allclose(pe,[0,0,.3],atol=1e-12)
        np.testing.assert_allclose(re,0,atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(transformed[-1,:3,3]-transformed[0,:3,3]),2.3)

    def test_wrong_initial_timestamp_or_gauge_is_rejected(self):
        pose=np.eye(4);binding={'timestamp_ns':'1234567890123456789','T_vio_imu':pose.tolist()}
        self.assertTrue(check_initialization(1234567890123456789,pose,binding)['passed'])
        with self.assertRaisesRegex(ValueError,'timestamp'):
            check_initialization(1234567890123456790,pose,binding)
        changed=pose.copy();changed[0,3]=.01
        with self.assertRaisesRegex(ValueError,'initialization'):
            check_initialization(1234567890123456789,changed,binding)
        changed=pose.copy();changed[:3,:3]=Rotation.from_euler('z',.1).as_matrix()
        with self.assertRaisesRegex(ValueError,'initialization'):
            check_initialization(1234567890123456789,changed,binding)

    def test_changed_input_or_executable_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'bound';source.write_bytes(b'calibrated source')
            relations={'schema':SCHEMA,'T_marker_imu':np.eye(4).tolist(),
                'world_relations':{'case':{'T_nokov_vio':np.eye(4).tolist()}},
                'bindings':{'files':[{'path':str(source),'sha256':sha256(source)}]}}
            validate_bindings(relations)
            source.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError,'changed'):
                validate_bindings(relations)

    def test_fixed_time_offset_and_rotating_lever_are_applied_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin=1_700_000_000_000_000_000
            path=Path(tmp)/'nokov_rigid_bodies.csv'
            with path.open('w',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['rigid_body_name','valid_numeric','device_timestamp_raw','x_mm','y_mm','z_mm','qx','qy','qz','qw'])
                for sec,angle in [(0,0),(1,90),(2,180)]:
                    q=Rotation.from_euler('z',angle,degrees=True).as_quat()
                    writer.writerow(['head',1,origin//1_000_000+sec*1000,sec*1000,0,1000,*q])
            mount=np.eye(4);mount[:3,3]=[.1,0,0]
            mount[:3,:3]=Rotation.from_euler('x',30,degrees=True).as_matrix()
            relations={'nokov_directory':tmp,'rigid_body':'head','T_marker_imu':mount.tolist(),'nokov_to_imu_offset_s':.2}
            reference=FixedReference(relations,origin)
            position,rotation,valid=reference.query(np.array([0,.2,.7,1.2,2.2,2.3]))
            np.testing.assert_array_equal(valid,[False,True,True,True,True,False])
            np.testing.assert_allclose(position[1],[.1,0,1],atol=1e-12)
            np.testing.assert_allclose(position[2],[.5+.1/np.sqrt(2),.1/np.sqrt(2),1],atol=1e-12)
            np.testing.assert_allclose(position[3],[1,.1,1],atol=1e-12)
            expected=Rotation.from_euler('z',90,degrees=True)*Rotation.from_matrix(mount[:3,:3])
            self.assertLess((expected.inv()*rotation[3]).magnitude(),1e-12)

    def test_full_evaluator_counts_first_frame_and_does_not_remove_saved_alignment_error(self):
        import run_vio_nokov_fixed as full
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);origin=1_700_000_000_000_000_000
            (output/'input_validation.json').write_text(json.dumps({'first_timestamp_ns':origin,'prepared_groups':4}))
            with (output/'nokov_rigid_bodies.csv').open('w',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['rigid_body_name','valid_numeric','device_timestamp_raw','x_mm','y_mm','z_mm','qx','qy','qz','qw'])
                for sec in range(4):writer.writerow(['head',1,origin//1_000_000+sec*1000,0,0,0,0,0,0,1])
            world=np.eye(4);world[1,3]=.2
            binding={'T_nokov_vio':world.tolist(),'fit_poses':3,'fit_interval_s':[0,2],
                     'initialization':{'timestamp_ns':str(origin),'T_vio_imu':np.eye(4).tolist()}}
            relations={'ego_session':'synthetic','nokov_directory':tmp,'rigid_body':'head',
                       'T_marker_imu':np.eye(4).tolist(),'nokov_to_imu_offset_s':0.,
                       'camera_time_policy':{'camera_to_imu_offset_ns':0},
                       'sensor_calibration_interval_s':[.5,20.],
                       'world_relations':{case:binding for case in full.CASES}}
            package=output/'relations.json';package.write_text(json.dumps(relations))
            for case in full.CASES:
                directory=output/'runs'/case/'repeat_1';directory.mkdir(parents=True)
                (directory/'vio_timing.json').write_text('{"wall_s":0.1}')
            raw={origin+sec*1_000_000_000:np.eye(4) for sec in range(4)}
            with patch.object(full,'load_basalt_trajectory',return_value=raw),patch.object(full,'plot_trajectories'), \
                 patch('analyze_basalt_ego4_comparison.aligned_poses',side_effect=AssertionError('must not register')), \
                 patch('calibrate_nokov_ego4.calibrate',side_effect=AssertionError('must not calibrate')):
                summary,data,_=full.evaluate_full(output,relations,package)
            self.assertFalse(summary['time_or_mount_or_world_refitted'])
            for case in full.CASES:
                self.assertEqual(summary['cases'][case]['all']['poses'],4)
                self.assertAlmostEqual(summary['cases'][case]['all']['position_rmse_m'],.2)
                np.testing.assert_allclose(data['cases'][case]['position_error_m'],.2)
                np.testing.assert_array_equal(summary['cases'][case]['world_transform_loaded'],world)


if __name__=='__main__':
    unittest.main()
