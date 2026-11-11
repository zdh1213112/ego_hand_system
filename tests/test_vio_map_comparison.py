"""A comparison must use one display frame, retaining each method's errors."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ego_data.map_view import change_frame,enrich_map_data

CASES=('original_2cam','monado_2cam','monado_4cam')


def write_poses(path,stamps,positions,rotations):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['timestamp_ns','x_m','y_m','z_m','qw','qx','qy','qz'])
        for stamp,p,q in zip(stamps,positions,rotations.as_quat()):
            writer.writerow([int(stamp),*p,q[3],*q[:3]])


def fixture(output):
    times=np.arange(4)*.1
    stamps=np.int64(1_700_000_000_000_000_000)+np.arange(4,dtype=np.int64)*100_000_000
    truth=np.array([[.3,-.2,1.2],[.4,-.1,1.21],[.5,.05,1.22],[.6,.1,1.23]])
    truth_rotation=Rotation.from_euler('xyz',[[.15,-.25,.6],[.16,-.24,.7],[.18,-.22,.8],[.2,-.2,.9]])
    write_poses(output/'evaluation/nokov_imu_reference.csv',stamps,truth,truth_rotation)
    info={'first_timestamp_ns':int(stamps[0]),'last_timestamp_ns':int(stamps[-1]),'prepared_groups':4,'duration_s':.3}
    (output/'input_validation.json').write_text(json.dumps(info))
    data={'time':times.tolist(),'maps':{},'summary':{'cases':{},'duration_s':.3,'poses_per_case':4}}
    poses={}
    for i,case in enumerate(CASES):
        offset=np.array([.03+i*.06,.01-i*.04,-.02+i*.01])
        estimate=truth+offset+times[:,None]*np.array([.01*i,-.02*i,.005*i])
        orientation=truth_rotation*Rotation.from_euler('z',.01+i*.03)
        write_poses(output/'evaluation'/f'{case}_imu_nokov_world.csv',stamps,estimate,orientation)
        source=output/'runs'/case/'repeat_1/map_lm';source.mkdir(parents=True)
        (source/'sparse_landmarks.csv').write_text('landmark_id,x_m,y_m,z_m,observation_count\n1,0,0,1,4\n2,1,1,2,5\n')
        (source.parent/'vio_timing.json').write_text('{"wall_s":0.03,"returncode":0}')
        points=np.array([[.8,-.5,.4],[1.,.7,1.8]])+offset
        stats={'label':case,'color':'#f00','all':{'poses':4,'position_rmse_m':.1,'orientation_rmse_deg':1.},
               'wall_s':.03,'mapping':{'landmarks':2,'reprojection_rmse_px':1.,'wall_s':.02}}
        data['maps'][case]={'points':points.tolist(),'online':estimate.tolist(),'reference':truth.tolist(),
                           'keyframes':estimate[[0,-1]].tolist(),'keyframe_times_s':[0.,.3],
                           'first_seen_s':[0.,.2],'observations':[4,5],'stats':stats}
        data['summary']['cases'][case]=stats
        poses[case]=(estimate,orientation)
    return data,truth,truth_rotation,poses


class VioMapComparisonTest(unittest.TestCase):
    def test_every_trajectory_truth_and_map_share_the_reference_imu_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);data,truth,truth_rotation,poses=fixture(output)
            before=deepcopy(data)
            result=enrich_map_data(output,data)
            first=np.eye(4);first[:3,:3]=truth_rotation[0].as_matrix();first[:3,3]=truth[0]
            transform=np.linalg.inv(first)
            np.testing.assert_allclose(result['comparison']['imu0']['T_imu0_nokov'],transform,atol=1e-12)
            np.testing.assert_allclose(result['display_frames']['T_imu0_nokov'],transform,atol=1e-12)
            expected_truth=change_frame(truth,transform)
            np.testing.assert_allclose(result['comparison']['imu0']['reference'],expected_truth,atol=1e-7)
            np.testing.assert_allclose(result['comparison']['imu0']['reference'][0],[0,0,0],atol=1e-7)
            for case in CASES:
                estimate,orientation=poses[case]
                np.testing.assert_allclose(result['comparison']['nokov']['online'][case],estimate,atol=1e-12)
                np.testing.assert_allclose(result['comparison']['imu0']['online'][case],change_frame(estimate,transform),atol=1e-7)
                np.testing.assert_allclose(result['maps'][case]['imu0']['points'],change_frame(before['maps'][case]['points'],transform),atol=1e-7)
                np.testing.assert_allclose(result['maps'][case]['imu0']['keyframes'],change_frame(before['maps'][case]['keyframes'],transform),atol=1e-7)
                for key in ['points','online','reference','keyframes']:
                    np.testing.assert_array_equal(result['maps'][case][key],before['maps'][case][key])

    def test_comparison_keeps_initial_errors_and_between_algorithm_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            data,truth,_,poses=fixture(Path(directory));result=enrich_map_data(Path(directory),data)
            comparison=result['comparison']['imu0'];reference=np.array(comparison['reference'])
            for case in CASES:
                estimate=np.array(comparison['online'][case]);raw=poses[case][0]
                np.testing.assert_allclose(np.linalg.norm(estimate-reference,axis=1),np.linalg.norm(raw-truth,axis=1),atol=2e-7)
                self.assertGreater(np.linalg.norm(estimate[0]),.02)
            for a,b in zip(CASES[:-1],CASES[1:]):
                display_delta=np.array(comparison['online'][a])-comparison['online'][b]
                world_delta=poses[a][0]-poses[b][0]
                np.testing.assert_allclose(np.linalg.norm(display_delta,axis=1),np.linalg.norm(world_delta,axis=1),atol=2e-7)

    def test_attitudes_use_the_same_left_rotation_as_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            data,_,truth_rotation,poses=fixture(Path(directory));result=enrich_map_data(Path(directory),data)
            for case in CASES:
                expected=(truth_rotation[0].inv()*poses[case][1]).as_matrix()
                actual=np.array(result['comparison']['imu0']['imu_rotations'][case]).reshape(-1,3,3)
                np.testing.assert_allclose(actual,expected,atol=1e-8)
                np.testing.assert_allclose(np.array(result['comparison']['nokov']['imu_rotations'][case]).reshape(-1,3,3),poses[case][1].as_matrix(),atol=1e-8)


if __name__=='__main__':unittest.main()
