"""Saved-result layout adapters must preserve frames and score only valid truth."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ego_data.vio_results import load_vio_result
from ego_data.map_view import trajectory_comparison_data


def fixture(root,kind):
    origin=1_700_000_000_000_000_000
    stamps=[origin+i*100_000_000 for i in range(4)]
    info={'session':'synthetic','first_timestamp_ns':origin,'prepared_groups':4,'duration_s':.3,'camera_time_shift_ns':0}
    (root/'input_validation.json').write_text(json.dumps(info))
    ref=root/'reference';ref.mkdir()
    with (ref/'nokov_rigid_bodies.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['frame_no','device_timestamp_raw','rigid_body_name','valid_numeric','x_mm','y_mm','z_mm','qx','qy','qz','qw'])
        for i in range(41):w.writerow([i,origin//1_000_000+i*10,'head',1,1000+i*10,0,0,0,0,0,1])
    sensor={'reference_directory':str(ref),'rigid_body':'head','T_marker_imu':np.eye(4).tolist(),
            'mocap_to_imu_offset_s':0,'training_interval_imu_s':[0,.1]}
    (root/'nokov_calibration.json').write_text(json.dumps(sensor))
    dataset=root/'dataset_4cam';dataset.mkdir()
    (dataset/'frame_map.csv').write_text('timestamp_ns\n'+''.join(f'{s}\n' for s in stamps))
    case='original_2cam';run=root/'runs'/case/'repeat_1';run.mkdir(parents=True)
    (run/'trajectory.csv').write_text('#timestamp,x,y,z,qw,qx,qy,qz\n'+''.join(f'{s},{i*.1},0,0,1,0,0,0\n' for i,s in enumerate(stamps)))
    world=np.eye(4);world[0,3]=1
    record={'directory':str(run),'T_nokov_vio':world.tolist()}
    if kind=='legacy':
        record.pop('T_nokov_vio');record.update(world_registration_rotation=world[:3,:3].tolist(),world_registration_translation_m=world[:3,3].tolist(),translation_error_all_m={'rmse':0})
        summary={'results':{case:{'repeats':[record]}}}
    elif kind=='overlap':
        record['all_overlap']={'position_m':{'rmse':0}}
        summary={'protocol':{'scored_interval_s':[.1,.2],'world_registration_interval_s':[.1,.2]},'results':{case:{'runs':[record]}}}
    else:
        summary={'cases':{case:{'world_transform_loaded':world.tolist(),'all':{'position_rmse_m':0}}},'fixed_parameters_only':True}
    name='summary.json' if kind=='frozen' else 'comparison_summary.json'
    (root/name).write_text(json.dumps(summary))
    return root,stamps,run


class ResultAdapterTest(unittest.TestCase):
    def test_nonzero_camera_offset_cannot_be_relabelled_as_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root,_,_=fixture(Path(temp),'legacy')
            path=root/'input_validation.json';info=json.loads(path.read_text())
            info['camera_time_shift_ns']=1801936;path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError,'Reprepare and rerun VIO'):
                load_vio_result(root)

    def test_all_saved_layouts_preserve_metric_scale_and_existing_registration(self):
        for kind,expected in [('legacy','legacy_repeats'),('overlap','overlap_repeats'),('frozen','frozen_full')]:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                root,_,_=fixture(Path(temp),kind);result=load_vio_result(root)
                run=result['cases']['original_2cam']['runs'][0]
                self.assertEqual(result['format'],expected)
                np.testing.assert_allclose(run['poses'][:,:3,3],run['reference'][:,:3,3],atol=1e-12)
                self.assertLess(run['stats']['all']['position_rmse_m'],1e-12)

    def test_overlap_uses_native_timestamps_and_exports_do_not_overwrite_input_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root,stamps,_=fixture(Path(temp),'overlap');before=(root/'comparison_summary.json').read_bytes()
            result=load_vio_result(root);data=trajectory_comparison_data(result,root)
            self.assertEqual(data['errors']['timestamps_ns'],[str(s) for s in stamps[1:3]])
            self.assertEqual(len(data['time']),2)
            np.testing.assert_allclose(data['comparison']['imu0']['reference'][0],[0,0,0],atol=1e-12)
            self.assertEqual((root/'comparison_summary.json').read_bytes(),before)
            self.assertTrue((root/'viewer_summary.json').exists())

    def test_duplicate_output_timestamps_fail_before_visualization(self):
        with tempfile.TemporaryDirectory() as temp:
            root,_,run=fixture(Path(temp),'legacy');path=run/'trajectory.csv'
            with path.open('a') as f:f.write(path.read_text().splitlines()[-1]+'\n')
            with self.assertRaisesRegex(ValueError,'duplicate'):load_vio_result(root)

    def test_changed_saved_world_cannot_silently_realign_to_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            root,_,_=fixture(Path(temp),'overlap');path=root/'comparison_summary.json';saved=json.loads(path.read_text())
            saved['results']['original_2cam']['runs'][0]['T_nokov_vio'][0][3]=2
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError,'reproduce saved metrics'):load_vio_result(root)


if __name__=='__main__':unittest.main()
