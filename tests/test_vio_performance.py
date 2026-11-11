from pathlib import Path
import json
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ego_data.vio_performance import collect_vio_performance,read_stats


class VioPerformanceTest(unittest.TestCase):
    def test_ubjson_numeric_stats_preserve_seconds_and_large_double_values(self):
        def key(value):return b'i'+struct.pack('>b',len(value))+value.encode()
        data=b'{'+key('measure')+b'['+b''.join(b'D'+struct.pack('>d',v) for v in [.001,.004,.020])+b']'+key('frame_id')+b'[D'+struct.pack('>d',1.79379353273175e18)+b']}'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'stats.ubjson';path.write_bytes(data)
            stats=read_stats(path)
        self.assertEqual(stats['measure'],[.001,.004,.020])
        self.assertEqual(stats['frame_id'],[1.79379353273175e18])

    def test_measured_backend_and_process_throughput_have_different_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'runs'/'case'/'repeat_1';directory.mkdir(parents=True)
            (root/'input_validation.json').write_text('{"prepared_groups":3}')
            (directory/'stats_sums.json').write_text(json.dumps({'measure':[.001,.004,.020],'optimize':[.002,.010]}))
            (directory/'stats_vio.json').write_text(json.dumps({'exec_time_s':[.25],'resident_memory_peak':[1024**2]}))
            (directory/'vio_timing.json').write_text('{"wall_s":0.3}')
            (directory/'trajectory.csv').write_text('#timestamp,pose\n1700000000000000000,0\n1700000000100000000,0\n1700000000200000000,0\n')
            (directory/'vio_resource.txt').write_text('User time (seconds): 0.5\nSystem time (seconds): 0.1\nPercent of CPU this job got: 200%\nMaximum resident set size (kbytes): 4096\n')
            result=collect_vio_performance(root,'case')
        self.assertAlmostEqual(result['backend_frame_ms']['mean'],25/3)
        self.assertEqual(result['backend_frame_ms']['median'],4)
        self.assertEqual(result['backend_frame_ms']['max'],20)
        self.assertEqual(result['backend_frame_ms']['count'],3)
        self.assertAlmostEqual(result['wall_ms_per_frame'],100)
        self.assertAlmostEqual(result['throughput_fps'],10)
        self.assertAlmostEqual(result['recording_duration_s'],.2)
        self.assertAlmostEqual(result['rtf'],1.5)
        self.assertEqual(result['cpu_percent'],200)
        self.assertEqual(result['peak_rss_mib'],4)
        self.assertIsNone(result['end_to_end_frame_latency_ms'])
        self.assertTrue(result['backend_samples_match_poses'])
        self.assertEqual(result['output_coverage_percent'],100)
        json.dumps(result,allow_nan=False)

    def test_missing_runtime_data_remains_null_instead_of_using_capture_period(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'runs'/'case'/'repeat_1';directory.mkdir(parents=True)
            (directory/'trajectory.csv').write_text('#timestamp\n1700000000000000000\n1700000000033333333\n')
            result=collect_vio_performance(root,'case')
        self.assertIsNone(result['backend_frame_ms']['mean'])
        self.assertIsNone(result['wall_ms_per_frame'])
        self.assertIsNone(result['throughput_fps'])
        self.assertIsNone(result['peak_rss_mib'])
        self.assertEqual(result['processed_frames'],2)
        json.dumps(result,allow_nan=False)


if __name__=='__main__':unittest.main()
