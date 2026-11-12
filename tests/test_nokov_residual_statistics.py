import base64
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from visualize_nokov_mano import export_html, frame_residual_statistics


class FrameResidualStatisticsTests(unittest.TestCase):
    def test_statistics_use_original_float32_residuals_in_metres(self):
        errors = np.array([[.001, .003, .009]], dtype=np.float32)
        stats = frame_residual_statistics(errors)
        a, b, c = map(float, errors[0])
        self.assertEqual(stats['mean'], [(a + b + c) / 3])
        self.assertEqual(stats['max'], [c])
        self.assertEqual(stats['median'], [b])
        self.assertAlmostEqual(stats['p95'][0], b + .9 * (c - b))

    def test_missing_observations_do_not_change_denominator_or_maximum(self):
        errors = np.array([[np.nan, .002, .006, np.nan],
                           [np.inf, np.nan, .005, -np.inf]], dtype=np.float32)
        stats = frame_residual_statistics(errors)
        self.assertAlmostEqual(stats['mean'][0], .004, places=9)
        self.assertAlmostEqual(stats['max'][0], .006, places=9)
        for values in stats.values():
            self.assertEqual(values[1], float(errors[1, 2]))

    def test_all_missing_is_json_null_and_valid_zero_stays_zero(self):
        errors = np.array([[np.nan, np.nan], [0, np.nan]], dtype=np.float32)
        before = errors.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            stats = frame_residual_statistics(errors)
        for values in stats.values():
            self.assertEqual(values, [None, 0.0])
        self.assertEqual(json.loads(json.dumps(stats, allow_nan=False)), stats)
        np.testing.assert_array_equal(errors, before)

    def test_export_samples_every_third_frame_and_retains_last(self):
        errors = np.array([[.001, .003], [.8, .9], [.7, .8],
                           [np.nan, np.nan], [.004, np.nan]], dtype=np.float32)
        data = dict(frame_no=np.arange(10, 15), time_s=np.arange(5) / 90, handedness=np.array('right'),
                    faces=np.array([[0, 1, 2]]), joint_names=np.array(['wrist']),
                    marker_names=np.array(['a', 'b']), marker_error_m=errors,
                    marker_valid=np.isfinite(errors), attachment_face_ids=np.array([0, 0]))
        for key, count in [('vertices', 3), ('joints', 21), ('markers_world_m', 2), ('predicted_markers', 2)]:
            data[key] = np.zeros((5, count, 3), dtype=np.float32)
        data['vertices'][:, 0] = [.0123456, .0234567, .0345678]
        data['joints'][:, 0] = [.001, .002, .003]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'fit_residuals.png').write_bytes(b'test image')
            export_html(directory, data, {})
            html = (directory / 'mano_viewer.src.html').read_text()
        payload = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html).group(1))
        self.assertEqual(payload['frames'], [10, 13, 14])
        self.assertEqual(payload['valid'], [2, 0, 1])
        detail = np.frombuffer(base64.b64decode(payload['detail_vertices']), dtype='<f4').reshape(3, 3, 3)
        expected = (data['vertices'][[0, 3, 4]].astype(np.float64)
                    - data['joints'][[0, 3, 4], :1]).astype('<f4')
        np.testing.assert_array_equal(detail, expected)
        self.assertFalse(np.allclose(detail, np.round(detail * 10000) / 10000, atol=1e-7))
        self.assertNotIn('__MARKER_INSPECTOR__', html)
        self.assertAlmostEqual(payload['mean'][0], .002, places=9)
        self.assertEqual(payload['max'][0], float(errors[0, 1]))
        for key in ('mean', 'max', 'median', 'p95'):
            self.assertIsNone(payload[key][1])
            self.assertEqual(payload[key][2], float(errors[4, 0]))
        json.dumps(payload, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
