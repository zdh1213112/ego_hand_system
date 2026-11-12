from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_shared_wilor_multiview import measure_geometry, paired_residuals, render_comparison, _probe, hand_frame_histograms
from ego_data.calibration import CameraCalibration
from gem_shared_geometry import project_base


def test_geometry_metrics_exclude_invalid_joints_and_count_partial_hands():
    cal = CameraCalibration('camera0', 'optical', 'PINHOLE', (100, 100),
        [[80, 0, 50], [0, 80, 50], [0, 0, 1]], [], np.eye(4))
    points = np.zeros((21, 3))
    points[:, 2] = 1
    pixels, _, _ = project_base(cal, points)
    pixels += [3, 4]
    points[20] = np.nan
    rows = {7: {'hands': [{'side': 1, 'joints_base_m': points.tolist(),
        'views': {'camera0': {'detection_index': 5, 'joints_2d': pixels.tolist()}}}]}}
    result, errors = measure_geometry(rows, {'camera0': cal})
    assert result['frames_with_finite_3d'] == result['finite_hands'] == 1
    assert result['complete_21_joint_hands'] == 0
    assert result['finite_3d_joints'] == 20
    assert result['selected_view_residual_px']['median'] == pytest.approx(5.)
    assert len(errors) == 20
    histograms = hand_frame_histograms(rows, [6, 7])
    assert histograms['stored_hand_entries'] == {'0': 1, '1': 1, '2': 0}
    assert histograms['hands_with_any_finite_3d_joint'] == {'0': 1, '1': 1, '2': 0}
    assert histograms['complete_21_joint_hands'] == {'0': 2, '1': 0, '2': 0}
    rows[7]['hands'].append(rows[7]['hands'][0])
    with pytest.raises(ValueError, match='duplicate'):
        measure_geometry(rows, {'camera0': cal})


def test_pairing_rejects_different_detections_and_hypothesis_pixels():
    keys = [(3, 1, 'camera0', i) for i in range(4)]
    before = {k: {'error_px': 10., 'detection_index': 2, 'observed': np.array([50., 55.])} for k in keys}
    after = {k: {'error_px': 4., 'detection_index': 2, 'observed': np.array([50., 55.])} for k in keys}
    after[keys[1]]['detection_index'] = 3
    after[keys[2]]['observed'] = np.array([55., 50.])
    before[keys[3]]['detection_index'] = None
    stats, samples = paired_residuals(before, after)
    assert len(samples) == 1
    assert stats['delta_shared_minus_legacy']['median'] == -6.
    assert stats['excluded_different_or_missing_detection'] == 2
    assert stats['excluded_different_observation'] == 1
    empty, samples = paired_residuals({}, after)
    assert empty['legacy']['count'] == 0 and empty['legacy']['median'] is None
    assert samples == []


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='ffmpeg unavailable')
def test_comparison_video_keeps_frame_count_and_rejects_mismatched_sources(tmp_path):
    sources = []
    for index in (0, 1):
        path = tmp_path / f'input{index}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        f'color=c={"red" if index == 0 else "blue"}:s=160x100:r=10',
                        '-frames:v', '3', '-c:v', 'libx264', '-threads', '1', str(path)], check=True)
        sources.append(path)
    report = {'synchronized_frames': 3, 'videos': {'legacy': str(sources[0]), 'shared': str(sources[1])}}
    output = tmp_path / 'comparison.mp4'
    render_comparison(report, output)
    assert _probe(output) == (3, 10.)
    assert report['videos']['comparison'] == str(output)
    with pytest.raises(FileExistsError):
        render_comparison(report, output)
    report['synchronized_frames'] = 4
    with pytest.raises(ValueError, match='identical synchronized'):
        render_comparison(report, tmp_path / 'invalid.mp4')
