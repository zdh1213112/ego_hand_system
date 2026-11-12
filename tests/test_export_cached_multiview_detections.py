from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_cached_multiview_detections import detection_row, export_cache


def fixture_row():
    sync = {'sync_index': '0', 'camera0_frame_index': '3', 'camera0_timestamp_ns': '12345'}
    hand = {'detection_index': 0, 'hypothesis_index': 0, 'is_right': 0, 'detector_is_right': 1,
            'bbox_xyxy': [10., 20., 70., 80.], 'confidence': .75, 'proposal_source': 'detector',
            'joints_2d': [[999, 999]], 'mano': 'must not be copied'}
    record = {'sync_index': 0, 'source_frame_index': 3, 'timestamp_ns': 12345,
              'hands': [hand, dict(hand, hypothesis_index=1, is_right=1)]}
    return record, sync


def test_recovery_exactly_deduplicates_boxes_without_pose_fields():
    record, sync = fixture_row()
    output = detection_row(record, sync, 'camera0', [100, 100])
    assert len(output['boxes']) == 1
    box = output['boxes'][0]
    assert box['bbox_xyxy'] == record['hands'][0]['bbox_xyxy']
    assert box['confidence'] == .75 and box['detector_is_right'] == 1
    assert set(box) == {'detection_index', 'bbox_xyxy', 'confidence', 'detector_is_right', 'source'}


@pytest.mark.parametrize('change,message', [
    ('bounds', 'bounds'), ('conflict', 'disagree'), ('missing_index', 'dropped'),
    ('timestamp', 'timestamp mismatch'), ('source_frame', 'source frame mismatch'),
])
def test_recovery_rejects_corrupt_or_ambiguous_source(change, message):
    record, sync = fixture_row()
    if change == 'bounds': record['hands'][0]['bbox_xyxy'] = [-1., 20., 70., 80.]
    elif change == 'conflict': record['hands'][1]['confidence'] = .8
    elif change == 'missing_index':
        record['hands'] = [dict(record['hands'][0], detection_index=1, hypothesis_index=2)]
    elif change == 'timestamp': record['timestamp_ns'] += 1
    elif change == 'source_frame': record['source_frame_index'] += 1
    with pytest.raises(ValueError, match=message):
        detection_row(record, sync, 'camera0', [100, 100])


def test_export_fails_if_pose_adapter_lost_a_whole_box(tmp_path):
    dataset, source, output = tmp_path / 'dataset', tmp_path / 'source', tmp_path / 'output'
    dataset.mkdir(); (source / 'camera0').mkdir(parents=True)
    (dataset / 'manifest.json').write_text(json.dumps({'camera_ids': ['camera0'], 'image_size': [100, 100]}))
    (dataset / 'multiview_frames.csv').write_text('sync_index,camera0_frame_index,camera0_timestamp_ns\n0,3,12345\n')
    record, _ = fixture_row()
    (source / 'camera0/predictions.jsonl').write_text(json.dumps(record) + '\n')
    summary = {'stage': 'gem_multiview_dual_hypothesis', 'camera_ids': ['camera0'], 'dataset': str(dataset),
               'frame_count': 1, 'detector': '/original/detector.pt', 'cameras': {'camera0': {
                   'frame_count': 1, 'physical_box_count': 2, 'detected_frame_count': 1, 'detector_confidence': .3}}}
    (source / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='lost original detector boxes'):
        export_cache(dataset, source, output)
    assert not output.exists() and not list(tmp_path.glob('output.partial-*'))
    summary['cameras']['camera0']['physical_box_count'] = 1
    (source / 'summary.json').write_text(json.dumps(summary))
    report = export_cache(dataset, source, output)
    assert report['detector'] == '/original/detector.pt'
    assert report['source_prediction_files']['camera0']['sha256']
    assert not report['contains_pose_or_keypoints']
    assert json.loads((output / 'camera0/predictions.jsonl').read_text())['boxes'][0]['confidence'] == .75


def test_inference_loader_checks_available_frame_and_timestamp_fields(tmp_path):
    from wilor_multiview_inference import _load_external_boxes
    record, sync = fixture_row()
    row = detection_row(record, sync, 'camera0', [100, 100])
    path = tmp_path / 'camera0/predictions.jsonl'; path.parent.mkdir()
    path.write_text(json.dumps(row) + '\n')
    assert _load_external_boxes(tmp_path, ('camera0',), [sync])['camera0'][0] == row
    row['timestamp_ns'] += 1
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='timestamp mismatch'):
        _load_external_boxes(tmp_path, ('camera0',), [sync])
