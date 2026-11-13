from pathlib import Path
import copy
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ego_hand_inference.raw_detector_overlay import RawDetectorOverlay, draw_detector_boxes, original_detector_boxes


def record():
    # Both WiLoR hypotheses belong to one right-hand detector box.
    hand = {'detection_index': 0, 'hypothesis_index': 0, 'is_right': 0,
            'detector_is_right': 1, 'confidence': .94, 'bbox_xyxy': [10.25, 35., 70.75, 90.],
            'proposal_source': 'cached_original_detector', 'joints_2d': []}
    return {'sync_index': 0, 'source_frame_index': 3, 'timestamp_ns': 123,
            'hands': [hand, dict(hand, hypothesis_index=1, is_right=1)]}


def test_original_detector_class_and_box_survive_opposite_pose_hypotheses():
    row = record()
    boxes = original_detector_boxes(row, (100, 100))
    assert boxes == [{'detection_index': 0, 'bbox_xyxy': [10.25, 35., 70.75, 90.],
                      'confidence': .94, 'detector_is_right': 1}]
    original = copy.deepcopy(boxes)
    image = np.zeros((100, 100, 3), np.uint8)
    draw_detector_boxes(image, boxes)
    assert boxes == original
    np.testing.assert_array_equal(image[60, 10], [0, 255, 255])


def test_multiple_same_side_detections_are_all_preserved():
    row = record()
    row['hands'] += [dict(row['hands'][0], detection_index=1, hypothesis_index=2,
                         bbox_xyxy=[72., 35., 98., 90.])]
    assert len(original_detector_boxes(row, (100, 100))) == 2


@pytest.mark.parametrize('change', ['score_conflict', 'box_conflict', 'non_detector'])
def test_ambiguous_or_synthetic_boxes_are_rejected(change):
    row = record()
    if change == 'score_conflict':
        row['hands'][1]['confidence'] = .8
    elif change == 'box_conflict':
        row['hands'][1]['bbox_xyxy'] = [11., 35., 70., 90.]
    else:
        row['hands'][0]['proposal_source'] = 'sam3'
    with pytest.raises(ValueError):
        original_detector_boxes(row, (100, 100))


def test_boxes_render_without_any_shared_hand_and_validate_synchronization(tmp_path):
    dataset, predictions = tmp_path / 'dataset', tmp_path / 'predictions'
    dataset.mkdir()
    (predictions / 'camera0').mkdir(parents=True)
    row = record()
    empty = {'sync_index': 1, 'source_frame_index': 4, 'timestamp_ns': 456, 'hands': []}
    sync = [{'sync_index': '0', 'camera0_frame_index': '3', 'camera0_timestamp_ns': '123'},
            {'sync_index': '1', 'camera0_frame_index': '4', 'camera0_timestamp_ns': '456'}]
    path = predictions / 'camera0/predictions.jsonl'
    path.write_text(json.dumps(row) + '\n' + json.dumps(empty) + '\n')
    (predictions / 'summary.json').write_text(json.dumps({'dataset': str(dataset),
        'camera_ids': ['camera0'], 'frame_count': 2, 'cameras': {'camera0': {'physical_box_count': 1}}}))
    overlay = RawDetectorOverlay(predictions, dataset, ['camera0'], sync, (100, 100))
    image = np.zeros((100, 100, 3), np.uint8)
    assert len(overlay.draw(image, 'camera0', 0)) == 1 and image.any()
    blank = np.zeros_like(image)
    assert overlay.draw(blank, 'camera0', 1) == [] and not blank.any()
    assert overlay.drawn == 1 and overlay.config['independent_of_shared_fit']
    sync[0]['camera0_timestamp_ns'] = '999'
    with pytest.raises(ValueError, match='frame/timestamp mismatch'):
        RawDetectorOverlay(predictions, dataset, ['camera0'], sync, (100, 100))
