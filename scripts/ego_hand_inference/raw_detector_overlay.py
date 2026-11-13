"""Draw original detector outputs independently of multiview hand fitting."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ego_hand_inference.orbbec_io import fingerprint


DETECTOR_OVERLAY_STYLE = 'raw_detector_boxes_final_labels_v2'


def original_detector_boxes(record, image_size):
    """Collapse pose hypotheses by detector ID, preserving every physical box."""
    width, height = image_size
    boxes = {}
    for hand in record['hands']:
        if hand.get('proposal_source', 'detector') not in ('detector', 'cached_original_detector'):
            raise ValueError('requested detector boxes include a non-detector proposal')
        index = int(hand['detection_index'])
        box = np.asarray(hand['bbox_xyxy'], dtype=float)
        score, side = float(hand['confidence']), hand.get('detector_is_right')
        if (index < 0 or box.shape != (4,) or not np.isfinite(box).all()
                or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            raise ValueError('invalid original detector box')
        if not np.isfinite(score) or not 0 <= score <= 1 or side not in (None, 0, 1):
            raise ValueError('invalid detector class/confidence')
        value = {'detection_index': index, 'bbox_xyxy': box.tolist(),
                 'confidence': score, 'detector_is_right': side}
        if index in boxes and boxes[index] != value:
            raise ValueError('pose hypotheses disagree on the original detector output')
        boxes[index] = value
    return [boxes[key] for key in sorted(boxes)]


def draw_detector_boxes(image, boxes):
    """Only round coordinates for rasterization; never expand or refit boxes."""
    for box in boxes:
        side = box['detector_is_right']
        color = (255, 255, 0) if side == 0 else (0, 255, 255) if side == 1 else (200, 200, 200)
        corners = np.rint(box['bbox_xyxy']).astype(np.int32)
        cv2.rectangle(image, tuple(corners[:2]), tuple(corners[2:]), (20, 20, 20), 7, cv2.LINE_AA)
        cv2.rectangle(image, tuple(corners[:2]), tuple(corners[2:]), color, 3, cv2.LINE_AA)
        label = 'L' if side == 0 else 'R' if side == 1 else '?'
        # Keep the original shared-video label style. FINAL is a display
        # label, not evidence that this detector observation entered the fit.
        cv2.putText(image, f'FINAL {label}',
                    (int(corners[0]), max(28, int(corners[1]) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, color, 2, cv2.LINE_AA)


class RawDetectorOverlay:
    def __init__(self, predictions, dataset, cameras, sync_rows, image_size):
        self.root = Path(predictions).resolve()
        summary_path = self.root / 'summary.json'
        summary = json.loads(summary_path.read_text())
        if Path(summary['dataset']).resolve() != Path(dataset).resolve():
            raise ValueError('detector predictions belong to a different dataset')
        if not set(cameras) <= set(summary['camera_ids']):
            raise ValueError('detector predictions do not cover the requested cameras')
        expected = {int(row['sync_index']): row for row in sync_rows}
        if len(expected) != len(sync_rows):
            raise ValueError('duplicate synchronized frame IDs')
        self.frames, self.counts = {}, {}
        signatures = [fingerprint(summary_path)]
        for camera in cameras:
            path = self.root / camera / 'predictions.jsonl'
            signatures.append(fingerprint(path))
            records = {}
            with path.open() as stream:
                for record in map(json.loads, stream):
                    sync = int(record['sync_index'])
                    if sync not in expected:
                        continue
                    if sync in records:
                        raise ValueError(f'{camera}: duplicate detector frame {sync}')
                    row = expected[sync]
                    if (int(record['source_frame_index']) != int(row[f'{camera}_frame_index'])
                            or int(record['timestamp_ns']) != int(row[f'{camera}_timestamp_ns'])):
                        raise ValueError(f'{camera}/{sync}: detector frame/timestamp mismatch')
                    records[sync] = original_detector_boxes(record, image_size)
            if records.keys() != expected.keys():
                raise ValueError(f'{camera}: missing detector frames')
            count = sum(map(len, records.values()))
            if len(expected) == summary['frame_count'] and count != summary['cameras'][camera]['physical_box_count']:
                raise ValueError(f'{camera}: physical detector box count differs from inference summary')
            self.frames[camera] = records
            self.counts[camera] = count
        self.config = {'style': DETECTOR_OVERLAY_STYLE, 'predictions': str(self.root), 'inputs': signatures,
                       'label': 'FINAL L/R', 'selection': 'all original detector boxes once per detection ID',
                       'independent_of_shared_fit': True}
        self.drawn = 0

    def draw(self, image, camera, sync):
        boxes = self.frames[camera][sync]
        draw_detector_boxes(image, boxes)
        self.drawn += len(boxes)
        return boxes
