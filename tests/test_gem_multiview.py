"""GEM's 25-joint adapter and single-side observations used by fusion."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gem_multiview_inference as gem
import fuse_multiview_wilor_guided as fusion
from normalize_orbbec_ego4_session import session_stream_file


def test_gem_soma_mapping_preserves_frame_indices_and_hand_side():
    points = np.column_stack((np.arange(25), np.arange(25) + 100))
    sync = {"sync_index": "7", "camera0_frame_index": "11", "camera0_timestamp_ns": "12345"}
    result = gem.adapt_frame({"instances": [{"side": "right", "bbox": [10, 20, 6],
        "keypoints_2d": points, "detection_score": 0.75}]}, sync, "camera0")
    assert result["sync_index"] == 7 and result["source_frame_index"] == 11
    hand = result["hands"][0]
    assert hand["is_right"] == hand["detector_is_right"] == 1
    assert hand["bbox_xyxy"] == [7, 17, 13, 23]
    assert np.asarray(hand["joints_2d"])[:, 0].tolist() == list(gem.SOMA_TO_21)
    assert [hand["joints_2d"][i][0] for i in (4, 8, 12, 16, 20)] == [4, 9, 14, 19, 24]


def test_fusion_accepts_only_available_hypotheses(monkeypatch):
    left = {"confidence": 0.8, "detector_is_right": 0, "bbox_xyxy": [0, 0, 5, 5]}
    right = {"confidence": 0.9, "detector_is_right": 1, "bbox_xyxy": [10, 10, 15, 15]}
    groups = {5: {0: left}, 7: {1: right}}
    detections = fusion._complete_detections(groups, 4, suppress_duplicate_boxes=True)
    assert set(detections) == {5, 7}
    assert fusion._ordered_hand_pairs(detections, groups, "strict") == [(5, 7)]
    monkeypatch.setattr(fusion, "_candidate_error", lambda *args: 0.0)
    selected, _ = fusion._match_camera("camera0", groups, {0: {"points": []}, 1: {"points": []}},
                                       None, 20, 4, "strict")
    assert selected == {0: left, 1: right}


def test_gem_rejects_unexpected_joint_layout():
    with pytest.raises(ValueError, match="25 SOMA"):
        gem.adapt_frame({"instances": [{"side": "left", "bbox": [1, 2, 3],
                                        "keypoints_2d": np.zeros((21, 2))}]},
                        {"sync_index": 0, "camera0_frame_index": 0,
                         "camera0_timestamp_ns": 0}, "camera0")


def test_find_single_part_ego4_recording(tmp_path):
    part = tmp_path / "part0001"
    part.mkdir()
    video = part / "session_camera_left_part0001.mp4"
    pts = part / "session_camera_left_part0001_pts.csv"
    video.touch()
    pts.touch()
    assert session_stream_file(tmp_path, "left", timestamps=False) == video
    assert session_stream_file(tmp_path, "left", timestamps=True) == pts
    (part / "session_camera_left_part0002.mp4").touch()
    with pytest.raises(RuntimeError, match="Multiple parts"):
        session_stream_file(tmp_path, "left", timestamps=False)


def test_gem_uses_two_hypotheses_for_each_original_detection():
    boxes = np.array([[10, 20, 40, 60], [100, 120, 160, 150]], dtype=np.float32)
    instances = gem.make_hypotheses(boxes, [0.9, 0.7], [1, 0])
    assert [item["side"] for item in instances] == ["left", "right", "left", "right"]
    assert [item["track_id"] for item in instances] == [0, 1, 2, 3]
    assert instances[0]["bbox"] == [25, 40, 80]
    assert instances[2]["bbox"] == [130, 135, 160]
    points = np.column_stack((np.arange(25), np.arange(25) + 100))
    frame = {"instances": [
        {"track_id": item["track_id"], "side": item["side"], "keypoints_2d": points,
         "projection_valid": np.ones(25, dtype=bool), "camera_translation": [0, 0, 1]}
        for item in instances
    ]}
    row = {"sync_index": "3", "camera0_frame_index": "9", "camera0_timestamp_ns": "20"}
    result = gem.serialize_hypotheses(frame, row, "camera0", boxes, [0.9, 0.7], [1, 0])
    assert [(h["detection_index"], h["is_right"], h["detector_is_right"]) for h in result["hands"]] == [
        (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0)
    ]
    assert result["hands"][0]["bbox_xyxy"] == [10, 20, 40, 60]


def test_gem_detector_uses_baseline_frame_batching():
    class Reader:
        def read(self, index):
            return f"frame{index}"

    calls = []
    def detector(images, **options):
        calls.append((images, options))
        return [f"detection{n}" for n in range(len(images))]

    rows = [{"camera0_frame_index": str(n)} for n in (1, 3, 4)]
    items = list(gem.iter_detected_frames(Reader(), rows, "camera0", detector, 0.2, True, 2))
    assert [item[1] for item in items] == ["frame1", "frame3", "frame4"]
    assert [len(call[0]) for call in calls] == [2, 1]
    assert calls[0][1] == {"conf": 0.2, "iou": 0.5, "verbose": False, "half": True}


def test_mesh_uses_inference_projection_and_hides_negative_depth():
    from render_gem_multiview import project_mesh
    pixels, valid = project_mesh([[1, 2, 2], [0, 0, -1]],
                                 [[100, 0, 50], [0, 100, 60], [0, 0, 1]])
    np.testing.assert_allclose(pixels[0], [100, 160])
    assert valid.tolist() == [True, False]
    assert np.isnan(pixels[1]).all()


def test_mesh_follows_selected_hypothesis_not_physical_side_label():
    from render_gem_multiview import select_mesh_instance
    left = {"track_id": 4, "keypoints_2d": np.zeros((25, 2))}
    right = {"track_id": 5, "keypoints_2d": np.ones((25, 2))}
    unrelated = {"track_id": 6, "keypoints_2d": np.ones((25, 2))}
    frame = {"instances": [left, right, unrelated]}
    assert select_mesh_instance(frame, {"hypothesis_index": 4}) is left
    assert select_mesh_instance(frame, {
        "detection_index": 2, "wilor_joints_2d": np.ones((21, 2)), "side": 0,
    }) is right
    assert select_mesh_instance(frame, {"hypothesis_index": 999}) is None


def test_mesh_held_track_uses_original_source_frame(monkeypatch):
    import render_gem_multiview as render
    old = {"track_id": 0, "vertices_camera": "original", "topology": "left", "intrinsics": "K"}
    current = dict(old, vertices_camera="unrelated new detection")
    overlay = render.GemMeshOverlay.__new__(render.GemMeshOverlay)
    overlay.frames = {"camera0": {7: {"instances": [old]}, 8: {"instances": [current]}}}
    overlay.topologies = {"camera0": {"left": "faces"}}
    overlay.stats = {"requested": 0, "drawn": 0, "missing": 0, "outside_image": 0}
    seen = []
    monkeypatch.setattr(render, "draw_mesh_surface", lambda image, xyz, *args: seen.append(xyz) or True)
    overlay.draw(None, "camera0", 8, {"mesh_source_frame_index": 7, "hypothesis_index": 0}, 0)
    assert seen == ["original"]
    assert overlay.stats["drawn"] == 1


def test_mesh_surface_changes_only_visible_region():
    from render_gem_multiview import draw_mesh_surface
    image = np.full((60, 60, 3), 80, dtype=np.uint8)
    vertices = np.array([[10, 10, 1], [40, 10, 1], [25, 40, 1]], dtype=float)
    assert draw_mesh_surface(image, vertices, [[0, 1, 2]], np.eye(3), 0)
    assert np.any(image[20, 25] != 80)
    assert np.all(image[:5] == 80)
    before = image.copy()
    assert not draw_mesh_surface(image, vertices * [1, 1, -1], [[0, 1, 2]], np.eye(3), 0)
    np.testing.assert_array_equal(image, before)
