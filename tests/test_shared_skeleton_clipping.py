from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ego_hand_inference.render_shared_wilor_multiview import clip_projected_bone, draw_shared_skeleton


@pytest.mark.parametrize('a,b,expected', [
    ((50, 130), (50, 40), ((50, 99), (50, 40))),
    ((-30, 50), (130, 50), ((0, 50), (99, 50))),
    ((-20, -20), (20, 20), ((0, 0), (20, 20))),
    ((-20, 40), (-20, 80), None),
    ((1e12, 50), (-1e12, 50), ((99, 50), (0, 50))),
])
def test_clip_segments_with_offscreen_endpoints(a, b, expected):
    assert clip_projected_bone(a, b, 100, 100) == expected


def test_offscreen_wrist_keeps_visible_palm_lines_without_inventing_boundary_joint():
    pixels = np.full((21, 2), np.nan)
    pixels[0], pixels[5] = (50, 130), (50, 40)
    image = np.zeros((100, 100, 3), np.uint8)
    result = draw_shared_skeleton(image, pixels, np.isfinite(pixels).all(1), 0)
    assert result == {'visible_joint_count': 1, 'drawn_bone_count': 1, 'clipped_bone_count': 1}
    np.testing.assert_array_equal(image[80, 50], [255, 255, 0])
    assert not image[99, 56].any()  # No circle is placed at the clipped line end.
    assert pixels[0, 1] == 130  # Projection/model data remains unchanged.


def test_invalid_projection_does_not_connect_to_visible_joint():
    pixels = np.full((21, 2), np.nan)
    pixels[0], pixels[5] = (50, 130), (50, 40)
    valid = np.isfinite(pixels).all(1)
    valid[0] = False
    image = np.zeros((100, 100, 3), np.uint8)
    result = draw_shared_skeleton(image, pixels, valid, 0)
    assert result['drawn_bone_count'] == 0 and not image[80].any()
