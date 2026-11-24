"""WiLoR's square perspective input sampled directly from a calibrated raw box."""
import numpy as np

from tools.hand_finetune.crop_geometry import PerspectiveCrop


def rectify_hand(image, raw_box, K, distortion, resolution, rescale, left):
    # The backbone uses the central 192 columns of its 256-square input.
    view = PerspectiveCrop.from_box(raw_box, K, distortion, resolution,
                                    padding=rescale, width_factor=256 / 192)
    patch = view.rectify(image)
    if left:
        patch = np.ascontiguousarray(patch[:, ::-1])
    return view, patch


def view_mano_orientation(view, native_orientation, right):
    # The inverse of physical_mano_orientation acts on canonical MANO parameters.
    native_from_view = view.physical_mano_orientation(np.eye(3), right)
    return (native_from_view.T @ native_orientation).astype(np.float32)
