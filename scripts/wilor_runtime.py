"""WiLoR model loading and projection for offline hand inference."""

from __future__ import annotations
import os
from pathlib import Path
import sys
import numpy as np
from inference_common import choose_device, load_detector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WILOR_ROOT = (
    Path(os.environ.get("EGO_WILOR_ROOT", PROJECT_ROOT / "third_party/WiLoR"))
    .expanduser()
    .resolve()
)
sys.path.insert(0, str(WILOR_ROOT))
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.configs import get_config
from wilor.models import WiLoR
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full


def load_wilor_model(checkpoint: Path, config: Path, mano_model_dir: Path):
    from check_python_environment import import_chumpy_compatibly

    import_chumpy_compatibly()
    model_cfg = get_config(str(config), update_cachedir=True)
    model_cfg.defrost()
    if "vit" in model_cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    model_cfg.MANO.DATA_DIR = str(mano_model_dir.resolve())
    model_cfg.MANO.MODEL_PATH = str(mano_model_dir.resolve())
    model_cfg.MANO.MEAN_PARAMS = str(
        (WILOR_ROOT / "mano_data" / "mano_mean_params.npz").resolve()
    )
    model_cfg.freeze()
    model = WiLoR.load_from_checkpoint(
        str(checkpoint), strict=False, cfg=model_cfg, map_location="cpu"
    )
    return model, model_cfg


def project_points(
    points: np.ndarray,
    camera_translation: np.ndarray,
    focal_length: float,
    image_size: np.ndarray,
) -> np.ndarray:
    points = points + camera_translation
    points = points / points[..., -1:]
    camera_center = np.asarray([image_size[0] / 2.0, image_size[1] / 2.0])
    return np.column_stack(
        (
            focal_length * points[:, 0] + camera_center[0],
            focal_length * points[:, 1] + camera_center[1],
        )
    )
