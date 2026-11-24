from pathlib import Path
from typing import Optional, Union

from .mano_wrapper import MANO
from .wilor import WiLoR

from .discriminator import Discriminator


def resolve_mano_model_path(mano_path: Union[str, Path]) -> Path:
    """Resolve either a MANO model directory or an extracted MANO release."""
    mano_path = Path(mano_path).expanduser().resolve()
    candidates = [
        mano_path,
        mano_path / 'models',
        mano_path / 'mano_v1_2',
        mano_path / 'mano_v1_2' / 'models',
    ]
    for candidate in candidates:
        if (candidate / 'MANO_RIGHT.pkl').is_file():
            return candidate
    if mano_path.is_file() and mano_path.name == 'MANO_RIGHT.pkl':
        return mano_path
    raise FileNotFoundError(
        f'Could not find MANO_RIGHT.pkl under {mano_path}. '
        'Pass either the model file, its containing directory, or the extracted '
        'MANO release directory.'
    )


# Kept for callers created before the resolver became part of the public API.
_resolve_mano_model_path = resolve_mano_model_path


def load_wilor_config(
    cfg_path: Union[str, Path],
    mano_path: Optional[Union[str, Path]] = None,
):
    """Load the model config and resolve all local MANO paths."""
    from wilor.configs import get_config
    model_cfg = get_config(str(cfg_path), update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if ('vit' in model_cfg.MODEL.BACKBONE.TYPE) and ('BBOX_SHAPE' not in model_cfg.MODEL):
        
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()
        
        # Update config to be compatible with demo

    if ('DATA_DIR' in model_cfg.MANO):
        model_cfg.defrost()
        repo_root = Path(__file__).resolve().parents[2]
        if mano_path is None:
            mano_model_path = resolve_mano_model_path(repo_root / 'mano_data')
        else:
            mano_model_path = resolve_mano_model_path(mano_path)
        mean_params_path = repo_root / 'mano_data' / 'mano_mean_params.npz'
        model_cfg.MANO.DATA_DIR = str(mano_model_path)
        model_cfg.MANO.MODEL_PATH = str(mano_model_path)
        model_cfg.MANO.MEAN_PARAMS = str(mean_params_path)
        model_cfg.freeze()

    return model_cfg


def load_wilor(
    checkpoint_path,
    cfg_path,
    mano_path: Optional[Union[str, Path]] = None,
    adapter_path: Optional[Union[str, Path]] = None,
    init_renderer: bool = True,
    map_location: str = 'cpu',
):
    """Load a base WiLoR checkpoint and an optional lightweight adapter."""
    from .adapter import load_adapter_weights

    print('Loading ', checkpoint_path)
    model_cfg = load_wilor_config(cfg_path, mano_path=mano_path)

    model = WiLoR.load_from_checkpoint(
        checkpoint_path,
        strict=False,
        cfg=model_cfg,
        init_renderer=init_renderer,
        map_location=map_location,
    )
    if adapter_path is not None:
        load_adapter_weights(adapter_path, model)
    return model, model_cfg
