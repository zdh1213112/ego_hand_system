from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from wilor.datasets.glove_dataset import GloveDataset
from wilor.datasets.glove_dataset import write_manifest_rows
from wilor.datasets.opentouch_dataset import prepare_opentouch_dataset
from wilor.models.adapter import load_adapter_weights, save_adapter_checkpoint
from wilor.models import WiLoR, load_wilor_config
from wilor.models.lora import iter_lora_modules
from wilor.training.config import load_training_config
from wilor.training.evaluation import evaluate_glove_set
from wilor.training.engine import run_training
from wilor.training.glove import (
    GloveTrainer,
    GloveTrainingConfig,
    build_optimizer,
    load_glove_model,
    set_finetune_mode,
    trainable_parameters,
)
from wilor.training.losses import GloveLossComputer, load_glove_loss_config
from wilor.training.losses import local_keypoint_3d_loss
from wilor.training.recipes import DomainAdaptRecipe
from wilor.utils.hand_geometry import palm_scale_normalize


REPO_ROOT = Path(__file__).resolve().parents[1]


def _frozen_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.gpu
def test_real_model_gradient_checkpoint_and_grouped_evaluation(
    tmp_path: Path,
) -> None:
    if os.environ.get("WILOR_RUN_GPU_TESTS") != "1":
        pytest.skip("set WILOR_RUN_GPU_TESTS=1 to run the real GPU test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    required = [
        REPO_ROOT / "pretrained_models" / "wilor_final.ckpt",
        REPO_ROOT / "pretrained_models" / "model_config.yaml",
        REPO_ROOT / "mano_data" / "mano_v1_2" / "models" / "MANO_RIGHT.pkl",
        REPO_ROOT / "datasets" / "dataset" / "train.jsonl",
        REPO_ROOT / "datasets" / "dataset" / "val.jsonl",
        REPO_ROOT / "datasets" / "dataset" / "dataset_profile.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("real checkpoint, MANO, or glove dataset is missing")

    config = GloveTrainingConfig(
        root=REPO_ROOT / "datasets" / "dataset",
        checkpoint=required[0],
        model_config=required[1],
        mano_path=REPO_ROOT / "mano_data" / "mano_v1_2",
        output=tmp_path,
        max_steps=1,
    )
    device = torch.device("cuda")
    model = load_glove_model(config, device)
    assert sum(parameter.numel() for parameter in trainable_parameters(model)) == 8_340_768
    optimizer = build_optimizer(model, 1e-4, 1e-5, 1e-4)
    import json
    profile = json.loads(required[5].read_text(encoding="utf-8"))
    loss_config = load_glove_loss_config(
        REPO_ROOT / "configs" / "glove" / "stage1_keypoints.yaml", profile
    )
    trainer = GloveTrainer(
        model, optimizer, device, use_amp=True, grad_clip=1.0,
        loss_computer=GloveLossComputer(model, loss_config),
    )
    loader = DataLoader(
        GloveDataset(required[3], max_samples=1),
        batch_size=1,
        shuffle=False,
    )
    frozen_before = _frozen_fingerprint(model)
    loss, error = trainer.run_batch(next(iter(loader)), training=True)
    assert np_isfinite(loss) and np_isfinite(error)
    assert _frozen_fingerprint(model) == frozen_before

    coarse_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if name.startswith((
            "backbone.decpose", "backbone.decshape", "backbone.deccam"
        ))
    ]
    refiner_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if name.startswith("refine_net")
    ]
    for gradients in (coarse_gradients, refiner_gradients):
        assert gradients
        assert all(value is not None and torch.isfinite(value).all()
                   for value in gradients)
        assert sum(float(value.abs().sum()) for value in gradients) > 0

    adapter = tmp_path / "adapter.pt"
    save_adapter_checkpoint(adapter, model, optimizer, 1, required[0], loss)
    load_adapter_weights(adapter, model)
    set_finetune_mode(model, training=False)
    test_loader = DataLoader(
        GloveDataset(required[4], max_frames=2),
        batch_size=3,
        shuffle=False,
    )
    metrics = evaluate_glove_set(
        model, test_loader, device, tmp_path / "visualization", use_amp=True
    )
    assert metrics["visualization_frames"] == 2
    assert len(list((tmp_path / "visualization" / "overlays").glob("*.jpg"))) == 2


def np_isfinite(value: float) -> bool:
    return value not in (float("inf"), float("-inf")) and value == value


@pytest.mark.gpu
def test_real_domain_adapt_lora_step_skips_refinenet() -> None:
    if os.environ.get("WILOR_RUN_GPU_TESTS") != "1":
        pytest.skip("set WILOR_RUN_GPU_TESTS=1 to run the real GPU test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    decoded = REPO_ROOT / "datasets/opentouch_decoded"
    required = [
        REPO_ROOT / "pretrained_models/wilor_final.ckpt",
        REPO_ROOT / "pretrained_models/model_config.yaml",
        REPO_ROOT / "mano_data/mano_v1_2/models/MANO_RIGHT.pkl",
        decoded / "frames.jsonl",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("real checkpoint, MANO, or decoded OpenTouch data is missing")

    config = load_training_config(REPO_ROOT / "configs/glove/domain_adapt.yaml")
    model_config = load_wilor_config(config.model_config, config.mano_path)
    model = WiLoR.load_from_checkpoint(
        str(config.checkpoint),
        cfg=model_config,
        init_renderer=False,
        strict=False,
        map_location="cpu",
    )
    recipe = DomainAdaptRecipe(config)
    recipe.configure_model(model)
    model = model.cuda()
    recipe.set_mode(model, training=True)
    optimizer = recipe.build_optimizer(model)

    row = json.loads((decoded / "frames.jsonl").open().readline())
    image = cv2.imread(str(decoded / row["image"]), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    rgb = image[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (
        rgb - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    with np.load(decoded / row["clip_data"]) as clip:
        landmarks = clip["right_hand_landmarks"][int(row["clip_data_index"])]
    batch = {
        "img": torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).cuda(),
        "local_keypoints_3d": palm_scale_normalize(
            torch.from_numpy(landmarks.astype(np.float32)).unsqueeze(0)
        ).cuda(),
    }
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = recipe.forward(model, batch, training=True)
        loss = local_keypoint_3d_loss(
            output["coarse_local_keypoints_3d"], batch["local_keypoints_3d"]
        )
    loss.backward()
    optimizer.step()
    lora_modules = list(iter_lora_modules(model.backbone))
    assert len(lora_modules) == 32 * 4
    assert any(module.lora_B.weight.grad is not None for module in lora_modules)
    assert all(parameter.grad is None for parameter in model.refine_net.parameters())
    assert torch.isfinite(loss)


@pytest.mark.gpu
def test_unified_domain_engine_one_step(tmp_path: Path) -> None:
    if os.environ.get("WILOR_RUN_GPU_TESTS") != "1":
        pytest.skip("set WILOR_RUN_GPU_TESTS=1 to run the real GPU test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    decoded = tmp_path / "decoded"
    frame_rows = []
    bbox_rows = []
    for clip_index, clip_id in enumerate(("clip_a", "clip_b")):
        image_relative = Path("images") / "smoke" / clip_id / "000000.jpg"
        image_path = decoded / image_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(
            str(image_path), np.full((480, 640, 3), 80, dtype=np.uint8)
        )
        landmarks = np.zeros((1, 21, 3), dtype=np.float32)
        landmarks[0, 5] = [1.0, 0.0, 0.0]
        landmarks[0, 9] = [0.0, 1.0, 0.0]
        landmarks[0, 17] = [-1.0, 0.0, 0.0]
        landmarks[0, 8, 2] = 0.4 + clip_index * 0.1
        clip_relative = Path("clips") / "smoke" / f"{clip_id}.npz"
        clip_path = decoded / clip_relative
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            clip_path,
            right_hand_landmarks=landmarks,
            source_frame_indices=np.asarray([0], dtype=np.int64),
        )
        sample_id = f"smoke::{clip_id}::000000"
        common = {
            "sample_id": sample_id,
            "source_file": "smoke.hdf5",
            "clip_id": clip_id,
            "source_frame_index": 0,
        }
        frame_rows.append({
            **common,
            "clip_data_index": 0,
            "image": str(image_relative),
            "clip_data": str(clip_relative),
        })
        bbox_rows.append({**common, "bbox_xyxy": [200, 100, 440, 420]})
    write_manifest_rows(decoded / "frames.jsonl", frame_rows)
    bbox_path = tmp_path / "bboxes.jsonl"
    write_manifest_rows(bbox_path, bbox_rows)
    prepared = tmp_path / "prepared"
    prepare_opentouch_dataset(
        decoded, bbox_path, prepared, sample_fps=30, source_fps=30, val_ratio=0.5
    )

    config = load_training_config(REPO_ROOT / "configs/glove/domain_adapt.yaml")
    config.dataset_root = prepared
    config.train_manifest = None
    config.val_manifest = None
    config.output = tmp_path / "output"
    config.max_steps = 1
    config.batch_size = 1
    config.num_workers = 0
    config.device = "cuda:0"
    config.val_steps = 1
    config.val_batches = 1
    config.save_steps = 1
    result = run_training(config)
    assert result["status"] == "passed"
    assert result["current_step"] == 1
    checkpoint = torch.load(
        config.output / "last_trainable.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["format"] == "wilor_glove_trainable_v2"
    assert checkpoint["recipe"] == "domain_adapt"
    assert checkpoint["model_modifications"]["lora"]["rank"] == 16
