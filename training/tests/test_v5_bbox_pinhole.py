"""Acceptance tests for the opt-in calibrated crop training recipe."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import evaluate_glove as offline
from tools.hand_finetune.crop_geometry import PerspectiveCrop
from wilor.datasets.glove_dataset import GloveDataset, SyncFrameBatchSampler, write_manifest_rows
from wilor.models.adapter import FULL_FORMAT
from wilor.models.mano_wrapper import MANO
from wilor.models.wilor import WiLoR
from wilor.training import engine
from wilor.training.config import load_training_config, resolved_training_config
from wilor.training.losses import (UNIFIED_LOSS_NAMES, V5_LOSS_SOURCES, UnifiedLossComputer,
                                   anatomy_bend, unified_loss_config_from_mapping)
from test_glove_dataset import _write_sample
from test_visualization import _EvaluationModel

TRAINING = Path(__file__).resolve().parents[1]
CONFIGS = TRAINING / "configs/glove"
V5 = CONFIGS / "nokov_v5_bbox_pinhole.yaml"


def native_sample(root):
    row = _write_sample(root, 0, 1, 10, "camera0")
    row.update(source_group="example")
    label_path = root / row["label"]
    data = np.load(label_path, allow_pickle=True).item()
    distortion = np.array([0.123456789012345, -0.034, 0.005, 0.0])
    data.update(image_space="native_raw", camera_model="KB", distortion=distortion)
    np.save(label_path, data)
    (root / "summary.json").write_text(json.dumps({"image_space": "native_raw"}))
    manifest = root / "val.jsonl"
    write_manifest_rows(manifest, [row])
    return manifest, row, distortion


def test_skip_invalid_proposals_and_keep_other_hypotheses(monkeypatch):
    monkeypatch.syspath_prepend(str(TRAINING.parent / "scripts"))
    monkeypatch.syspath_prepend(str(TRAINING.parent / "inference"))
    from ego_hand_inference.native_glove_observations import run_frames
    boxes = SimpleNamespace(conf=torch.ones(2), cls=torch.ones(2, dtype=torch.long),
                            xyxy=torch.tensor([[252, 35, 290, 65], [30, 34, 60, 65]]).float())
    detector = SimpleNamespace(predict=lambda *a, **kw: [SimpleNamespace(boxes=boxes)])

    def forward(batch, train=False):
        count = len(batch["img"])
        uv = torch.zeros(count, 21, 2)
        uv[batch["side"] == 0] = float("nan")
        return {"pred_keypoints_2d": uv, "pred_keypoints_3d": torch.zeros(count, 21, 3),
                "pred_root_camera_m": torch.ones(count, 3),
                "pred_mano_params": {"global_orient": torch.eye(3).repeat(count, 1, 1, 1)}}

    args = SimpleNamespace(batch_size=2, max_detections_per_class=0, crop_coefficients=None,
                           rectified=True, rescale=2.)
    cfg = SimpleNamespace(MODEL=SimpleNamespace(IMAGE_SIZE=64, IMAGE_MEAN=[0]*3, IMAGE_STD=[1]*3))
    cal = {"K": np.array([[40, 0, 40], [0, 40, 40], [0, 0, 1]], np.float32),
           "D": np.zeros(4), "size": np.array([320, 120])}
    counts = Counter()
    with ThreadPoolExecutor(2) as pool:
        result = run_frames([np.zeros((120, 320, 3), np.uint8)], detector, SimpleNamespace(forward_step=forward),
                            cfg, cal, .3, args, torch.device("cpu"), pool, counts)
    assert [(h["detection_index"], h["is_right"]) for h in result[0]] == [(1, 1)]
    assert counts["skipped_invalid_proposals"] == 3


@pytest.mark.parametrize("use_adapter,factor", [(True, 1.6), (False, 1.8), (False, 2.)])
def test_offline_evaluation_uses_training_scale_and_double_distortion(tmp_path, monkeypatch, use_adapter, factor):
    manifest, _, distortion = native_sample(tmp_path)
    base = tmp_path / "full.ckpt"
    mods = {"physical_camera": True, "physical_camera_model": "kb_bbox_pinhole"}
    if factor != 2.:
        mods["rescale_factor"] = 1.8
    torch.save({"format": FULL_FORMAT, "model_modifications": mods}, base)
    adapter = tmp_path / "adapter.pt"
    torch.save({"training_metadata": {"config": {"data": {"rescale_factor": factor}}}}, adapter)
    def load_model(cfg, device):
        model = _EvaluationModel()
        model.physical_camera_head = SimpleNamespace()
        model.marker_head_enabled = False
        WiLoR.on_load_checkpoint(model, torch.load(cfg.checkpoint, weights_only=False))
        return model
    monkeypatch.setattr(offline, "load_glove_model", load_model)
    monkeypatch.setattr(offline, "load_adapter_weights", lambda path, model: torch.load(path, weights_only=False))
    monkeypatch.setattr(offline, "set_finetune_mode", lambda *a, **kw: None)
    scales = []
    def dataset(*args, **kwargs):
        scales.append(kwargs["rescale_factor"])
        return GloveDataset(*args, **kwargs)
    monkeypatch.setattr(offline, "GloveDataset", dataset)
    captured = []
    native_pixels = PerspectiveCrop.native_pixels
    def project(view, pixels):
        captured.append(view.distortion.copy())
        return native_pixels(view, pixels)
    monkeypatch.setattr(PerspectiveCrop, "native_pixels", project)
    args = ["evaluate_glove.py", "--root", str(tmp_path), "--manifest", str(manifest), "--device", "cpu",
            "--checkpoint", str(base), "--model-config", str(V5), "--output", str(tmp_path / "eval"),
            "--no-save-overlays"]
    if use_adapter:
        args.extend(["--adapter", str(adapter)])
    monkeypatch.setattr(sys, "argv", args)
    assert offline.main() == 0
    assert scales == [factor] and captured
    for coefficients in captured:
        assert coefficients.dtype == np.float64
        np.testing.assert_array_equal(coefficients, distortion)
    sample = GloveDataset(manifest, camera_model="kb_bbox_pinhole")[0]
    assert sample["camera_distortion"].dtype == torch.float32


@pytest.mark.parametrize("name,signature", [
    ("nokov_v4_joint_lora", "e37787006cf1d0c940411eae1771176bcacb7fa6f7f7d0b004c77a7eb43e0ba5"),
    ("nokov_v4_bbox_pinhole", "bb545f866dd87b801a217d54abbf260b7826dd22534be08751ad4509b04cb8a6"),
])
def test_v4_resume_signature_at_46b7df8(monkeypatch, name, signature):
    cfg = load_training_config(CONFIGS / (name + ".yaml"))
    # Freeze only the machine-specific artifact descriptors. The rest of the
    # semantic payload and its golden hash were evaluated at 46b7df8.
    names = {cfg.checkpoint: "coarse_start.ckpt", cfg.model_config: "model_config.yaml"}
    monkeypatch.setattr(engine, "_artifact", lambda p: {"name": names.get(p, p.name)})
    def metadata(config):
        profile = {"supervision": {k: {"available": True} for k in config.losses}}
        losses = unified_loss_config_from_mapping(config.losses, profile, config.source)
        return engine._training_metadata(config, Path("dataset_profile.json"),
                                         (Path("train.jsonl"), Path("val.jsonl")), losses.effective)
    assert metadata(cfg)["signature"] == signature
    assert not cfg.augmentation_options() and not cfg.schedule_options()
    with_loss = replace(cfg, losses={**cfg.losses, "anatomy_bend": 0.})
    assert metadata(with_loss)["signature"] != signature
    assert metadata(with_loss)["config"]["losses"]["anatomy_bend"] == 0.


def test_real_mano_signed_flexion_and_finite_gradients():
    from smplx.lbs import batch_rodrigues
    mano = MANO(model_path=str(TRAINING.parent / "models/mano/MANO_RIGHT.pkl"),
                is_rhand=True, flat_hand_mean=True, use_pca=False)
    def joints(angles):
        rotations = batch_rodrigues(angles).reshape(1, 15, 3, 3)
        return mano(betas=torch.zeros(1, 10), hand_pose=rotations,
                    global_orient=torch.eye(3).reshape(1, 1, 3, 3)).joints
    neutral = joints(torch.zeros(15, 3))
    normal = F.normalize(torch.cross(neutral[:, 5] - neutral[:, 0], neutral[:, 17] - neutral[:, 0], dim=-1))
    axis = F.normalize(torch.cross(neutral[:, 9] - neutral[:, 0], normal, dim=-1))
    assert normal[0, 1] < -.9
    # Real open MANO fingertips have a small rest curvature.
    assert anatomy_bend(neutral) < .01
    for degrees, natural in [([90., 65., 45.], True), ([0., -30., -20.], False), ([30., 60., -35.], False)]:
        pose = torch.zeros(15, 3)
        pose[:12] = (torch.deg2rad(torch.tensor(degrees))[:, None] * axis).repeat(4, 1)
        pose.requires_grad_()
        penalty = anatomy_bend(joints(pose))
        if natural:
            assert penalty < 1e-6
        else:
            assert penalty > .1
            grad, = torch.autograd.grad(penalty, pose)
            assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_sync_groups_are_intact_without_duplicates():
    rows = [dict(source_group=g, sync_index=t, side=s, camera=c)
            for g in ("a", "b") for t in (0, 1) for s in (0, 1) for c in range(2 + t)]
    rows.extend(dict(source_group="a", sync_index=-1, side=0, camera=0) for _ in range(3))
    sampler = SyncFrameBatchSampler(rows, batch_size=5, seed=7)
    orders = []
    for _ in range(2):
        count = len(sampler)
        batches = list(sampler)
        assert len(batches) == count and max(map(len, batches)) <= 5
        assert sorted(i for batch in batches for i in batch) == list(range(len(rows)))
        batch_ids = {}
        for j, batch in enumerate(batches):
            for i in batch:
                row = rows[i]
                if row["sync_index"] >= 0:
                    key = row["source_group"], row["sync_index"], row["side"]
                    assert batch_ids.setdefault(key, j) == j
        orders.append(batches)
    assert orders[0] != orders[1]


def test_optional_losses_masks_units_and_gradients():
    torch.manual_seed(224)
    def variable(*shape):
        return torch.randn(shape, requires_grad=True)
    pred = variable(3, 21, 3)
    output = {"pred_keypoints_3d": pred, "pred_keypoints_2d": variable(3, 21, 2),
              "coarse_root_camera_m": variable(3, 3),
              "pred_mano_params": {"hand_pose": variable(3, 15, 3, 3), "betas": variable(3, 10)}}
    batch = {"source_group": ["s"] * 3, "sync_index": torch.tensor([1, 1, 2]), "side": torch.ones(3),
             "camera_root_m": torch.zeros(3, 3), "geometry_supervised": torch.tensor([1, 0, 1]),
             "keypoints_3d": torch.cat([torch.randn(3, 21, 3), torch.ones(3, 21, 1)], -1),
             "keypoints_2d": torch.cat([torch.randn(3, 21, 2), torch.ones(3, 21, 1)], -1)}
    batch["keypoints_2d"][1, 20, -1] = 0
    batch["keypoints_3d"][1, 20, -1] = 0
    weights = {k: float(k in V5_LOSS_SOURCES) for k in UNIFIED_LOSS_NAMES}
    loss_cfg = unified_loss_config_from_mapping(weights, {"supervision": {
        "keypoints_2d": {"available": True}, "keypoints_3d": {"available": True}}}, V5)
    for key in V5_LOSS_SOURCES:
        single = replace(loss_cfg, effective={k: float(k == key) for k in loss_cfg.effective})
        total, _ = UnifiedLossComputer(torch.nn.Identity(), single)(batch, output)
        inputs = [pred, output["pred_keypoints_2d"], output["coarse_root_camera_m"], *output["pred_mano_params"].values()]
        grads = torch.autograd.grad(total, inputs, allow_unused=True)
        assert total > 0 and all(torch.isfinite(g).all() for g in grads if g is not None)
        assert any(g.abs().sum() > 0 for g in grads if g is not None)
        if key == "coarse_physical_root_3d":
            assert not grads[2][1].any()
        if key.startswith("fingertips"):
            assert not grads[1 if key.endswith("2d") else 0][1, 20].any()
        if key == "multiview_pose_consistency":
            assert not grads[-1][2].any()
            batch["sync_index"] = torch.full((3,), -1)
            assert UnifiedLossComputer(torch.nn.Identity(), single)(batch, output)[0] == 0
    zeros = torch.zeros(2, 21, 3, requires_grad=True)
    anatomy_bend(zeros).backward()
    assert torch.isfinite(zeros.grad).all()


def test_recipe_and_scheduler():
    cfg = load_training_config(V5)
    assert not cfg.group_by_sync_frame and not cfg.augmentation_options()
    assert cfg.camera_model == "physical_perspective" and cfg.max_steps == 15000
    assert resolved_training_config(cfg)["data"]["pinhole_crop"] == "local_pinhole"
    assert cfg.lr_schedule == "constant" and cfg.init_weights is None
    assert cfg.local_export.detector is None and cfg.local_export.detector_cache is None
    assert cfg.losses["coarse_keypoints_2d"] == .005
    def scale(step):
        return engine.lr_multiplier(step, 15000, "cosine", 500, .05)
    assert scale(0) == 0 and scale(1) == pytest.approx(1 / 500)
    assert scale(500) == 1 and scale(15000) == .05
    assert scale(7750) == pytest.approx((1 + .05) / 2)
    assert engine.lr_multiplier(15000, 15000, "constant", 0, .05) == 1
    for kw in ({"blur_prob": 1.0}, {"lr_schedule": "linear"}, {"detector_probability": .5},
               {"camera_model": "virtual"}, {"data_type": "opentouch"}):
        with pytest.raises(ValueError):
            replace(cfg, **kw).validate()


@pytest.mark.parametrize("failure", ["bounds", "horizon"])
def test_augmented_crop_falls_back_to_the_original_geometry(tmp_path, monkeypatch, failure):
    from wilor.datasets import bbox_pinhole

    manifest, row, _ = native_sample(tmp_path)
    original = GloveDataset(manifest, camera_model="kb_bbox_pinhole")[0]
    shifted = original["raw_bbox"].numpy() + [5, 0, 5, 0]
    cache = tmp_path / "detections.jsonl"
    cache.write_text(json.dumps(dict(row, recording_id="example", schema_version=1, proposals=[
        {"bbox": shifted.tolist(), "score": .9, "side": 1}
    ])))
    inputs = {"detector_cache": cache}
    if failure == "horizon":
        offsets = tmp_path / "offsets.json"
        offsets.write_text(json.dumps([[.05, 0, 0, 0]]))
        inputs = {"offset_samples": offsets}
    dataset = GloveDataset(manifest, camera_model="kb_bbox_pinhole", detector_probability=1., **inputs)
    np.testing.assert_allclose(dataset[0]["raw_bbox"], shifted)
    calls = []
    rectify = bbox_pinhole.rectify_hand

    def reject_candidate(image, box, *args):
        calls.append(box.copy())
        if np.allclose(box, shifted):
            if failure == "bounds":
                raise ValueError("Box outside invertible KB calibration")
            view, patch = rectify(image, box, *args)
            # Reverse the view direction to put the labelled points behind the horizon.
            return replace(view, rotation=view.rotation * np.array([[1], [-1], [-1]])), patch
        return rectify(image, box, *args)

    monkeypatch.setattr(bbox_pinhole, "rectify_hand", reject_candidate)
    sample = dataset[0]
    np.testing.assert_allclose(calls, [shifted, original["raw_bbox"].numpy()])
    for key in ("img", "raw_bbox", "view_rotation", "keypoints_2d", "keypoints_3d", "camera_root_m"):
        torch.testing.assert_close(sample[key], original[key], rtol=0, atol=0)


def test_altered_appearance_is_training_only(tmp_path, monkeypatch):
    manifest, _, _ = native_sample(tmp_path)
    cfg = replace(load_training_config(CONFIGS / "nokov_v4_bbox_pinhole.yaml"), dataset_root=tmp_path, crop_jitter=0,
                  erasing_prob=.99, blur_prob=.99)
    cfg.mano_path = tmp_path
    (tmp_path / "MANO_RIGHT.pkl").write_bytes(b"asset identity for data-loader test")
    asset_hash = engine.sha256_file(tmp_path / "MANO_RIGHT.pkl")
    (tmp_path / "dataset_profile.json").write_text(json.dumps({
        "mano_assets": {"MANO_RIGHT.pkl": {"sha256": asset_hash}},
        "geometry_validation": {"valid": True}
    }))
    train, val, _ = engine.build_datasets(cfg, manifest, manifest)
    with monkeypatch.context() as m:
        m.setattr(np.random, "random", lambda: 0.)
        augmented = train[0]
    reference = GloveDataset(manifest, camera_model="kb_bbox_pinhole")[0]
    assert torch.equal(val[0]["img"], reference["img"])
    assert not torch.equal(augmented["img"], reference["img"])
    for key in ("raw_bbox", "keypoints_2d", "keypoints_3d"):
        assert torch.equal(augmented[key], reference[key])


@pytest.mark.parametrize("resume,mode,expect_reset", [
    (False, "physical_kb", True), (False, "kb_bbox_pinhole", False), (True, "physical_kb", False),
])
def test_only_cross_mode_initialization_resets_the_last_layer(tmp_path, monkeypatch, resume, mode, expect_reset):
    path = tmp_path / "weights.pt"
    path.touch()
    cfg = replace(load_training_config(CONFIGS / "nokov_v4_bbox_pinhole.yaml"),
                  init_weights=None if resume else path, resume=path if resume else None)
    net = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3))
    model = SimpleNamespace(physical_camera_head=SimpleNamespace(net=net))

    def load(path, model):
        for param in model.physical_camera_head.net.parameters():
            torch.nn.init.ones_(param)
        return {"model_modifications": {"physical_camera_model": mode}}

    monkeypatch.setattr(engine, "load_trainable_weights", load)
    engine._preload_weights(cfg, model)
    assert torch.all(net[0].weight == 1) and torch.all(net[0].bias == 1)
    assert all(torch.all(p == (0 if expect_reset else 1)) for p in net[-1].parameters())


def test_validation_errors_use_visible_points_and_actual_crop_size():
    truth3d = torch.zeros(2, 21, 4)
    truth2d = torch.zeros(2, 21, 3)
    truth3d[..., 3] = 1
    truth3d[1, 20, 3] = 0
    truth2d[0, 20, 2] = 1
    prediction = torch.zeros(2, 21, 3)
    prediction[0, 20, 0] = .009
    prediction[1, 20, 0] = 100
    crop_prediction = torch.full((2, 21, 2), .1)
    sums = engine._validation_errors({"pred_keypoints_3d": prediction, "pred_keypoints_2d": crop_prediction},
                                     {"keypoints_3d": truth3d, "keypoints_2d": truth2d,
                                      "img": torch.zeros(2, 3, 30, 40)})
    torch.testing.assert_close(sums["val_fingertip_mpjpe_mm"], torch.tensor([9., 9.]))
    torch.testing.assert_close(sums["val_pinky_tip_mm"], torch.tensor([9., 1.]))
    torch.testing.assert_close(sums["val_reproj_crop_px"], torch.tensor([5., 1.]))
