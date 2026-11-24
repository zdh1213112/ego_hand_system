from __future__ import annotations

from pathlib import Path

import pytest
import torch
from yacs.config import CfgNode

from wilor.models.adapter import (
    TRAINABLE_FORMAT,
    load_adapter_weights,
    load_trainable_weights,
    resume_trainable_checkpoint,
    save_full_checkpoint,
    save_trainable_checkpoint,
    trainable_state_dict,
)
from wilor.models.lora import LoRAConfig, inject_vit_lora, set_lora_trainable


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(3, 3)
        self.proj = torch.nn.Linear(3, 3)


class _Mlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 3)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.mlp = _Mlp()


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.decpose = torch.nn.Linear(3, 3)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.refine_net = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.BatchNorm1d(3),
        )


@pytest.mark.parametrize("with_lora", [False, True])
def test_full_checkpoint_restores_frozen_weights_and_inference_branches(
    tmp_path: Path, monkeypatch, with_lora: bool,
) -> None:
    from wilor.models import wilor as implementation

    monkeypatch.setattr(implementation, "create_backbone", lambda cfg: _Backbone())
    monkeypatch.setattr(implementation, "RefineNet", lambda *a, **kw: torch.nn.BatchNorm1d(3))
    monkeypatch.setattr(implementation, "MANO", lambda **kw: torch.nn.Identity())
    cfg = CfgNode({"MODEL": {"BACKBONE": {}}, "LOSS_WEIGHTS": {"ADVERSARIAL": 0.0}, "MANO": {}})
    model = implementation.WiLoR(cfg, init_renderer=False).eval()
    if with_lora:
        _configure_stage1(model)
        model.physical_camera_enabled = True
        model.physical_camera_head.camera_model = "physical_kb"
        model.physical_camera_head.native_root_prior = "bbox_geometry"
        model.marker_head_enabled = True
        with torch.no_grad():
            model.backbone.blocks[0].attn.qkv.lora_B.weight.fill_(0.45)
    model.refine_net.running_mean.fill_(0.25)
    path = tmp_path / "best_full.ckpt"
    save_full_checkpoint(path, model, step=8)

    restored = implementation.WiLoR.load_from_checkpoint(
        path, init_renderer=False, map_location="cpu", strict=True,
    ).eval()
    assert restored.physical_camera_enabled == with_lora
    assert restored.marker_head_enabled == with_lora
    if with_lora:
        assert restored.physical_camera_head.camera_model == "physical_kb"
        assert restored.physical_camera_head.native_root_prior == "bbox_geometry"
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
    x = torch.randn(2, 3)
    torch.testing.assert_close(restored.backbone.blocks[0].attn.qkv(x),
                               model.backbone.blocks[0].attn.qkv(x), rtol=0, atol=0)


def _configure_stage1(model: _Model) -> torch.optim.Optimizer:
    for parameter in model.parameters():
        parameter.requires_grad = False
    inject_vit_lora(
        model.backbone,
        LoRAConfig(rank=2, alpha=4, dropout=0.0),
    )
    set_lora_trainable(model.backbone, True)
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )


def test_trainable_checkpoint_reconstructs_lora_for_stage_initialization(
    tmp_path: Path,
) -> None:
    model = _Model()
    optimizer = _configure_stage1(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_B.weight"):
                parameter.fill_(0.25)
    expected = trainable_state_dict(model)
    checkpoint = tmp_path / "stage1.pt"
    save_trainable_checkpoint(
        checkpoint,
        model,
        optimizer,
        step=3,
        base_checkpoint=tmp_path / "base.ckpt",
        recipe="domain_adapt",
        training_metadata={"signature": "stage1"},
    )
    fresh = _Model()
    # Public inference loading auto-detects the v2 format and reconstructs LoRA.
    payload = load_adapter_weights(checkpoint, fresh)
    assert payload["format"] == TRAINABLE_FORMAT
    for name, value in expected.items():
        torch.testing.assert_close(fresh.state_dict()[name], value)

    # Stage 2 may freeze the imported LoRA while training different heads; the
    # next checkpoint must still carry the adapted visual representation.
    for parameter in fresh.parameters():
        parameter.requires_grad = False
    for parameter in fresh.refine_net.parameters():
        parameter.requires_grad = True
    stage2_optimizer = torch.optim.AdamW(fresh.refine_net.parameters(), lr=1e-3)
    stage2_path = tmp_path / "stage2.pt"
    save_trainable_checkpoint(
        stage2_path,
        fresh,
        stage2_optimizer,
        step=1,
        base_checkpoint=tmp_path / "base.ckpt",
        recipe="refiner_finetune",
    )
    stage2 = torch.load(stage2_path, map_location="cpu", weights_only=False)
    assert any(name.endswith("lora_B.weight") for name in stage2["state_dict"])
    assert any(name.startswith("refine_net") for name in stage2["state_dict"])


def test_trainable_checkpoint_restores_trainable_module_buffers(
    tmp_path: Path,
) -> None:
    model = _Model()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.refine_net.parameters():
        parameter.requires_grad = True
    model.refine_net.train()
    model.refine_net(torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 8.0, 16.0],
    ]))

    expected = trainable_state_dict(model)
    buffer_names = {
        name for name in expected
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    assert buffer_names == {
        "refine_net.1.running_mean",
        "refine_net.1.running_var",
        "refine_net.1.num_batches_tracked",
    }

    optimizer = torch.optim.AdamW(model.refine_net.parameters(), lr=1e-3)
    checkpoint = tmp_path / "stage2-with-buffers.pt"
    save_trainable_checkpoint(
        checkpoint,
        model,
        optimizer,
        step=5,
        base_checkpoint=tmp_path / "base.ckpt",
        recipe="refiner_finetune",
    )

    fresh = _Model()
    load_trainable_weights(checkpoint, fresh)
    for name in buffer_names:
        torch.testing.assert_close(fresh.state_dict()[name], expected[name])


def test_stage2_checkpoint_carries_frozen_full_vit_initialization(
    tmp_path: Path,
) -> None:
    stage1 = _Model()
    for parameter in stage1.parameters():
        parameter.requires_grad = False
    for block in stage1.backbone.blocks:
        for parameter in block.parameters():
            parameter.requires_grad = True
    with torch.no_grad():
        stage1.backbone.blocks[0].attn.qkv.weight.fill_(0.375)
    stage1_optimizer = torch.optim.AdamW(
        [parameter for parameter in stage1.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    stage1_path = tmp_path / "stage1-full-vit.pt"
    save_trainable_checkpoint(
        stage1_path,
        stage1,
        stage1_optimizer,
        step=1,
        base_checkpoint=tmp_path / "base.ckpt",
        recipe="domain_adapt",
    )

    stage2 = _Model()
    load_trainable_weights(stage1_path, stage2)
    for parameter in stage2.parameters():
        parameter.requires_grad = False
    for parameter in stage2.refine_net.parameters():
        parameter.requires_grad = True
    stage2_optimizer = torch.optim.AdamW(stage2.refine_net.parameters(), lr=1e-4)
    stage2_path = tmp_path / "stage2-refiner.pt"
    save_trainable_checkpoint(
        stage2_path,
        stage2,
        stage2_optimizer,
        step=1,
        base_checkpoint=tmp_path / "base.ckpt",
        recipe="refiner_finetune",
    )
    payload = torch.load(stage2_path, map_location="cpu", weights_only=False)
    carried_name = "backbone.blocks.0.attn.qkv.weight"
    assert carried_name in payload["state_dict"]
    torch.testing.assert_close(
        payload["state_dict"][carried_name],
        torch.full_like(payload["state_dict"][carried_name], 0.375),
    )

    inference = _Model()
    load_trainable_weights(stage2_path, inference)
    torch.testing.assert_close(
        inference.state_dict()[carried_name],
        torch.full_like(inference.state_dict()[carried_name], 0.375),
    )


def test_injected_lora_inherits_base_device_and_dtype(tmp_path: Path) -> None:
    model = _Model()
    optimizer = _configure_stage1(model)
    checkpoint = tmp_path / "stage1.pt"
    save_trainable_checkpoint(
        checkpoint,
        model,
        optimizer,
        step=1,
        base_checkpoint=tmp_path / "base.ckpt",
        recipe="domain_adapt",
    )

    fresh = _Model().to(dtype=torch.float64)
    load_trainable_weights(checkpoint, fresh)
    lora_parameters = [
        parameter for name, parameter in fresh.named_parameters()
        if name.endswith(("lora_A.weight", "lora_B.weight"))
    ]
    assert lora_parameters
    assert all(parameter.dtype == torch.float64 for parameter in lora_parameters)


def test_trainable_resume_is_strict_and_checks_shapes(tmp_path: Path) -> None:
    model = _Model()
    optimizer = _configure_stage1(model)
    checkpoint = tmp_path / "valid.pt"
    save_trainable_checkpoint(
        checkpoint, model, optimizer, 1, tmp_path / "base.ckpt", "domain_adapt"
    )
    fresh = _Model()
    load_trainable_weights(checkpoint, fresh)
    fresh_optimizer = _configure_stage1(fresh)
    payload = resume_trainable_checkpoint(checkpoint, fresh, fresh_optimizer)
    assert payload["step"] == 1

    broken = torch.load(checkpoint, map_location="cpu", weights_only=False)
    name = next(iter(broken["state_dict"]))
    broken["state_dict"][name] = torch.zeros(1)
    broken_path = tmp_path / "broken.pt"
    torch.save(broken, broken_path)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_trainable_weights(broken_path, _Model())
