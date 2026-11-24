from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wilor.models.adapter import (
    ADAPTER_FORMAT,
    adapter_state_dict,
    load_adapter_weights,
    resume_adapter_checkpoint,
    save_adapter_checkpoint,
)


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decpose = torch.nn.Linear(2, 2)
        self.decshape = torch.nn.Linear(2, 2)
        self.deccam = torch.nn.Linear(2, 2)
        self.frozen = torch.nn.Linear(2, 2)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.refine_net = torch.nn.Linear(2, 2)


def _optimizer(model: _Model) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        list(model.backbone.decpose.parameters())
        + list(model.backbone.decshape.parameters())
        + list(model.backbone.deccam.parameters())
        + list(model.refine_net.parameters()),
        lr=1e-3,
    )


def test_adapter_round_trip_restores_weights_and_optimizer(tmp_path: Path) -> None:
    model = _Model()
    optimizer = _optimizer(model)
    optimizer.zero_grad()
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()

    expected = adapter_state_dict(model)
    frozen_before = model.backbone.frozen.weight.detach().clone()
    path = tmp_path / "adapter.pt"
    save_adapter_checkpoint(
        path,
        model,
        optimizer,
        step=7,
        base_checkpoint=tmp_path / "base.ckpt",
        val_loss=0.25,
        training_metadata={"signature": "same-stage"},
    )
    with torch.no_grad():
        for value in model.parameters():
            value.add_(10)

    payload = resume_adapter_checkpoint(path, model, optimizer)
    assert payload["format"] == ADAPTER_FORMAT
    assert payload["step"] == 7
    assert payload["val_loss"] == 0.25
    assert payload["training_metadata"] == {"signature": "same-stage"}
    for name, value in expected.items():
        torch.testing.assert_close(model.state_dict()[name], value)
    assert not torch.equal(model.backbone.frozen.weight, frozen_before)


def test_adapter_rejects_missing_or_wrong_shaped_values(tmp_path: Path) -> None:
    model = _Model()
    optimizer = _optimizer(model)
    valid_path = tmp_path / "valid.pt"
    save_adapter_checkpoint(
        valid_path,
        model,
        optimizer,
        step=1,
        base_checkpoint=tmp_path / "base.ckpt",
    )
    payload = torch.load(valid_path, map_location="cpu", weights_only=False)
    removed_name = next(iter(payload["state_dict"]))
    payload["state_dict"].pop(removed_name)
    missing_path = tmp_path / "missing.pt"
    torch.save(payload, missing_path)
    with pytest.raises(ValueError, match="missing"):
        load_adapter_weights(missing_path, model)

    payload = torch.load(valid_path, map_location="cpu", weights_only=False)
    name = next(iter(payload["state_dict"]))
    payload["state_dict"][name] = torch.zeros(1)
    shape_path = tmp_path / "shape.pt"
    torch.save(payload, shape_path)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_adapter_weights(shape_path, model)
