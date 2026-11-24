from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from wilor.models.lora import LoRALinear, iter_lora_modules
from wilor.training.config import load_training_config
from wilor.training.recipes import (
    DomainAdaptRecipe,
    JointLoRAFineTuneRecipe,
    RefinerFineTuneRecipe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(4, 4)
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.tanh(self.qkv(value)))


class _Mlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(value)))


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.mlp = _Mlp()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.attn(value) + self.mlp(value)


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(32)])
        self.decpose = torch.nn.Linear(4, 15 * 9)
        self.decshape = torch.nn.Linear(4, 10)
        self.deccam = torch.nn.Linear(4, 3)
        self.register_buffer("init_betas", torch.zeros((1, 10)))
        self.use_checkpoint = False

    def pose(self, image: torch.Tensor) -> torch.Tensor:
        value = image.mean(dim=(-2, -1))[:, :1].repeat(1, 4).unsqueeze(1)
        for block in self.blocks:
            value = block(value)
        return self.decpose(value.mean(dim=1)).reshape(-1, 15, 3, 3)


class _Mano(torch.nn.Module):
    def forward(self, hand_pose: torch.Tensor, **_: object) -> SimpleNamespace:
        batch_size = hand_pose.shape[0]
        base = torch.zeros((batch_size, 21, 3), device=hand_pose.device)
        base[:, 5] = torch.tensor([1.0, 0.0, 0.0], device=hand_pose.device)
        base[:, 9] = torch.tensor([0.0, 1.0, 0.0], device=hand_pose.device)
        base[:, 17] = torch.tensor([-1.0, 0.0, 0.0], device=hand_pose.device)
        tip = torch.zeros_like(base)
        tip[:, 8, 2] = hand_pose[:, 0, 0, 0]
        return SimpleNamespace(joints=base + tip)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.refine_net = torch.nn.Linear(2, 2)
        self.mano = _Mano()
        self.refine_called = False

    def forward_coarse_step(self, batch, train=False):
        del train
        pose = self.backbone.pose(batch["img"])
        batch_size = pose.shape[0]
        return {
            "coarse_mano_params": {
                "hand_pose": pose,
                "global_orient": torch.eye(3).reshape(1, 1, 3, 3).repeat(
                    batch_size, 1, 1, 1
                ),
                "betas": torch.zeros((batch_size, 10)),
            },
            "coarse_keypoints_3d": torch.zeros((batch_size, 21, 3)),
        }

    def forward_step(self, batch, train=False):
        del train
        self.refine_called = True
        batch_size = batch["img"].shape[0]
        prediction = torch.zeros((batch_size, 21, 3))
        return {"pred_keypoints_3d": prediction}


def test_domain_recipe_trains_all_block_lora_and_skips_refiner() -> None:
    config = load_training_config(REPO_ROOT / "configs/glove/domain_adapt.yaml")
    model = _Model()
    recipe = DomainAdaptRecipe(config)
    recipe.configure_model(model)
    modules = list(iter_lora_modules(model.backbone))
    assert len(modules) == 32 * 4
    assert all(isinstance(module, LoRALinear) for module in modules)
    assert {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    } == {
        name for name, _ in model.named_parameters()
        if name.endswith(("lora_A.weight", "lora_B.weight"))
    }
    output = recipe.forward(
        model, {"img": torch.randn((2, 3, 8, 8))}, training=True
    )
    output["coarse_local_keypoints_3d"].square().mean().backward()
    assert not model.refine_called
    assert any(module.lora_B.weight.grad is not None for module in modules)
    assert model.refine_net.weight.grad is None
    assert model.backbone.deccam.weight.grad is None


def test_domain_recipe_can_train_full_vit_and_skip_refiner() -> None:
    config = load_training_config(
        REPO_ROOT
        / "configs/glove/domain_adapt_opentouch_original_full_vit.yaml"
    )
    model = _Model()
    recipe = DomainAdaptRecipe(config)
    recipe.configure_model(model)
    assert not list(iter_lora_modules(model.backbone))
    assert all(
        parameter.requires_grad for block in model.backbone.blocks
        for parameter in block.parameters()
    )
    assert not model.backbone.decpose.weight.requires_grad
    assert not model.backbone.decshape.weight.requires_grad
    assert not model.backbone.deccam.weight.requires_grad
    assert not model.refine_net.weight.requires_grad
    recipe.set_mode(model, training=True)
    assert model.backbone.training
    assert not model.backbone.decpose.training
    output = recipe.forward(
        model, {"img": torch.randn((2, 3, 8, 8))}, training=True
    )
    output["coarse_local_keypoints_3d"].square().mean().backward()
    assert not model.refine_called
    assert model.backbone.blocks[0].attn.qkv.weight.grad is not None
    assert model.refine_net.weight.grad is None


def test_refiner_recipe_preserves_existing_head_and_refiner_policy() -> None:
    config = load_training_config(
        REPO_ROOT / "configs/glove/refiner_finetune.yaml"
    )
    model = _Model()
    recipe = RefinerFineTuneRecipe(config)
    recipe.configure_model(model)
    assert model.backbone.decpose.weight.requires_grad
    assert model.backbone.decshape.weight.requires_grad
    assert model.backbone.deccam.weight.requires_grad
    assert model.refine_net.weight.requires_grad
    assert not model.backbone.blocks[0].attn.qkv.weight.requires_grad
    recipe.forward(model, {"img": torch.randn((1, 3, 8, 8))}, training=True)
    assert model.refine_called


def test_joint_lora_recipe_runs_full_path_and_updates_all_three_groups() -> None:
    class FullModel(_Model):
        def forward_step(self, batch, train=False):
            del train
            self.refine_called = True
            pose = self.backbone.pose(batch["img"])
            coarse = pose.mean(dim=(1, 2, 3))
            refined = self.refine_net(torch.stack((coarse, coarse), dim=-1))
            points = (coarse + refined[:, 0]).reshape(-1, 1, 1)
            return {"pred_keypoints_3d": points.expand(-1, 21, 3)}

    config = load_training_config(
        REPO_ROOT / "configs/glove/joint_lora_finetune.yaml"
    )
    model = FullModel()
    recipe = JointLoRAFineTuneRecipe(config)
    recipe.configure_model(model)
    recipe.set_mode(model, training=True)
    optimizer = recipe.build_optimizer(model)
    assert [group["lr"] for group in optimizer.param_groups] == [
        config.vit_lr, config.coarse_lr, config.refiner_lr,
    ]
    assert len(list(iter_lora_modules(model.backbone))) == 32 * 4
    assert not model.backbone.blocks[0].attn.qkv.base.weight.requires_grad
    assert model.backbone.decpose.weight.requires_grad
    assert model.backbone.decshape.weight.requires_grad
    assert model.backbone.deccam.weight.requires_grad
    assert model.refine_net.weight.requires_grad
    assert model.backbone.blocks[0].attn.qkv.training
    output = recipe.forward(
        model, {"img": torch.randn((2, 3, 8, 8))}, training=True
    )
    output["pred_keypoints_3d"].square().mean().backward()
    assert model.refine_called
    assert model.backbone.blocks[0].attn.qkv.lora_B.weight.grad is not None
    assert model.backbone.decpose.weight.grad is not None
    assert model.refine_net.weight.grad is not None


def test_lora_stage2_freezes_imported_lora() -> None:
    stage1 = load_training_config(REPO_ROOT / "configs/glove/lora_stage1.yaml")
    stage2 = load_training_config(REPO_ROOT / "configs/glove/lora_stage2.yaml")
    model = _Model()
    DomainAdaptRecipe(stage1).configure_model(model)
    RefinerFineTuneRecipe(stage2).configure_model(model)
    assert all(not p.requires_grad for module in iter_lora_modules(model.backbone)
               for p in (module.lora_A.weight, module.lora_B.weight))
    assert model.backbone.decpose.weight.requires_grad
    assert model.refine_net.weight.requires_grad
