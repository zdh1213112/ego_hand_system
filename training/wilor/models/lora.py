"""Minimal LoRA support for WiLoR's ViT backbone."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    blocks: str | tuple[int, ...] = "all"
    targets: tuple[str, ...] = (
        "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2",
    )
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05

    def validate(self, depth: int | None = None) -> None:
        if self.blocks != "all":
            if not isinstance(self.blocks, tuple) or not self.blocks:
                raise ValueError("LoRA blocks must be 'all' or a non-empty tuple")
            if depth is not None and any(i < 0 or i >= depth for i in self.blocks):
                raise ValueError(f"LoRA block index is outside [0, {depth})")
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        allowed = {"attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"}
        unknown = set(self.targets).difference(allowed)
        if unknown:
            raise ValueError(f"unsupported LoRA targets: {sorted(unknown)}")

    def to_dict(self) -> dict:
        value = asdict(self)
        value["blocks"] = self.blocks if self.blocks == "all" else list(self.blocks)
        value["targets"] = list(self.targets)
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "LoRAConfig":
        blocks = value.get("blocks", "all")
        if isinstance(blocks, list):
            blocks = tuple(int(item) for item in blocks)
        return cls(
            blocks=blocks,
            targets=tuple(value.get("targets", cls.targets)),
            rank=int(value.get("rank", 16)),
            alpha=float(value.get("alpha", 32.0)),
            dropout=float(value.get("dropout", 0.05)),
        )


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_A = nn.Linear(
            base.in_features, rank, bias=False, **factory_kwargs
        )
        self.lora_B = nn.Linear(
            rank, base.out_features, bias=False, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / float(rank)
        self.enabled = True
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.base(value)
        if self.enabled:
            result = result + self.lora_B(self.lora_A(self.dropout(value))) * self.scaling
        return result


def _resolve_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    pieces = path.split(".")
    parent = root
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    return parent, pieces[-1]


def inject_vit_lora(backbone: nn.Module, config: LoRAConfig) -> list[str]:
    """Inject LoRA modules and return their fully qualified backbone paths."""
    if not hasattr(backbone, "blocks"):
        raise TypeError("the selected backbone has no transformer blocks")
    depth = len(backbone.blocks)
    config.validate(depth)
    indices = range(depth) if config.blocks == "all" else config.blocks
    injected: list[str] = []
    for index in indices:
        block = backbone.blocks[index]
        for target in config.targets:
            parent, attribute = _resolve_parent(block, target)
            current = getattr(parent, attribute)
            path = f"blocks.{index}.{target}"
            if isinstance(current, LoRALinear):
                if (
                    current.lora_A.out_features != config.rank
                    or current.scaling != float(config.alpha) / config.rank
                ):
                    raise ValueError(f"existing LoRA configuration differs at {path}")
            else:
                setattr(parent, attribute, LoRALinear(
                    current, config.rank, config.alpha, config.dropout,
                ))
            injected.append(path)
    setattr(backbone, "_wilor_lora_config", config)
    return injected


def iter_lora_modules(module: nn.Module) -> Iterator[LoRALinear]:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for child in iter_lora_modules(module)
        for parameter in (child.lora_A.weight, child.lora_B.weight)
    ]


def lora_parameter_names(module: nn.Module) -> set[str]:
    return {
        name for name, _ in module.named_parameters()
        if name.endswith(("lora_A.weight", "lora_B.weight"))
    }


def set_lora_enabled(module: nn.Module, enabled: bool) -> None:
    for child in iter_lora_modules(module):
        child.enabled = bool(enabled)


def set_lora_trainable(module: nn.Module, trainable: bool) -> None:
    for child in iter_lora_modules(module):
        child.lora_A.weight.requires_grad = bool(trainable)
        child.lora_B.weight.requires_grad = bool(trainable)


@contextmanager
def lora_disabled(module: nn.Module) -> Iterable[None]:
    states = [(child, child.enabled) for child in iter_lora_modules(module)]
    try:
        for child, _ in states:
            child.enabled = False
        yield
    finally:
        for child, enabled in states:
            child.enabled = enabled
