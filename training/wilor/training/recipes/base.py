"""Minimal training recipe interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch

from ..config import TrainingConfig


class TrainingRecipe(ABC):
    name: str
    metric_name: str
    mpjpe_name = "mpjpe_mm"

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    @abstractmethod
    def configure_model(self, model: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_mode(self, model: torch.nn.Module, training: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        model: torch.nn.Module,
        batch: Mapping[str, Any],
        training: bool,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(
        self, model: torch.nn.Module
    ) -> torch.optim.Optimizer:
        raise NotImplementedError

    @abstractmethod
    def metric(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, Any],
    ) -> tuple[str, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
        return [parameter for parameter in model.parameters() if parameter.requires_grad]
