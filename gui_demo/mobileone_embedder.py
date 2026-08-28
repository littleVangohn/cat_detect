"""Minimal MobileOne-S1 embedding model used by the PC GUI."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class StudentModel(nn.Module):
    """MobileOne-S1 backbone with the trained 384D projection head."""

    def __init__(self, name: str = "mobileone_s1", pretrained: bool = False):
        super().__init__()
        if name != "mobileone_s1":
            raise ValueError(f"GUI only supports mobileone_s1, got {name!r}")
        self.name = name
        self.backbone = timm.create_model(
            "mobileone_s1.apple_in1k", pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.num_features, 384, bias=False),
            nn.BatchNorm1d(384),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return F.normalize(self.projection(features), dim=1)
