"""
PAC-Net: Prototype-Anchored Calibration Network

Forward pass:
    image → DINOv2-S (frozen) → 384-dim
           → Projection MLP   → 256-dim (L2-normalised)
           → Prototype Bank   → density [B, C], raw logits [B, C]
           → Calibration Head → calibrated logits [B, C]

Trainable parameters: Projection MLP (~600K) + Prototype Bank (~15K) +
                       Calibration Head (~8K) + log_scale (1) ≈ 2.1M total
DINOv2-S backbone: 22M params, all frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

from .backbone       import build_backbone
from .prototype_bank import PrototypeBank
from .calibration_head import CalibrationHead, GlobalTemperatureCalibration


# ---------------------------------------------------------------------------
# Config dataclass (replaces yaml at model level)
# ---------------------------------------------------------------------------

@dataclass
class PACNetConfig:
    num_classes:       int   = 7
    backbone:          str   = 'dinov2_vits14'   # or 'vit_b_16' / 'resnet50'
    frozen_backbone:   bool  = True
    proj_dim:          int   = 256
    proj_hidden:       int   = 512
    num_prototypes:    int   = 5
    cal_hidden:        int   = 64
    cal_init_temp:     float = 1.5
    use_global_temp:   bool  = False             # ablation A8


# ---------------------------------------------------------------------------
# Projection MLP
# ---------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)   # L2-normalised


# ---------------------------------------------------------------------------
# PAC-Net
# ---------------------------------------------------------------------------

class PACNet(nn.Module):
    def __init__(self, cfg: PACNetConfig):
        super().__init__()
        self.cfg = cfg

        # 1. Backbone (frozen DINOv2-S → 384-dim)
        self.backbone  = build_backbone(cfg.backbone, cfg.frozen_backbone)
        backbone_dim   = self.backbone.out_dim

        # 2. Projection MLP → normalised 256-dim prototype space
        self.projector = ProjectionMLP(backbone_dim, cfg.proj_dim, cfg.proj_hidden)

        # 3. Prototype bank: C classes × K prototypes × 256-dim
        self.proto_bank = PrototypeBank(cfg.num_classes, cfg.proj_dim,
                                        cfg.num_prototypes)

        # 4. Calibration head (adaptive) or global temperature (ablation)
        if cfg.use_global_temp:
            self.cal_head = GlobalTemperatureCalibration(cfg.cal_init_temp)
        else:
            self.cal_head = CalibrationHead(cfg.num_classes, cfg.cal_hidden,
                                            cfg.cal_init_temp)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ):
        """
        Args:
            x:          [B, 3, H, W]
            return_all: if True, also return features, density, raw logits

        Returns (default):
            cal_logits: [B, C]   — use for loss + evaluation

        Returns (return_all=True):
            cal_logits, raw_logits, density, features
        """
        # Step 1: frozen backbone
        raw_feat = self.backbone(x)                       # [B, backbone_dim]

        # Step 2: project to prototype space
        feat = self.projector(raw_feat)                   # [B, proj_dim], L2-norm

        # Step 3: prototype bank → density + logits
        density    = self.proto_bank.compute_density(feat)   # [B, C]
        raw_logits = self.proto_bank.compute_logits(feat)    # [B, C]

        # Step 4: calibration
        if self.cfg.use_global_temp:
            cal_logits = self.cal_head(raw_logits)
        else:
            cal_logits = self.cal_head.calibrate(raw_logits, density)  # [B, C]

        if return_all:
            return cal_logits, raw_logits, density, feat
        return cal_logits, raw_logits, density

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns calibrated softmax probabilities [B, C]."""
        cal_logits, _, _ = self.forward(x)
        return F.softmax(cal_logits, dim=-1)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pac_net(num_classes: int, **kwargs) -> PACNet:
    cfg = PACNetConfig(num_classes=num_classes, **kwargs)
    return PACNet(cfg)
