"""
Backbone encoders for PAC-Net.
Primary: DINOv2-S (frozen, 384-dim).
Fallback: ViT-B/16 or ResNet-50 from torchvision.
"""

import torch
import torch.nn as nn


BACKBONE_DIMS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'vit_b_16':      768,
    'resnet50':      2048,
}


class DINOv2Backbone(nn.Module):
    def __init__(self, model_name: str = 'dinov2_vits14', frozen: bool = True):
        super().__init__()
        self.model = torch.hub.load(
            'facebookresearch/dinov2', model_name, pretrained=True
        )
        self.out_dim = BACKBONE_DIMS[model_name]
        self.frozen = frozen
        if frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                return self.model(x)          # [B, 384]
        return self.model(x)


class TorchvisionBackbone(nn.Module):
    """Drop-in replacement using torchvision models (no hub download needed)."""

    def __init__(self, arch: str = 'vit_b_16', frozen: bool = True):
        super().__init__()
        import torchvision.models as tvm
        if arch == 'vit_b_16':
            base = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            self.out_dim = 768
            # Strip classification head – keep encoder only
            base.heads = nn.Identity()
            self.model = base
        elif arch == 'resnet50':
            base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
            self.out_dim = 2048
            base.fc = nn.Identity()
            self.model = base
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        self.frozen = frozen
        if frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)


def build_backbone(name: str = 'dinov2_vits14', frozen: bool = True) -> nn.Module:
    """Factory — tries DINOv2 first, falls back gracefully."""
    if name.startswith('dinov2'):
        try:
            return DINOv2Backbone(name, frozen)
        except Exception as e:
            print(f"[backbone] DINOv2 load failed ({e}), falling back to vit_b_16")
            return TorchvisionBackbone('vit_b_16', frozen)
    if name == 'resnet18_scratch':
        return ScratchResNet18(frozen)
    return TorchvisionBackbone(name, frozen)


class ScratchResNet18(nn.Module):
    """ResNet-18 with random init — no download, CPU-friendly demo backbone."""
    def __init__(self, frozen: bool = False):
        super().__init__()
        import torchvision.models as tvm
        base = tvm.resnet18(weights=None)
        base.fc = nn.Identity()
        self.model = base
        self.out_dim = 512
        self.frozen = frozen
        if frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)
