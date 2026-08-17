"""
Prototype Memory Bank for PAC-Net.

Stores K learnable L2-normalised prototypes per class.
Classification logits = max cosine similarity over K prototypes * scale.
Density score per class drives calibration head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int, num_prototypes: int = 5):
        super().__init__()
        self.C = num_classes
        self.D = feat_dim
        self.K = num_prototypes

        # Shape: [C, K, D]  — learnable
        self.prototypes = nn.Parameter(torch.empty(num_classes, num_prototypes, feat_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # Learnable log-temperature for similarity scaling
        self.log_scale = nn.Parameter(torch.tensor(2.3026))   # exp(2.3) ≈ 10

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def normalized(self) -> torch.Tensor:
        """Returns L2-normalized prototypes [C, K, D]."""
        return F.normalize(self.prototypes, dim=-1)

    def compute_density(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute per-class density score (max cosine sim over K prototypes).

        Args:
            features: [B, D] L2-normalised projected features

        Returns:
            density: [B, C]  — values in [-1, 1]
        """
        protos = self.normalized()                # [C, K, D]
        # Reshape for batch-wise cosine similarity
        # features: [B, D] → [B, 1, 1, D]
        f = features[:, None, None, :]            # [B, 1, 1, D]
        # protos: [C, K, D] → [1, C, K, D]
        p = protos[None, :, :, :]                 # [1, C, K, D]
        # dot product along D: [B, C, K]
        sim = (f * p).sum(dim=-1)
        # Max over K prototypes → [B, C]
        density, _ = sim.max(dim=-1)
        return density

    def compute_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns similarity logits [B, C].
        scale = exp(log_scale), clamped to [1, 100] for stability.
        """
        scale = self.log_scale.exp().clamp(1.0, 100.0)
        density = self.compute_density(features)
        return scale * density                    # [B, C]

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def diversity_loss(self) -> torch.Tensor:
        """
        Encourage prototypes within the same class to be diverse.
        Penalises mean off-diagonal cosine similarity across all classes.
        """
        protos = self.normalized()                # [C, K, D]
        total = 0.0
        for c in range(self.C):
            p = protos[c]                         # [K, D]
            gram = p @ p.T                        # [K, K]
            eye  = torch.eye(self.K, device=p.device, dtype=p.dtype)
            off_diag = gram * (1.0 - eye)
            total = total + off_diag.sum() / (self.K * (self.K - 1) + 1e-8)
        return total / self.C

    # ------------------------------------------------------------------
    # Nearest-prototype retrieval (evaluation / visualisation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def nearest_prototypes(self, features: torch.Tensor, top_k: int = 1):
        """
        For each sample, return (class_idx, prototype_idx, similarity).
        features: [B, D] L2-normalised
        """
        protos  = self.normalized().view(self.C * self.K, self.D)   # [C*K, D]
        sims    = features @ protos.T                                 # [B, C*K]
        topk    = sims.topk(top_k, dim=-1)
        indices = topk.indices                                        # [B, top_k]
        class_idx = indices // self.K
        proto_idx = indices %  self.K
        return class_idx, proto_idx, topk.values
