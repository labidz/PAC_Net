"""
PAC-Net Combined Loss

L_total = L_LA_CE            (logit-adjusted cross-entropy — handles imbalance)
        + λ₁ * L_focal       (focal loss on tail classes only — extra push)
        + λ₂ * L_reliability (differentiable soft-ECE — calibration signal)
        + λ₃ * L_diversity   (prototype diversity — prevents collapse)

All components are differentiable and operate on calibrated logits
except diversity, which is computed directly from the prototype bank.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# 1. Logit-Adjustment Cross-Entropy  (Menon et al. 2021)
# ---------------------------------------------------------------------------

class LogitAdjustmentCE(nn.Module):
    """
    Adds τ·log(π_k) to logits before CE, where π_k = n_k / N.
    This makes the Bayes-optimal classifier on imbalanced data
    equivalent to the standard CE optimum.

    At inference use raw logits (no adjustment) for balanced evaluation,
    or add τ·log(π_k) post-hoc for accuracy-optimised balanced decision.
    """

    def __init__(self, class_counts: List[int], tau: float = 1.0):
        super().__init__()
        self.tau = tau
        counts = torch.tensor(class_counts, dtype=torch.float)
        priors = counts / counts.sum()
        self.register_buffer('log_priors', torch.log(priors + 1e-8))   # [C]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        adjusted = logits + self.tau * self.log_priors          # broadcast over B
        return F.cross_entropy(adjusted, targets)

    def adjust_at_inference(self, logits: torch.Tensor) -> torch.Tensor:
        """Post-hoc logit adjustment for balanced-accuracy-maximising inference."""
        return logits + self.tau * self.log_priors


# ---------------------------------------------------------------------------
# 2. Focal Loss (tail classes only)
# ---------------------------------------------------------------------------

class TailFocalLoss(nn.Module):
    """
    Focal loss applied only to samples whose ground-truth label is a tail class.
    Tail classes = classes with fewer than `tail_threshold` training samples.
    Leaves head-class samples with standard CE (handled by LA-CE above).
    """

    def __init__(self, class_counts: List[int],
                 tail_threshold: Optional[int] = None,
                 gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        counts = torch.tensor(class_counts, dtype=torch.long)
        if tail_threshold is None:
            # Default: classes below median count are "tail"
            tail_threshold = int(counts.float().median().item())
        tail_mask = counts < tail_threshold
        self.register_buffer('tail_classes', tail_mask)   # [C] bool

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Find samples whose true class is a tail class
        is_tail = self.tail_classes[targets]              # [B] bool
        if is_tail.sum() == 0:
            return logits.new_tensor(0.0)

        logits_tail  = logits[is_tail]
        targets_tail = targets[is_tail]

        p      = F.softmax(logits_tail, dim=-1)
        p_t    = p.gather(1, targets_tail.unsqueeze(1)).squeeze(1)   # [B_tail]
        weight = (1.0 - p_t) ** self.gamma
        loss   = F.cross_entropy(logits_tail, targets_tail, reduction='none')
        return (weight * loss).mean()


# ---------------------------------------------------------------------------
# 3. Differentiable Soft-ECE Reliability Loss
# ---------------------------------------------------------------------------

class ReliabilityLoss(nn.Module):
    """
    Differentiable approximation of ECE using soft bin assignment via
    a Gaussian kernel (bandwidth = 1 / (2 * num_bins)).

    Gradient flows through predicted probabilities and soft weights.
    Bin boundaries are in [0, 1]; each sample is softly assigned to
    all bins based on its confidence score.
    """

    def __init__(self, num_bins: int = 15):
        super().__init__()
        self.M = num_bins
        # Bin centres: 1/(2M), 3/(2M), ..., (2M-1)/(2M)
        centers = torch.linspace(1 / (2 * num_bins),
                                 1 - 1 / (2 * num_bins),
                                 num_bins)
        self.register_buffer('centers', centers)
        self.bw = 1.0 / (2.0 * num_bins)   # Gaussian bandwidth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs       = F.softmax(logits, dim=-1)                 # [B, C]
        conf, preds = probs.max(dim=-1)                         # [B]
        correct     = preds.eq(targets).float()                 # [B]

        # Soft assignment: [B, M]
        conf_exp = conf.unsqueeze(1)                            # [B, 1]
        centers  = self.centers.unsqueeze(0)                    # [1, M]
        weights  = torch.exp(
            -0.5 * ((conf_exp - centers) / self.bw) ** 2
        )                                                        # [B, M]
        weights  = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted mean confidence and accuracy per bin
        bin_conf = (weights * conf_exp).sum(dim=0)              # [M]
        bin_acc  = (weights * correct.unsqueeze(1)).sum(dim=0)  # [M]
        bin_wts  = weights.sum(dim=0)                           # [M] – bin sizes

        # Weighted |conf - acc| (weight by bin occupancy)
        bin_wts  = bin_wts / (bin_wts.sum() + 1e-8)
        ece_soft = (bin_wts * (bin_conf - bin_acc).abs()).sum()
        return ece_soft


# ---------------------------------------------------------------------------
# 4. Combined Loss
# ---------------------------------------------------------------------------

class PACNetLoss(nn.Module):
    """
    Combines all four loss terms with configurable λ weights.

    Usage:
        criterion = PACNetLoss(class_counts=[3000, 300, ...], lambda_focal=0.5, ...)
        loss, breakdown = criterion(cal_logits, raw_logits, targets, proto_bank)
    """

    def __init__(
        self,
        class_counts:      List[int],
        lambda_focal:      float = 0.5,
        lambda_reliability: float = 0.3,
        lambda_diversity:  float = 0.1,
        la_tau:            float = 1.0,
        focal_gamma:       float = 2.0,
        num_bins:          int   = 15,
        tail_threshold:    Optional[int] = None,
    ):
        super().__init__()
        self.λ1 = lambda_focal
        self.λ2 = lambda_reliability
        self.λ3 = lambda_diversity

        self.la_ce       = LogitAdjustmentCE(class_counts, tau=la_tau)
        self.focal       = TailFocalLoss(class_counts, tail_threshold, focal_gamma)
        self.reliability = ReliabilityLoss(num_bins)

    def forward(
        self,
        cal_logits:  torch.Tensor,    # [B, C] — calibrated
        raw_logits:  torch.Tensor,    # [B, C] — before calibration head
        targets:     torch.Tensor,    # [B]
        proto_bank,                   # PrototypeBank instance (for diversity)
    ):
        l_la   = self.la_ce(raw_logits, targets)
        l_foc  = self.focal(raw_logits, targets)
        l_rel  = self.reliability(cal_logits, targets)
        l_div  = proto_bank.diversity_loss()

        total  = l_la + self.λ1 * l_foc + self.λ2 * l_rel + self.λ3 * l_div

        breakdown = {
            'loss/total':       total.item(),
            'loss/la_ce':       l_la.item(),
            'loss/focal':       l_foc.item(),
            'loss/reliability': l_rel.item(),
            'loss/diversity':   l_div.item(),
        }
        return total, breakdown
