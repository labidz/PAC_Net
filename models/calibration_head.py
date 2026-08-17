"""
Per-Class Calibration Head for PAC-Net.

Core idea: prototype density in feature space is a principled signal for
calibration quality. High density (sample near known class exemplars) →
low temperature (model is confident, trust it). Low density (far from
all prototypes) → high temperature (model is uncertain, flatten it).

Architecture:
    density [B, C] → per-sample MLP → temperature [B, C]
    calibrated_logits = raw_logits / temperature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationHead(nn.Module):
    """
    Lightweight MLP that maps per-class prototype density to calibration temperature.

    Input:  density [B, C]  (cosine similarity in [-1, 1])
    Output: temperature [B, C]  (positive, 1.0 = no change)
    """

    def __init__(self, num_classes: int, hidden_dim: int = 64,
                 init_temperature: float = 1.5):
        super().__init__()
        self.C = num_classes

        # Shared MLP applied independently to each class's density scalar
        # Input: 1 (density scalar) → Output: 1 (log temperature)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialise output bias so that T ≈ init_temperature at start
        import math
        with torch.no_grad():
            self.mlp[-1].bias.fill_(math.log(init_temperature))

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            density: [B, C]
        Returns:
            temperature: [B, C]  — positive via Softplus
        """
        B, C = density.shape
        d    = density.view(B * C, 1)           # [B*C, 1]
        log_t = self.mlp(d).view(B, C)          # [B, C]
        T    = F.softplus(log_t) + 1e-4         # ensure T > 0
        return T

    def calibrate(self, logits: torch.Tensor,
                  density: torch.Tensor) -> torch.Tensor:
        """
        Apply input-adaptive, per-class calibration.

        Args:
            logits:  [B, C]  (raw prototype similarity logits)
            density: [B, C]  (prototype density scores)
        Returns:
            calibrated_logits: [B, C]
        """
        T = self.forward(density)               # [B, C]
        return logits / T


class GlobalTemperatureCalibration(nn.Module):
    """
    Baseline: single global scalar temperature (post-hoc Temperature Scaling).
    Used in ablation A8.
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        import math
        self.log_T = nn.Parameter(torch.tensor(math.log(init_temperature)))

    def forward(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        return logits / self.log_T.exp().clamp(0.01, 10.0)
