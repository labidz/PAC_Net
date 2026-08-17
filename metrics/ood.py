"""
OOD Detection Metrics for PAC-Net.

Anomaly scores:
    density   — max prototype cosine similarity (PAC-Net primary)
    msp       — Maximum Softmax Probability (MSP, Hendrycks 2017 baseline)
    energy    — Free-energy score (Liu 2020 baseline)

Metrics:
    AUROC     — Area under ROC curve (in-dist=1, OOD=0)
    AUPR_in   — AUPR treating in-distribution as positive
    AUPR_out  — AUPR treating OOD as positive
    FPR@95TPR — False positive rate when TPR=0.95

Usage:
    scores_in  = density scores for in-distribution test set  [n_in]
    scores_out = density scores for OOD dataset               [n_out]
    report = ood_report(scores_in, scores_out)
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc
)
from typing import Dict


def fpr_at_tpr(in_scores: np.ndarray, out_scores: np.ndarray,
               tpr_level: float = 0.95) -> float:
    """
    Compute FPR when TPR = tpr_level.
    Higher score = more likely in-distribution.
    """
    # Threshold: smallest score such that ≥ tpr_level fraction of
    # in-distribution scores exceed it
    threshold = np.percentile(in_scores, 100.0 * (1 - tpr_level))
    fpr = (out_scores >= threshold).mean()
    return float(fpr)


def auroc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """AUROC: in-distribution labeled 1, OOD labeled 0."""
    labels = np.concatenate([np.ones(len(in_scores)),
                             np.zeros(len(out_scores))])
    scores = np.concatenate([in_scores, out_scores])
    return float(roc_auc_score(labels, scores))


def aupr_in(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """AUPR with in-distribution as positive."""
    labels = np.concatenate([np.ones(len(in_scores)),
                             np.zeros(len(out_scores))])
    scores = np.concatenate([in_scores, out_scores])
    p, r, _ = precision_recall_curve(labels, scores)
    return float(auc(r, p))


def aupr_out(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """AUPR with OOD as positive (negative detection)."""
    labels = np.concatenate([np.zeros(len(in_scores)),
                             np.ones(len(out_scores))])
    scores = np.concatenate([-in_scores, -out_scores])
    p, r, _ = precision_recall_curve(labels, scores)
    return float(auc(r, p))


def ood_report(
    in_scores:  np.ndarray,   # anomaly scores for in-distribution samples
    out_scores: np.ndarray,   # anomaly scores for OOD samples
    tpr_level:  float = 0.95,
) -> Dict[str, float]:
    return {
        'AUROC':     auroc(in_scores, out_scores),
        'AUPR_in':   aupr_in(in_scores, out_scores),
        'AUPR_out':  aupr_out(in_scores, out_scores),
        'FPR@95TPR': fpr_at_tpr(in_scores, out_scores, tpr_level),
    }


# ---------------------------------------------------------------------------
# Anomaly score extractors
# ---------------------------------------------------------------------------

def msp_score(probs: np.ndarray) -> np.ndarray:
    """Maximum Softmax Probability (Hendrycks & Gimpel, 2017)."""
    return probs.max(axis=-1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Free-energy score (Liu et al., 2020): E = T * log Σ exp(f/T)."""
    scaled = logits / temperature
    # numerically stable log-sum-exp
    max_l  = scaled.max(axis=-1, keepdims=True)
    lse    = np.log(np.exp(scaled - max_l).sum(axis=-1)) + max_l.squeeze()
    return temperature * lse   # higher = more in-distribution


def density_score(density: np.ndarray) -> np.ndarray:
    """
    PAC-Net prototype density:
    max cosine similarity over all prototypes.
    Higher = closer to training distribution = more in-distribution.
    density: [B, C] → take max over classes.
    """
    return density.max(axis=-1)
