"""
Calibration metrics for PAC-Net evaluation.

Implements:
    ECE   — Expected Calibration Error (uniform-mass binning)
    MCE   — Maximum Calibration Error
    ACE   — Adaptive Calibration Error (equal-mass bins)
    Brier — Brier Score (multi-class)
    NLL   — Negative Log-Likelihood
    reliability_data — bin-level data for reliability diagrams

All functions accept numpy arrays and are deterministic.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def confidence_accuracy(probs: np.ndarray, labels: np.ndarray):
    """Return (confidences, correctness) arrays."""
    conf  = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    acc   = (preds == labels).astype(float)
    return conf, acc


# ---------------------------------------------------------------------------
# ECE (uniform-width bins)
# ---------------------------------------------------------------------------

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE = Σ_m (|B_m| / n) · |acc(B_m) - conf(B_m)|
    """
    conf, acc = confidence_accuracy(probs, labels)
    n = len(conf)
    bins = np.linspace(0, 1, n_bins + 1)
    error = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        error += mask.sum() / n * abs(acc[mask].mean() - conf[mask].mean())
    return float(error)


# ---------------------------------------------------------------------------
# MCE
# ---------------------------------------------------------------------------

def mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Maximum calibration error over all bins."""
    conf, acc = confidence_accuracy(probs, labels)
    bins  = np.linspace(0, 1, n_bins + 1)
    errs  = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        errs.append(abs(acc[mask].mean() - conf[mask].mean()))
    return float(max(errs)) if errs else 0.0


# ---------------------------------------------------------------------------
# ACE (equal-mass bins)
# ---------------------------------------------------------------------------

def ace(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Adaptive Calibration Error: equal-frequency bins."""
    conf, acc = confidence_accuracy(probs, labels)
    n  = len(conf)
    order = np.argsort(conf)
    conf  = conf[order]
    acc   = acc[order]

    error = 0.0
    chunk = n // n_bins
    for i in range(n_bins):
        sl = slice(i * chunk, (i + 1) * chunk if i < n_bins - 1 else n)
        if (sl.stop - sl.start) == 0:
            continue
        error += abs(acc[sl].mean() - conf[sl].mean()) / n_bins
    return float(error)


# ---------------------------------------------------------------------------
# Brier Score (multi-class)
# ---------------------------------------------------------------------------

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier = (1/n) Σ_i Σ_k (p_ik - y_ik)²
    Divided by C so it is scale-comparable across datasets.
    """
    C    = probs.shape[1]
    oneh = np.eye(C)[labels]             # [n, C]
    return float(((probs - oneh) ** 2).sum(axis=-1).mean())


# ---------------------------------------------------------------------------
# Negative Log-Likelihood
# ---------------------------------------------------------------------------

def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    p_true = probs[np.arange(len(labels)), labels].clip(eps, 1.0)
    return float(-np.log(p_true).mean())


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------

def reliability_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with bin_conf, bin_acc, bin_count, bin_edges.
    Use for plotting reliability diagrams.
    """
    conf, acc = confidence_accuracy(probs, labels)
    n    = len(conf)
    bins = np.linspace(0, 1, n_bins + 1)
    b_conf  = np.zeros(n_bins)
    b_acc   = np.zeros(n_bins)
    b_count = np.zeros(n_bins, dtype=int)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        b_conf[i]  = conf[mask].mean()
        b_acc[i]   = acc[mask].mean()
        b_count[i] = mask.sum()

    return {
        'bin_conf':  b_conf,
        'bin_acc':   b_acc,
        'bin_count': b_count,
        'bin_edges': bins,
    }


# ---------------------------------------------------------------------------
# Aggregate calibration report
# ---------------------------------------------------------------------------

def calibration_report(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    return {
        'ECE':   ece(probs, labels, n_bins),
        'MCE':   mce(probs, labels, n_bins),
        'ACE':   ace(probs, labels, n_bins),
        'Brier': brier_score(probs, labels),
        'NLL':   nll(probs, labels),
    }
