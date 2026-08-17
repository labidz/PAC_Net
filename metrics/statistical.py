"""
Classification metrics and statistical significance testing.

Classification:
    accuracy, balanced_accuracy, macro_f1, macro_auc, per_class_f1
    shot_breakdown  — many/medium/few-shot per-class accuracy

Statistical significance:
    wilcoxon_holm   — pairwise Wilcoxon signed-rank with Holm-Bonferroni
                      correction over multiple seeds
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def classification_report_full(
    probs:       np.ndarray,   # [n, C] softmax probabilities
    labels:      np.ndarray,   # [n]    ground truth
    class_names: List[str],
) -> Dict[str, float]:
    preds = probs.argmax(axis=-1)
    C     = probs.shape[1]

    metrics = {
        'Accuracy':          float(accuracy_score(labels, preds)),
        'BalancedAccuracy':  float(balanced_accuracy_score(labels, preds)),
        'MacroF1':           float(f1_score(labels, preds,
                                            average='macro', zero_division=0)),
        'WeightedF1':        float(f1_score(labels, preds,
                                            average='weighted', zero_division=0)),
    }

    # Macro AUC (one-vs-rest, only if all classes present)
    if len(np.unique(labels)) == C:
        try:
            metrics['MacroAUC'] = float(roc_auc_score(
                labels, probs, multi_class='ovr', average='macro'
            ))
        except ValueError:
            metrics['MacroAUC'] = float('nan')

    # Per-class F1
    f1_per = f1_score(labels, preds, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        metrics[f'F1_{name}'] = float(f1_per[i]) if i < len(f1_per) else float('nan')

    return metrics


# ---------------------------------------------------------------------------
# Many / Medium / Few-shot breakdown
# ---------------------------------------------------------------------------

def shot_breakdown(
    preds:        np.ndarray,
    labels:       np.ndarray,
    class_counts: List[int],   # training counts per class
    many_thresh:  int = 100,
    few_thresh:   int = 20,
) -> Dict[str, float]:
    """
    Split classes into:
        many-shot  : n_train >= many_thresh
        medium-shot: few_thresh <= n_train < many_thresh
        few-shot   : n_train < few_thresh

    Reports per-shot mean per-class accuracy (macro within each shot group).
    """
    counts = np.array(class_counts)
    many   = np.where(counts >= many_thresh)[0]
    few    = np.where(counts < few_thresh)[0]
    medium = np.where((counts >= few_thresh) & (counts < many_thresh))[0]

    def _group_acc(class_indices):
        if len(class_indices) == 0:
            return float('nan')
        accs = []
        for c in class_indices:
            mask = labels == c
            if mask.sum() == 0:
                continue
            accs.append((preds[mask] == c).mean())
        return float(np.mean(accs)) if accs else float('nan')

    return {
        'acc_many':   _group_acc(many),
        'acc_medium': _group_acc(medium),
        'acc_few':    _group_acc(few),
        'n_many':     int(len(many)),
        'n_medium':   int(len(medium)),
        'n_few':      int(len(few)),
    }


# ---------------------------------------------------------------------------
# Statistical significance: Wilcoxon + Holm-Bonferroni
# ---------------------------------------------------------------------------

def wilcoxon_holm(
    our_scores:       List[float],          # metric values over seeds
    baseline_scores:  Dict[str, List[float]], # {baseline_name: [seed_scores]}
    alpha:            float = 0.05,
    alternative:      str   = 'greater',    # we test our > baseline
) -> Dict[str, Dict]:
    """
    For each baseline, run Wilcoxon signed-rank test against our method.
    Apply Holm-Bonferroni correction across all comparisons.

    Returns:
        {
            'ResNet50': {'statistic': ..., 'p_raw': ..., 'p_adj': ...,
                         'significant': bool, 'effect_size_r': float},
            ...
        }
    """
    results   = {}
    raw_pvals = []
    names     = []

    for name, scores in baseline_scores.items():
        try:
            stat, p = wilcoxon(our_scores, scores, alternative=alternative,
                               zero_method='pratt')
        except ValueError:
            stat, p = float('nan'), 1.0
        results[name] = {'statistic': float(stat), 'p_raw': float(p)}
        raw_pvals.append(p)
        names.append(name)

    # Holm-Bonferroni correction
    m      = len(raw_pvals)
    order  = np.argsort(raw_pvals)
    p_adj  = np.ones(m)
    for rank, idx in enumerate(order):
        p_adj[idx] = min(1.0, raw_pvals[idx] * (m - rank))

    for i, name in enumerate(names):
        p   = p_adj[i]
        results[name]['p_adj']      = float(p)
        results[name]['significant'] = bool(p < alpha)
        # Effect size: r = Z / sqrt(N)
        n = len(our_scores)
        from scipy.stats import norm
        try:
            z = norm.ppf(1 - raw_pvals[i] / 2)
        except Exception:
            z = float('nan')
        results[name]['effect_size_r'] = float(z / np.sqrt(n))

    return results


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(
    seed_results: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """
    Args:
        seed_results: list of metric dicts, one per seed
    Returns:
        {metric: (mean, std)}
    """
    all_keys = seed_results[0].keys()
    out = {}
    for key in all_keys:
        vals = [r[key] for r in seed_results if not np.isnan(r.get(key, float('nan')))]
        out[key] = (float(np.mean(vals)), float(np.std(vals)))
    return out
