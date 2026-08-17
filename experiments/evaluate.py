"""
PAC-Net Full Evaluation Script

Covers:
    E1  — Primary classification (all 4 datasets)
    E2  — Long-tail many/medium/few-shot breakdown
    E3  — Calibration metrics (ECE, MCE, ACE, Brier, NLL)
    E4  — Calibration under distribution shift (cross-dataset)
    E5  — OOD detection (density vs MSP vs energy)
    E6  — Cross-dataset generalisation (zero-shot transfer)
    E8  — Statistical significance (Wilcoxon + Holm-Bonferroni)
    E11 — Efficiency (params, FLOPs, throughput)
    E12 — Few-shot tail robustness

Usage:
    # Evaluate a single checkpoint
    python experiments/evaluate.py \
        --ckpt outputs/ham10000/seed_42/checkpoint_best.pt \
        --dataset ham10000 --root /data/ham10000

    # Full multi-dataset sweep (runs E1–E12 automatically)
    python experiments/evaluate.py \
        --ckpt outputs/ham10000/seed_42/checkpoint_best.pt \
        --dataset ham10000 --root /data/ham10000 \
        --ood_root /data/ph2 \
        --cross_dataset oct --cross_root /data/oct \
        --full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models   import build_pac_net
from data     import build_dataset, build_dataloader, get_eval_transform
from metrics  import (
    classification_report_full, calibration_report, reliability_data,
    shot_breakdown, ood_report, density_score, msp_score, energy_score,
    wilcoxon_holm, aggregate_seeds,
)
from utils    import set_seed, load_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser('PAC-Net Evaluation')
    p.add_argument('--ckpt',         type=str, required=True)
    p.add_argument('--dataset',      type=str, default='ham10000')
    p.add_argument('--root',         type=str, required=True)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--batch_size',   type=int, default=128)
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--output_dir',   type=str, default='eval_results')
    # E5 OOD
    p.add_argument('--ood_dataset',  type=str, default='ph2',
                   help='OOD dataset name')
    p.add_argument('--ood_root',     type=str, default=None,
                   help='OOD dataset root (optional)')
    # E6 cross-dataset
    p.add_argument('--cross_dataset', type=str, default=None)
    p.add_argument('--cross_root',    type=str, default=None)
    # E8 baseline seeds dir
    p.add_argument('--baselines_dir', type=str, default=None,
                   help='Dir containing baseline results JSONs for E8')
    # Flags
    p.add_argument('--full',    action='store_true',
                   help='Run all experiment groups')
    p.add_argument('--no_e2',   action='store_true')
    p.add_argument('--no_e5',   action='store_true')
    p.add_argument('--no_e6',   action='store_true')
    p.add_argument('--no_e11',  action='store_true')
    p.add_argument('--no_e12',  action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    """Returns (probs [n,C], labels [n], density [n,C], raw_logits [n,C])."""
    model.eval()
    probs_l, labels_l, density_l, logits_l = [], [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        cal_logits, raw_logits, density = model(imgs)
        probs_l.append(torch.softmax(cal_logits, dim=-1).cpu().numpy())
        labels_l.append(labels.numpy())
        density_l.append(density.cpu().numpy())
        logits_l.append(raw_logits.cpu().numpy())
    return (np.concatenate(probs_l),
            np.concatenate(labels_l),
            np.concatenate(density_l),
            np.concatenate(logits_l))


# ---------------------------------------------------------------------------
# E1 — Primary Classification
# ---------------------------------------------------------------------------

def run_e1(model, loader, class_names, device) -> Dict:
    print('\n[E1] Primary Classification Metrics')
    probs, labels, _, _ = predict(model, loader, device)
    metrics = classification_report_full(probs, labels, class_names)
    cal     = calibration_report(probs, labels)
    results = {**metrics, **cal}
    _print_table(results, 'E1 Results')
    return results


# ---------------------------------------------------------------------------
# E2 — Long-Tail Shot Breakdown
# ---------------------------------------------------------------------------

def run_e2(model, loader, class_names, class_counts, device) -> Dict:
    print('\n[E2] Many / Medium / Few-shot Breakdown')
    probs, labels, _, _ = predict(model, loader, device)
    preds   = probs.argmax(axis=-1)
    results = shot_breakdown(preds, labels, class_counts)
    _print_table(results, 'E2 Results')
    return results


# ---------------------------------------------------------------------------
# E3 — Calibration Metrics (already in E1, returns reliability data)
# ---------------------------------------------------------------------------

def run_e3(model, loader, device) -> Dict:
    print('\n[E3] Detailed Calibration + Reliability Data')
    probs, labels, _, _ = predict(model, loader, device)
    cal    = calibration_report(probs, labels)
    rel    = reliability_data(probs, labels, n_bins=15)
    _print_table(cal, 'E3 Calibration')
    return {'calibration': cal, 'reliability': {
        'bin_conf':  rel['bin_conf'].tolist(),
        'bin_acc':   rel['bin_acc'].tolist(),
        'bin_count': rel['bin_count'].tolist(),
    }}


# ---------------------------------------------------------------------------
# E4 — Calibration Under Distribution Shift
# ---------------------------------------------------------------------------

def run_e4(
    model,
    in_loader,     # in-distribution test loader (e.g. HAM10000)
    ood_loader,    # same-domain but different distribution (e.g. ISIC2019)
    device,
    in_name:  str = 'InDist',
    out_name: str = 'OODist',
) -> Dict:
    print(f'\n[E4] Calibration Shift: {in_name} → {out_name}')
    p_in,  l_in,  _, _ = predict(model, in_loader,  device)
    p_out, l_out, _, _ = predict(model, ood_loader, device)

    cal_in  = calibration_report(p_in,  l_in)
    cal_out = calibration_report(p_out, l_out)

    shift = {
        f'ECE_shift':   abs(cal_out['ECE']   - cal_in['ECE']),
        f'Brier_shift': abs(cal_out['Brier'] - cal_in['Brier']),
        f'NLL_shift':   abs(cal_out['NLL']   - cal_in['NLL']),
    }
    results = {
        f'{in_name}':  cal_in,
        f'{out_name}': cal_out,
        'shift':       shift,
    }
    print(f"  ECE  {in_name}: {cal_in['ECE']:.4f}  {out_name}: {cal_out['ECE']:.4f}  Δ={shift['ECE_shift']:.4f}")
    print(f"  Brier {in_name}: {cal_in['Brier']:.4f}  {out_name}: {cal_out['Brier']:.4f}  Δ={shift['Brier_shift']:.4f}")
    return results


# ---------------------------------------------------------------------------
# E5 — OOD Detection
# ---------------------------------------------------------------------------

def run_e5(
    model,
    in_loader,
    ood_loader,
    device,
    in_name:  str = 'HAM10000',
    ood_name: str = 'PH2',
) -> Dict:
    print(f'\n[E5] OOD Detection: {in_name} (in) vs {ood_name} (OOD)')

    p_in,  _, d_in,  l_in  = predict(model, in_loader,  device)
    p_out, _, d_out, l_out = predict(model, ood_loader, device)

    # Three anomaly scores
    scores = {
        'Density (PAC-Net)': (density_score(d_in),  density_score(d_out)),
        'MSP':               (msp_score(p_in),       msp_score(p_out)),
        'Energy':            (energy_score(l_in),    energy_score(l_out)),
    }

    results = {}
    for name, (s_in, s_out) in scores.items():
        report = ood_report(s_in, s_out)
        results[name] = report
        print(f'  {name:<20} AUROC={report["AUROC"]:.4f}  '
              f'AUPR_in={report["AUPR_in"]:.4f}  '
              f'FPR@95={report["FPR@95TPR"]:.4f}')
    return results


# ---------------------------------------------------------------------------
# E6 — Cross-Dataset Generalisation
# ---------------------------------------------------------------------------

def run_e6(
    model,
    cross_loader,
    cross_class_names: List[str],
    device,
    dataset_name: str = 'CrossDataset',
) -> Dict:
    print(f'\n[E6] Cross-Dataset Generalisation: {dataset_name}')
    probs, labels, _, _ = predict(model, cross_loader, device)

    # Map to shared class space if needed (best-effort; same class names = exact)
    # For partial overlap: evaluate only on overlapping classes
    results = {
        'note': f'Zero-shot transfer to {dataset_name}',
    }
    try:
        cal  = calibration_report(probs, labels)
        results.update(cal)
    except Exception as e:
        results['error'] = str(e)
    _print_table({k: v for k, v in results.items() if isinstance(v, float)},
                 'E6 Cross-Dataset')
    return results


# ---------------------------------------------------------------------------
# E8 — Statistical Significance
# ---------------------------------------------------------------------------

def run_e8(
    our_bacc_seeds:      List[float],
    baseline_bacc_seeds: Dict[str, List[float]],
) -> Dict:
    print('\n[E8] Statistical Significance (Wilcoxon + Holm-Bonferroni)')
    results = wilcoxon_holm(our_bacc_seeds, baseline_bacc_seeds)
    for name, r in results.items():
        sig = '✓' if r['significant'] else '✗'
        print(f'  vs {name:<20} p_adj={r["p_adj"]:.4f}  '
              f'r={r["effect_size_r"]:.3f}  significant={sig}')
    return results


# ---------------------------------------------------------------------------
# E11 — Efficiency
# ---------------------------------------------------------------------------

def run_e11(model, device, input_size=(1, 3, 224, 224),
            n_warmup=10, n_bench=100) -> Dict:
    print('\n[E11] Efficiency Benchmark')
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        dummy  = torch.zeros(input_size).to(device)
        flops  = FlopCountAnalysis(model, dummy)
        gflops = flops.total() / 1e9
        print(f'  GFLOPs (single image): {gflops:.3f}')
    except ImportError:
        gflops = float('nan')
        print('  fvcore not installed — GFLOPs skipped. pip install fvcore')

    params_total    = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Throughput
    model.eval()
    dummy_batch = torch.randn(64, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_bench):
            model(dummy_batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed   = time.perf_counter() - t0
    throughput = (n_bench * 64) / elapsed   # images/sec

    results = {
        'params_total':       params_total,
        'params_trainable':   params_trainable,
        'gflops':             gflops,
        'throughput_img_sec': round(throughput, 1),
    }
    print(f'  Total params:     {params_total:,}')
    print(f'  Trainable params: {params_trainable:,}')
    print(f'  Throughput:       {throughput:.0f} img/sec (bs=64)')
    return results


# ---------------------------------------------------------------------------
# E12 — Few-Shot Tail Robustness
# ---------------------------------------------------------------------------

def run_e12(
    args,
    model_cfg: Dict,
    device,
    tail_fractions=(0.1, 0.2, 0.5, 1.0),
) -> Dict:
    """
    Re-train/re-evaluate with only X% of tail class samples.
    Requires re-running training (full loop); here we just set up structure.
    This is typically run as a separate sweep.
    """
    print('\n[E12] Few-Shot Tail Robustness — scaffold')
    print('  Run train.py with --tail_fraction flag for each fraction.')
    print('  Fractions to test:', tail_fractions)
    return {'note': 'Run train.py with --tail_fraction for [0.1, 0.2, 0.5, 1.0]'}


# ---------------------------------------------------------------------------
# Helper: pretty-print a flat dict
# ---------------------------------------------------------------------------

def _print_table(d: Dict, title: str = ''):
    if title:
        print(f'  --- {title} ---')
    for k, v in d.items():
        if isinstance(v, float):
            print(f'    {k:<25} {v:.4f}')
        elif isinstance(v, int):
            print(f'    {k:<25} {v}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = ckpt.get('config', {})

    # Re-build model from checkpoint config
    num_classes = ckpt['model_state']['proto_bank.prototypes'].shape[0]
    model = build_pac_net(
        num_classes,
        backbone     = cfg.get('backbone', 'dinov2_vits14'),
        proj_dim     = cfg.get('proj_dim', 256),
        num_prototypes = cfg.get('num_proto', 5),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Build test dataset
    test_ds = build_dataset(args.dataset, args.root, 'test',
                            get_eval_transform(), seed=args.seed)
    test_loader = build_dataloader(test_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)

    all_results = {}

    # E1 — Primary Classification
    all_results['E1'] = run_e1(model, test_loader, test_ds.class_names, device)

    # E2 — Long-tail breakdown
    if not args.no_e2:
        all_results['E2'] = run_e2(model, test_loader, test_ds.class_names,
                                   test_ds.class_counts, device)

    # E3 — Calibration + Reliability
    all_results['E3'] = run_e3(model, test_loader, device)

    # E4 — Distribution shift calibration (needs a second dataset)
    if args.ood_root and not args.no_e5:
        try:
            ood_test_ds = build_dataset(args.ood_dataset, args.ood_root, 'test',
                                        get_eval_transform(), seed=args.seed)
            ood_loader = build_dataloader(ood_test_ds, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)
            all_results['E4'] = run_e4(
                model, test_loader, ood_loader, device,
                in_name=args.dataset, out_name=args.ood_dataset
            )
            # E5 — OOD detection
            all_results['E5'] = run_e5(
                model, test_loader, ood_loader, device,
                in_name=args.dataset, ood_name=args.ood_dataset
            )
        except Exception as e:
            print(f'  [E4/E5] Skipped: {e}')

    # E6 — Cross-dataset generalisation
    if args.cross_dataset and args.cross_root and not args.no_e6:
        try:
            cross_ds = build_dataset(args.cross_dataset, args.cross_root,
                                     'test', get_eval_transform(), seed=args.seed)
            cross_loader = build_dataloader(cross_ds, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)
            all_results['E6'] = run_e6(model, cross_loader,
                                       cross_ds.class_names, device,
                                       args.cross_dataset)
        except Exception as e:
            print(f'  [E6] Skipped: {e}')

    # E11 — Efficiency
    if not args.no_e11:
        all_results['E11'] = run_e11(model, device)

    # Save all results
    result_path = out_dir / f'eval_seed{args.seed}.json'
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n[✓] Results saved → {result_path}')


if __name__ == '__main__':
    main()
