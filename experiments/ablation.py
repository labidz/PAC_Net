"""
PAC-Net Ablation Runner — E7

Systematically trains 8 model variants and compares against full PAC-Net.

Ablation variants:
    A1 — No prototype bank (linear head)
    A2 — No logit adjustment
    A3 — No reliability loss (λ₂ = 0)
    A4 — No prototype diversity loss (λ₃ = 0)
    A5 — Unfrozen backbone (full fine-tuning)
    A6 — K=1 prototype per class
    A7 — K=10 prototypes per class
    A8 — Global temperature scalar instead of adaptive calibration head

Usage:
    python experiments/ablation.py \
        --dataset ham10000 --root /data/ham10000 \
        --seed 42 --epochs 30
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models   import build_pac_net, PACNetConfig, PACNet
from losses   import PACNetLoss, LogitAdjustmentCE
from data     import build_dataset, build_dataloader, get_train_transform, get_eval_transform
from metrics  import classification_report_full, calibration_report
from utils    import set_seed, save_checkpoint, make_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser('PAC-Net Ablation Study')
    p.add_argument('--dataset',    type=str, default='ham10000')
    p.add_argument('--root',       type=str, required=True)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--epochs',     type=int, default=30,
                   help='Fewer epochs for ablation speed')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--wd',         type=float, default=1e-4)
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--output_dir', type=str, default='outputs/ablation')
    p.add_argument('--variants',   type=str, nargs='+',
                   default=['A1','A2','A3','A4','A5','A6','A7','A8','FULL'],
                   help='Which variants to run')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Variant factory
# ---------------------------------------------------------------------------

class LinearHeadNet(nn.Module):
    """A1: DINOv2 + ProjectionMLP + Linear classifier (no prototypes)."""

    def __init__(self, num_classes, backbone_name='dinov2_vits14',
                 proj_dim=256):
        super().__init__()
        from models.backbone import build_backbone
        self.backbone  = build_backbone(backbone_name, frozen=True)
        from models.pac_net import ProjectionMLP
        self.projector = ProjectionMLP(self.backbone.out_dim, proj_dim)
        self.head      = nn.Linear(proj_dim, num_classes)

    def forward(self, x, return_all=False):
        feat    = self.backbone(x)
        proj    = self.projector(feat)
        logits  = self.head(proj)
        density = torch.zeros(x.shape[0], logits.shape[1], device=x.device)
        if return_all:
            return logits, logits, density, proj
        return logits, logits, density

    @property
    def proto_bank(self):
        return _NullProtoBank()


class _NullProtoBank:
    def diversity_loss(self):
        return torch.tensor(0.0)


def build_variant(name: str, num_classes: int,
                  class_counts: List[int]) -> tuple:
    """
    Returns (model, criterion) for each ablation variant.
    All share the same base hyperparameters as full PAC-Net.
    """
    base_lambda = dict(lambda_focal=0.5, lambda_reliability=0.3,
                       lambda_diversity=0.1, la_tau=1.0)

    if name == 'A1':
        # No prototype bank — linear head
        model = LinearHeadNet(num_classes)
        crit  = PACNetLoss(class_counts, **base_lambda)
        return model, crit

    if name == 'A2':
        # No logit adjustment (τ = 0)
        model = build_pac_net(num_classes)
        crit  = PACNetLoss(class_counts, la_tau=0.0,
                           lambda_focal=base_lambda['lambda_focal'],
                           lambda_reliability=base_lambda['lambda_reliability'],
                           lambda_diversity=base_lambda['lambda_diversity'])
        return model, crit

    if name == 'A3':
        # No reliability loss (λ₂ = 0)
        model = build_pac_net(num_classes)
        crit  = PACNetLoss(class_counts, lambda_reliability=0.0,
                           lambda_focal=base_lambda['lambda_focal'],
                           lambda_diversity=base_lambda['lambda_diversity'],
                           la_tau=base_lambda['la_tau'])
        return model, crit

    if name == 'A4':
        # No diversity loss (λ₃ = 0)
        model = build_pac_net(num_classes)
        crit  = PACNetLoss(class_counts, lambda_diversity=0.0,
                           lambda_focal=base_lambda['lambda_focal'],
                           lambda_reliability=base_lambda['lambda_reliability'],
                           la_tau=base_lambda['la_tau'])
        return model, crit

    if name == 'A5':
        # Full backbone fine-tuning (frozen_backbone=False)
        model = build_pac_net(num_classes, frozen_backbone=False)
        crit  = PACNetLoss(class_counts, **base_lambda)
        return model, crit

    if name == 'A6':
        # K=1 prototype per class
        model = build_pac_net(num_classes, num_prototypes=1)
        crit  = PACNetLoss(class_counts, **base_lambda)
        return model, crit

    if name == 'A7':
        # K=10 prototypes per class
        model = build_pac_net(num_classes, num_prototypes=10)
        crit  = PACNetLoss(class_counts, **base_lambda)
        return model, crit

    if name == 'A8':
        # Global temperature scalar (post-hoc style)
        model = build_pac_net(num_classes, use_global_temp=True)
        crit  = PACNetLoss(class_counts, **base_lambda)
        return model, crit

    if name == 'FULL':
        # Full PAC-Net
        model = build_pac_net(num_classes)
        crit  = PACNetLoss(class_counts, **base_lambda)
        return model, crit

    raise ValueError(f"Unknown variant: {name}")


# ---------------------------------------------------------------------------
# Training loop (shared)
# ---------------------------------------------------------------------------

def train_variant(
    name:         str,
    model:        nn.Module,
    criterion,
    train_loader,
    val_loader,
    class_names:  List[str],
    device:       torch.device,
    epochs:       int,
    lr:           float,
    wd:           float,
    out_dir:      Path,
) -> Dict:
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_bacc  = 0.0
    best_ece   = float('inf')
    best_metrics = {}

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            cal_logits, raw_logits, density = model(imgs)
            loss, _ = criterion(cal_logits, raw_logits, labels,
                                model.proto_bank)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate every 5 epochs (speed)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            metrics = _eval(model, val_loader, device, class_names)
            combined = metrics['ECE'] - metrics['BalancedAccuracy']
            if combined < (best_ece - best_bacc):
                best_bacc    = metrics['BalancedAccuracy']
                best_ece     = metrics['ECE']
                best_metrics = metrics

    return best_metrics


@torch.no_grad()
def _eval(model, loader, device, class_names) -> Dict:
    model.eval()
    probs_l, labels_l = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        cal_logits, _, _ = model(imgs)
        probs_l.append(F.softmax(cal_logits, dim=-1).cpu().numpy())
        labels_l.append(labels.numpy())
    import numpy as np
    probs  = np.concatenate(probs_l)
    labels = np.concatenate(labels_l)
    clf    = classification_report_full(probs, labels, class_names)
    cal    = calibration_report(probs, labels)
    return {**clf, **cal}


# ---------------------------------------------------------------------------
# Result table printer
# ---------------------------------------------------------------------------

def print_ablation_table(results: Dict[str, Dict]):
    metrics = ['BalancedAccuracy', 'MacroF1', 'ECE', 'Brier', 'NLL']
    header  = f"{'Variant':<10}" + ''.join(f'{m:>18}' for m in metrics)
    print('\n' + '='*len(header))
    print('  ABLATION RESULTS')
    print('='*len(header))
    print(header)
    print('-'*len(header))
    for name in ['FULL','A1','A2','A3','A4','A5','A6','A7','A8']:
        if name not in results:
            continue
        r    = results[name]
        row  = f"{name:<10}"
        for m in metrics:
            v   = r.get(m, float('nan'))
            row += f"{v:>18.4f}"
        if name == 'FULL':
            row += '  ← proposed'
        print(row)
    print('='*len(header))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_ds = build_dataset(args.dataset, args.root, 'train',
                             get_train_transform(), seed=args.seed)
    val_ds   = build_dataset(args.dataset, args.root, 'val',
                             get_eval_transform(), seed=args.seed)
    train_loader = build_dataloader(train_ds, args.batch_size,
                                    use_sampler=True,
                                    num_workers=args.num_workers)
    val_loader   = build_dataloader(val_ds, args.batch_size * 2,
                                    shuffle=False,
                                    num_workers=args.num_workers)

    all_results = {}
    for name in args.variants:
        print(f'\n{"="*55}')
        print(f'  Ablation variant: {name}')
        print(f'{"="*55}')
        t0 = time.time()
        try:
            model, criterion = build_variant(
                name, train_ds.num_classes, train_ds.class_counts
            )
            metrics = train_variant(
                name, model, criterion, train_loader, val_loader,
                train_ds.class_names, device,
                epochs=args.epochs, lr=args.lr, wd=args.wd,
                out_dir=out_dir,
            )
            all_results[name] = metrics
            print(f'  Done in {time.time()-t0:.0f}s | '
                  f'BAcc={metrics.get("BalancedAccuracy",0):.4f} | '
                  f'ECE={metrics.get("ECE",0):.4f}')
        except Exception as e:
            print(f'  FAILED: {e}')
            all_results[name] = {'error': str(e)}

    print_ablation_table(all_results)

    out_path = out_dir / f'ablation_seed{args.seed}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n[✓] Ablation results saved → {out_path}')


if __name__ == '__main__':
    main()
