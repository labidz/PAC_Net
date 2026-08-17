"""
PAC-Net Training Script

Usage:
    python experiments/train.py --dataset ham10000 --root /data/ham10000
    python experiments/train.py --dataset isic2019 --root /data/isic2019 --seed 43
    python experiments/train.py --dataset oct --root /data/oct --seeds 42 43 44 45 46

5-seed protocol runs automatically when --seeds is passed.
Results are saved to outputs/{dataset}/seed_{seed}/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models       import build_pac_net, PACNetConfig
from losses       import PACNetLoss
from data         import build_dataset, build_dataloader, get_train_transform, get_eval_transform
from metrics      import classification_report_full, calibration_report, ece
from utils        import set_seed, SEEDS, save_checkpoint, load_checkpoint, make_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser('PAC-Net Training')
    p.add_argument('--dataset',    type=str, default='ham10000',
                   choices=['ham10000','isic2019','oct','aptos'])
    p.add_argument('--root',       type=str, required=True,
                   help='Path to dataset root directory')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--seeds',      type=int, nargs='+', default=None,
                   help='Run multiple seeds (overrides --seed)')
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--wd',         type=float, default=1e-4)
    p.add_argument('--patience',   type=int, default=10,
                   help='Early stopping patience (epochs)')
    p.add_argument('--backbone',   type=str, default='dinov2_vits14')
    p.add_argument('--proj_dim',   type=int, default=256)
    p.add_argument('--num_proto',  type=int, default=5)
    p.add_argument('--lambda_focal', type=float, default=0.5)
    p.add_argument('--lambda_rel',   type=float, default=0.3)
    p.add_argument('--lambda_div',   type=float, default=0.1)
    p.add_argument('--la_tau',     type=float, default=1.0)
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--output_dir', type=str, default='outputs')
    p.add_argument('--resume',     type=str, default=None)
    p.add_argument('--no_sampler', action='store_true',
                   help='Disable weighted sampler (use shuffle instead)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single seed training run
# ---------------------------------------------------------------------------

def train_one_seed(args, seed: int) -> dict:
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir) / args.dataset / f'seed_{seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  Dataset: {args.dataset} | Seed: {seed} | Device: {device}')
    print(f'{"="*60}')

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_ds = build_dataset(args.dataset, args.root, 'train',
                             get_train_transform(), seed=seed)
    val_ds   = build_dataset(args.dataset, args.root, 'val',
                             get_eval_transform(), seed=seed)

    train_loader = build_dataloader(
        train_ds, batch_size=args.batch_size,
        shuffle=args.no_sampler, use_sampler=not args.no_sampler,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers,
    )

    num_classes  = train_ds.num_classes
    class_counts = train_ds.class_counts
    print(f'  Classes : {num_classes} | Counts: {class_counts}')
    print(f'  Imbalance ratio: {max(class_counts)/min(class_counts):.1f}:1')

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    cfg = PACNetConfig(
        num_classes     = num_classes,
        backbone        = args.backbone,
        frozen_backbone = True,
        proj_dim        = args.proj_dim,
        num_prototypes  = args.num_proto,
    )
    model = build_pac_net(num_classes, backbone=args.backbone,
                          proj_dim=args.proj_dim, num_prototypes=args.num_proto).to(device)
    print(f'  Trainable params: {model.trainable_params:,}'
          f'  (Total: {model.total_params:,})')

    # -----------------------------------------------------------------------
    # Loss, Optimiser, Scheduler
    # -----------------------------------------------------------------------
    criterion = PACNetLoss(
        class_counts     = class_counts,
        lambda_focal     = args.lambda_focal,
        lambda_reliability = args.lambda_rel,
        lambda_diversity = args.lambda_div,
        la_tau           = args.la_tau,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch    = 0
    best_val_ece   = float('inf')
    best_val_bacc  = 0.0
    patience_count = 0

    if args.resume:
        start_epoch, best_val_bacc = load_checkpoint(args.resume, model,
                                                     optimizer, device)

    # Tracking for best model selection: use ECE + BalAcc jointly
    best_combined = float('inf')

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = {k: 0.0 for k in ['total','la_ce','focal','reliability','diversity']}
        n_batches  = 0

        for imgs, labels in tqdm(train_loader, desc=f'Ep {epoch+1:02d} train',
                                 leave=False):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            cal_logits, raw_logits, density = model(imgs)
            loss, breakdown = criterion(cal_logits, raw_logits, labels,
                                        model.proto_bank)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_loss:
                epoch_loss[k] += breakdown.get(f'loss/{k}', 0.0)
            n_batches += 1

        scheduler.step()

        for k in epoch_loss:
            epoch_loss[k] /= n_batches

        # -----------------------------------------------------------------------
        # Validation
        # -----------------------------------------------------------------------
        val_metrics = evaluate_loader(model, val_loader, device,
                                      train_ds.class_names)

        combined = val_metrics['ECE'] - val_metrics['BalancedAccuracy']
        is_best  = combined < best_combined
        if is_best:
            best_combined = combined
            patience_count = 0
            save_checkpoint(
                make_checkpoint(model, optimizer, epoch, val_metrics, vars(args)),
                str(out_dir / 'checkpoint.pt'), is_best=True
            )
        else:
            patience_count += 1

        ep_log = {
            'epoch':         epoch + 1,
            'lr':            optimizer.param_groups[0]['lr'],
            'time_s':        round(time.time() - t0, 1),
            **{f'train_{k}': round(v, 4) for k, v in epoch_loss.items()},
            **{f'val_{k}':   round(v, 4) for k, v in val_metrics.items()},
        }
        history.append(ep_log)

        print(f'  Ep {epoch+1:02d} | '
              f'Loss {epoch_loss["total"]:.4f} | '
              f'valBAcc {val_metrics["BalancedAccuracy"]:.4f} | '
              f'valECE {val_metrics["ECE"]:.4f} | '
              f'patience {patience_count}/{args.patience}'
              + (' ✓ best' if is_best else ''))

        # Save history every epoch
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if patience_count >= args.patience:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    print(f'\n  Best combined score: {best_combined:.4f}')
    return {'seed': seed, 'best_combined': best_combined,
            'out_dir': str(out_dir)}


# ---------------------------------------------------------------------------
# Inference loop (returns probs, labels, density arrays)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_probs, all_labels, all_density = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        cal_logits, _, density = model(imgs)
        probs = torch.softmax(cal_logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
        all_density.append(density.cpu().numpy())
    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            np.concatenate(all_density))


def evaluate_loader(model, loader, device, class_names):
    probs, labels, _ = predict_loader(model, loader, device)
    clf  = classification_report_full(probs, labels, class_names)
    cal  = calibration_report(probs, labels)
    return {**clf, **cal}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]

    all_results = []
    for seed in seeds:
        res = train_one_seed(args, seed)
        all_results.append(res)

    if len(seeds) > 1:
        print('\n=== Multi-seed summary ===')
        for r in all_results:
            print(f"  Seed {r['seed']}  combined={r['best_combined']:.4f}  → {r['out_dir']}")


if __name__ == '__main__':
    main()
