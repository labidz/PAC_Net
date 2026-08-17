"""
PAC-Net Visualisation Script — E10

Produces:
    Fig 1: t-SNE of prototype space (head vs tail classes coloured)
    Fig 2: Reliability diagrams — PAC-Net vs Temperature Scaling vs Uncal
    Fig 3: Confusion matrix (per dataset)
    Fig 4: Uncertainty map — high-density vs low-density samples
    Fig 5: Prototype nearest-neighbour retrieval (qualitative)
    Fig 6: Per-class ECE bar chart

Usage:
    python experiments/visualize.py \
        --ckpt outputs/ham10000/seed_42/checkpoint_best.pt \
        --dataset ham10000 --root /data/ham10000 \
        --output_dir figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models  import build_pac_net
from data    import build_dataset, build_dataloader, get_eval_transform
from metrics import calibration_report, reliability_data
from utils   import set_seed


# ── colour palette (colourblind-safe) ────────────────────────────────────────
PALETTE = [
    '#0077BB', '#33BBEE', '#009988', '#EE7733',
    '#CC3311', '#EE3377', '#BBBBBB', '#44BB99',
]

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('PAC-Net Visualisation')
    p.add_argument('--ckpt',        type=str, required=True)
    p.add_argument('--dataset',     type=str, default='ham10000')
    p.add_argument('--root',        type=str, required=True)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--batch_size',  type=int, default=64)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output_dir',  type=str, default='figures')
    p.add_argument('--n_tsne',      type=int, default=2000,
                   help='Max samples for t-SNE (speed)')
    p.add_argument('--dpi',         type=int, default=300)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Prediction + feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all(model, loader, device):
    model.eval()
    probs_l, labels_l, features_l, density_l, logits_l = [], [], [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        cal_logits, raw_logits, density, feat = model(imgs, return_all=True)
        probs_l.append(F.softmax(cal_logits, dim=-1).cpu().numpy())
        labels_l.append(labels.numpy())
        features_l.append(feat.cpu().numpy())
        density_l.append(density.cpu().numpy())
        logits_l.append(raw_logits.cpu().numpy())
    return (np.concatenate(probs_l),
            np.concatenate(labels_l),
            np.concatenate(features_l),
            np.concatenate(density_l),
            np.concatenate(logits_l))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: t-SNE prototype space
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne(features, labels, class_names, class_counts,
              prototypes_np, out_path, n_samples=2000, dpi=300):
    """
    Scatter: sample features coloured by class.
    Prototypes shown as stars.
    Head/tail classes distinguished by marker size.
    """
    # Subsample for speed
    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(features), size=min(n_samples, len(features)),
                       replace=False)
    feat_s  = features[idx]
    label_s = labels[idx]

    # Include prototypes in t-SNE fit
    C, K, D = prototypes_np.shape
    proto_flat = prototypes_np.reshape(C * K, D)
    proto_flat = proto_flat / (np.linalg.norm(proto_flat, axis=-1, keepdims=True) + 1e-8)
    all_pts = np.concatenate([feat_s, proto_flat], axis=0)

    tsne = TSNE(n_components=2, random_state=0, perplexity=30,
                n_iter=1000, verbose=0)
    emb  = tsne.fit_transform(all_pts)
    emb_s = emb[:len(feat_s)]
    emb_p = emb[len(feat_s):]

    counts = np.array(class_counts)
    median = np.median(counts)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for c, cname in enumerate(class_names):
        mask   = label_s == c
        is_tail = counts[c] < median
        colour = PALETTE[c % len(PALETTE)]
        ax.scatter(emb_s[mask, 0], emb_s[mask, 1],
                   c=colour, s=(15 if is_tail else 8),
                   alpha=0.55, linewidths=0,
                   label=f'{cname} ({"tail" if is_tail else "head"})')

    # Prototypes as stars
    for c in range(C):
        colour = PALETTE[c % len(PALETTE)]
        p_emb  = emb_p[c*K:(c+1)*K]
        ax.scatter(p_emb[:, 0], p_emb[:, 1],
                   marker='*', s=200, c=colour,
                   edgecolors='black', linewidths=0.5, zorder=5)

    ax.legend(fontsize=8, ncol=2, framealpha=0.7)
    ax.set_title('t-SNE of PAC-Net Prototype Space\n(★ = class prototypes)',
                 fontsize=13)
    ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  [t-SNE] saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Reliability diagrams
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(methods: Dict[str, tuple],
                             out_path, dpi=300, n_bins=15):
    """
    methods: {name: (probs, labels)}
    """
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                             sharey=True)
    if len(methods) == 1:
        axes = [axes]

    colours = ['#0077BB', '#EE7733', '#BBBBBB']
    for ax, (name, (probs, labels)), col in zip(axes, methods.items(), colours):
        rel  = reliability_data(probs, labels, n_bins=n_bins)
        bconf = rel['bin_conf']
        bacc  = rel['bin_acc']
        bcnt  = rel['bin_count']
        edges = rel['bin_edges']
        widths = np.diff(edges)

        # Bars
        bars = ax.bar(edges[:-1], bacc, width=widths, align='edge',
                      color=col, alpha=0.75, edgecolor='white', label='Accuracy')
        # Confidence overlay (unfilled)
        ax.bar(edges[:-1], bconf, width=widths, align='edge',
               color='none', edgecolor='red', linestyle='--', linewidth=1.2,
               label='Confidence')
        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, label='Perfect')

        from metrics.calibration import ece as _ece
        ece_val = _ece(probs, labels, n_bins)
        ax.set_title(f'{name}\nECE = {ece_val:.4f}', fontsize=11)
        ax.set_xlabel('Confidence')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    axes[0].set_ylabel('Accuracy')
    fig.suptitle('Reliability Diagrams', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  [Reliability] saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(probs, labels, class_names, out_path, dpi=300):
    preds = probs.argmax(axis=-1)
    cm    = confusion_matrix(labels, preds, normalize='true')
    C     = len(class_names)

    fig, ax = plt.subplots(figsize=(max(6, C), max(5, C - 1)))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    for i in range(C):
        for j in range(C):
            val = cm[i, j]
            colour = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=colour)

    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True',      fontsize=11)
    ax.set_title('Normalised Confusion Matrix', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  [Confusion] saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Uncertainty map — density score distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_distribution(density, labels, class_names,
                                  class_counts, out_path, dpi=300):
    """
    KDE plot of max-density (anomaly score) per class.
    Tail classes vs head classes side-by-side.
    """
    max_density = density.max(axis=-1)   # [n]
    counts = np.array(class_counts)
    median = np.median(counts)

    head_d = max_density[np.isin(labels, np.where(counts >= median)[0])]
    tail_d = max_density[np.isin(labels, np.where(counts <  median)[0])]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(max_density.min(), max_density.max(), 50)

    ax.hist(head_d, bins=bins, density=True, alpha=0.6,
            color='#0077BB', label='Head classes')
    ax.hist(tail_d, bins=bins, density=True, alpha=0.6,
            color='#EE7733', label='Tail classes')

    ax.axvline(head_d.mean(), color='#0077BB', linestyle='--', linewidth=1.5)
    ax.axvline(tail_d.mean(), color='#EE7733', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Max Prototype Density (anomaly score)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Uncertainty Distribution: Head vs Tail Classes', fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  [Uncertainty] saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Per-class ECE bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_ece(probs, labels, class_names, class_counts,
                       out_path, dpi=300):
    """ECE computed per class (one-vs-rest binary calibration)."""
    from metrics.calibration import ece as _ece
    C     = len(class_names)
    counts = np.array(class_counts)
    median = np.median(counts)

    per_class = []
    for c in range(C):
        bin_labels = (labels == c).astype(int)
        bin_probs  = np.stack([1 - probs[:, c], probs[:, c]], axis=-1)
        per_class.append(_ece(bin_probs, bin_labels))

    colours = [('#0077BB' if counts[c] >= median else '#EE7733') for c in range(C)]

    fig, ax = plt.subplots(figsize=(max(8, C * 1.2), 4))
    bars = ax.bar(class_names, per_class, color=colours, edgecolor='white')

    ax.set_xlabel('Class', fontsize=11)
    ax.set_ylabel('ECE', fontsize=11)
    ax.set_title('Per-Class Calibration Error (ECE)', fontsize=12)
    ax.tick_params(axis='x', rotation=30)

    head_patch = mpatches.Patch(color='#0077BB', label='Head class')
    tail_patch = mpatches.Patch(color='#EE7733', label='Tail class')
    ax.legend(handles=[head_patch, tail_patch], fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  [Per-class ECE] saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling baseline for reliability diagram comparison
# ─────────────────────────────────────────────────────────────────────────────

def temperature_scale(logits_val, labels_val, logits_test):
    """Post-hoc temperature scaling: optimise T on val logits."""
    import torch
    from torch import optim
    T_param = torch.nn.Parameter(torch.tensor(1.5))
    opt     = optim.LBFGS([T_param], lr=0.01, max_iter=50)
    lv = torch.tensor(logits_val, dtype=torch.float32)
    ll = torch.tensor(labels_val, dtype=torch.long)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lv / T_param, ll)
        loss.backward()
        return loss

    opt.step(closure)
    T_best = T_param.item()

    scaled_test = logits_test / T_best
    scaled_test = torch.tensor(scaled_test, dtype=torch.float32)
    return F.softmax(scaled_test, dim=-1).numpy(), T_best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = ckpt.get('config', {})
    num_classes = ckpt['model_state']['proto_bank.prototypes'].shape[0]
    model = build_pac_net(
        num_classes,
        backbone     = cfg.get('backbone', 'dinov2_vits14'),
        proj_dim     = cfg.get('proj_dim', 256),
        num_prototypes = cfg.get('num_proto', 5),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])

    # Prototypes (for t-SNE)
    protos_np = model.proto_bank.prototypes.detach().cpu().numpy()  # [C, K, D]

    # Test data
    test_ds  = build_dataset(args.dataset, args.root, 'test',
                             get_eval_transform(), seed=args.seed)
    val_ds   = build_dataset(args.dataset, args.root, 'val',
                             get_eval_transform(), seed=args.seed)
    test_loader = build_dataloader(test_ds, args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)
    val_loader  = build_dataloader(val_ds, args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)

    print('\nExtracting features...')
    probs, labels, feats, density, logits = extract_all(model, test_loader, device)
    val_probs, val_labels, _, _, val_logits = extract_all(model, val_loader, device)

    class_names  = test_ds.class_names
    class_counts = test_ds.class_counts

    print('\nGenerating figures...')

    # Fig 1: t-SNE
    plot_tsne(feats, labels, class_names, class_counts, protos_np,
              out_dir / 'fig1_tsne.pdf',
              n_samples=args.n_tsne, dpi=args.dpi)

    # Fig 2: Reliability diagrams
    # Build temperature-scaled and uncalibrated comparisons
    uncal_probs = F.softmax(torch.tensor(logits, dtype=torch.float32),
                            dim=-1).numpy()
    ts_probs, T = temperature_scale(val_logits, val_labels, logits)
    print(f'  [Reliability] Temperature Scaling T={T:.3f}')

    plot_reliability_diagram(
        {
            'Uncalibrated':          (uncal_probs, labels),
            'Temperature Scaling':   (ts_probs,    labels),
            'PAC-Net (ours)':        (probs,        labels),
        },
        out_dir / 'fig2_reliability.pdf', dpi=args.dpi,
    )

    # Fig 3: Confusion matrix
    plot_confusion_matrix(probs, labels, class_names,
                          out_dir / 'fig3_confusion.pdf', dpi=args.dpi)

    # Fig 4: Uncertainty distribution
    plot_uncertainty_distribution(density, labels, class_names, class_counts,
                                  out_dir / 'fig4_uncertainty.pdf', dpi=args.dpi)

    # Fig 6: Per-class ECE
    plot_per_class_ece(probs, labels, class_names, class_counts,
                       out_dir / 'fig6_per_class_ece.pdf', dpi=args.dpi)

    print(f'\n[✓] All figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
