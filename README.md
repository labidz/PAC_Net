# PAC-Net: Prototype-Anchored Calibration Network
### Reliable Long-Tail Medical Image Classification

---

## Core Idea

Medical image classifiers trained on imbalanced data face two simultaneous
failures: **poor tail-class accuracy** and **poor calibration**. Existing
methods fix one while breaking the other — logit adjustment improves rare
class accuracy but inflates ECE; temperature scaling calibrates well only on
balanced distributions.

**PAC-Net** proposes a single principled solution: *prototype distance in
feature space is a learnable signal for both class prediction and calibration
quality*. A sample far from all class prototypes should produce high-entropy,
well-calibrated outputs — not confident wrong ones.

---

## Architecture

```
Image (224×224)
    │
    ▼
DINOv2-S (frozen, 22M params)  ──► 384-dim CLS features
    │
    ▼
Projection MLP  ──────────────────► 256-dim L2-normalised features
    │
    ▼
Prototype Bank (C × K × 256)  ─┬──► density [B,C]  (max cosine sim per class)
    │                           └──► raw logits [B,C] = scale × density
    │
    ▼
Calibration Head  ────────────────► per-class temperature T(density)
    │
    ▼
Calibrated logits = raw_logits / T(density)
```

**Trainable parameters:** ~2.1M (Projection MLP + Prototype Bank + Calibration Head)
**DINOv2-S backbone:** 22M params, frozen throughout.

---

## Datasets

| Dataset        | Domain       | Classes | Images  | IR    | Role              |
|----------------|------------- |---------|---------|-------|-------------------|
| HAM10000       | Dermatology  | 7       | 10,015  | 58:1  | Primary benchmark |
| ISIC 2019      | Dermatology  | 8       | 25,331  | 74:1  | Scale validation  |
| Retinal OCT    | Ophthalmology| 4       | 84,495  | 3:1   | Domain shift      |
| APTOS 2019     | Ophthalmology| 5       | 3,662   | 8:1   | Small dataset     |
| PH2            | Dermatology  | 3       | 200     | —     | OOD eval only     |

All datasets are public. Download links below.

---

## Setup

```bash
# Clone / copy this directory, then:
pip install -r requirements.txt

# DINOv2 will be downloaded automatically on first run via torch.hub
# If hub download is blocked, set backbone: vit_b_16 in the config
```

---

## Dataset Download

```
HAM10000  : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
ISIC 2019 : https://challenge.isic-archive.com/data/#2019
OCT       : https://www.kaggle.com/datasets/paultimothymooney/kermany2018
APTOS 2019: https://www.kaggle.com/competitions/aptos2019-blindness-detection
PH2       : https://www.fc.up.pt/addi/ph2%20database.html
```

---

## Training

### Single seed
```bash
python experiments/train.py \
    --dataset ham10000 \
    --root /data/ham10000 \
    --seed 42

python experiments/train.py \
    --dataset isic2019 \
    --root /data/isic2019

python experiments/train.py \
    --dataset oct \
    --root /data/oct

python experiments/train.py \
    --dataset aptos \
    --root /data/aptos
```

### 5-seed protocol (for paper results)
```bash
python experiments/train.py \
    --dataset ham10000 --root /data/ham10000 \
    --seeds 42 43 44 45 46

# Results saved to:
#   outputs/ham10000/seed_42/checkpoint_best.pt
#   outputs/ham10000/seed_42/history.json
#   ...
```

---

## Evaluation (E1–E6, E8, E11)

```bash
python experiments/evaluate.py \
    --ckpt  outputs/ham10000/seed_42/checkpoint_best.pt \
    --dataset ham10000 --root /data/ham10000 \
    --ood_dataset ph2 --ood_root /data/ph2 \
    --cross_dataset oct --cross_root /data/oct \
    --full

# Results → eval_results/ham10000/eval_seed42.json
```

---

## Ablation Study (E7)

```bash
python experiments/ablation.py \
    --dataset ham10000 --root /data/ham10000 \
    --seed 42 --epochs 30 \
    --variants A1 A2 A3 A4 A5 A6 A7 A8 FULL
```

Ablation variants:

| Variant | Change                                      |
|---------|---------------------------------------------|
| A1      | No prototype bank → linear head             |
| A2      | No logit adjustment (τ = 0)                 |
| A3      | No reliability loss (λ₂ = 0)               |
| A4      | No prototype diversity loss (λ₃ = 0)       |
| A5      | Full backbone fine-tuning (not frozen)      |
| A6      | K = 1 prototype per class                   |
| A7      | K = 10 prototypes per class                 |
| A8      | Global temperature scalar (post-hoc style)  |

---

## Visualisation (E10)

```bash
python experiments/visualize.py \
    --ckpt outputs/ham10000/seed_42/checkpoint_best.pt \
    --dataset ham10000 --root /data/ham10000 \
    --output_dir figures/

# Produces (PDF, 300 DPI):
#   fig1_tsne.pdf           t-SNE of prototype space
#   fig2_reliability.pdf    Reliability diagrams (3-way comparison)
#   fig3_confusion.pdf      Normalised confusion matrix
#   fig4_uncertainty.pdf    Density distribution (head vs tail)
#   fig6_per_class_ece.pdf  Per-class calibration error
```

---

## Experiment Groups (Paper)

| Group | Description                          | Script                    |
|-------|--------------------------------------|---------------------------|
| E1    | Primary classification (all 4 ds)   | evaluate.py               |
| E2    | Many/Medium/Few-shot breakdown       | evaluate.py               |
| E3    | Calibration: ECE/MCE/ACE/Brier/NLL  | evaluate.py               |
| E4    | Calibration under dist. shift        | evaluate.py --ood_root    |
| E5    | OOD detection AUROC/AUPR/FPR@95     | evaluate.py --ood_root    |
| E6    | Cross-dataset generalisation         | evaluate.py --cross_root  |
| E7    | 8-variant ablation table             | ablation.py               |
| E8    | Wilcoxon + Holm-Bonferroni tests     | evaluate.py               |
| E9    | Sensitivity (λ, K, proj_dim)         | train.py (sweep manually) |
| E10   | t-SNE, reliability, confusion, etc.  | visualize.py              |
| E11   | Efficiency (params, FLOPs, speed)    | evaluate.py               |
| E12   | Few-shot tail robustness             | train.py --tail_fraction  |

---

## Key Hyperparameters

| Parameter          | Default | Range tested     |
|--------------------|---------|------------------|
| K (prototypes/cls) | 5       | {1, 3, 5, 10, 20}|
| proj_dim           | 256     | {128, 256, 512}  |
| λ₁ (focal)        | 0.5     | {0.1, 0.5, 1, 2} |
| λ₂ (reliability)  | 0.3     | {0.1, 0.3, 0.5}  |
| λ₃ (diversity)    | 0.1     | {0.05, 0.1, 0.2} |
| τ (logit adj.)     | 1.0     | {0.5, 1.0, 2.0}  |
| lr                 | 3e-4    | fixed            |
| epochs             | 50      | dataset-specific |

---

## Baselines to Compare Against

Reproduce these with their official repos, then use evaluate.py for metrics:

**Long-tail methods:**
- LDAM (2019): `https://github.com/kaidic/LDAM-DRS`
- Logit Adjustment (2021): `https://github.com/google-research/google-research/tree/master/logit_adjustment`
- PaCo (2021): `https://github.com/jiequancui/Parametric-Contrastive-Learning`
- BCL (2022): `https://github.com/FlamieZhu/Balanced-Contrastive-Learning`

**Calibration methods (apply post-hoc):**
- Temperature Scaling: implemented in `visualize.py::temperature_scale`
- CPCS: `https://github.com/alinlab/CPCS`

---

## Repository Structure

```
pac_net/
├── models/
│   ├── backbone.py          DINOv2-S encoder (frozen), ViT-B/16 fallback
│   ├── prototype_bank.py    K-prototype memory bank per class
│   ├── calibration_head.py  Density-conditioned temperature MLP
│   └── pac_net.py           Full model assembly + factory
├── losses/
│   └── combined_loss.py     LA-CE + Focal + Soft-ECE + Diversity
├── data/
│   ├── datasets.py          HAM10000, ISIC19, OCT, APTOS, PH2 loaders
│   └── transforms.py        Train / eval augmentation pipelines
├── metrics/
│   ├── calibration.py       ECE, MCE, ACE, Brier, NLL, reliability_data
│   ├── ood.py               AUROC, AUPR, FPR@95TPR, density/MSP/energy scores
│   └── statistical.py       Wilcoxon+Holm-Bonferroni, shot breakdown, aggregation
├── experiments/
│   ├── train.py             5-seed training with early stopping
│   ├── evaluate.py          E1–E6, E8, E11
│   ├── ablation.py          E7 — 8 variant ablation
│   └── visualize.py         E10 — all figures
├── configs/
│   ├── ham10000.yaml
│   ├── isic2019.yaml
│   ├── oct.yaml
│   └── aptos.yaml
├── utils/
│   ├── seed.py              Deterministic seeding (5 seeds)
│   └── checkpoint.py        Save/load checkpoints
└── requirements.txt
```

---

## Expected Results (preliminary targets)

| Dataset   | Bal.Acc ↑ | MacroF1 ↑ | ECE ↓ | Brier ↓ |
|-----------|-----------|-----------|-------|---------|
| HAM10000  | ≥0.78     | ≥0.74     | ≤0.04 | ≤0.12   |
| ISIC 2019 | ≥0.75     | ≥0.71     | ≤0.05 | ≤0.14   |
| OCT       | ≥0.97     | ≥0.97     | ≤0.02 | ≤0.04   |
| APTOS     | ≥0.82     | ≥0.79     | ≤0.06 | ≤0.15   |

These are design targets, not guaranteed. Actual results depend on seed variance.

---

## Citation

```bibtex
@article{pacnet2025,
  title   = {PAC-Net: Prototype-Anchored Calibration Network for Reliable
             Long-Tail Medical Image Classification},
  author  = {Labid, Md. Rohul Amin and ...},
  journal = {Computers in Biology and Medicine},
  year    = {2025},
}
```
