"""
Dataset loaders for PAC-Net experiments.

Supported:
    HAM10000     — 10,015 images, 7 classes  (IR ≈ 58:1)
    ISIC2019     — 25,331 images, 8 classes  (IR ≈ 74:1)
    RetinalOCT   — 84,495 images, 4 classes  (Kermany 2018, Kaggle)
    APTOS2019    — 3,662  images, 5 classes
    PH2          — 200 images, 3 classes (OOD eval only)

All loaders return (image, label) and expose:
    .class_names    List[str]
    .class_counts   List[int]  (training split only)
    .num_classes    int
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Generic ImageFolder-style base
# ---------------------------------------------------------------------------

class LabelledImageDataset(Dataset):
    """Base: list of (path, label) pairs."""

    def __init__(self, samples: List[Tuple[str, int]],
                 class_names: List[str],
                 transform: Optional[Callable] = None):
        self.samples     = samples
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.transform   = transform

        counts = [0] * self.num_classes
        for _, lbl in samples:
            counts[lbl] += 1
        self.class_counts = counts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _split_samples(samples, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Stratified split into train/val/test."""
    rng = np.random.default_rng(seed)
    from collections import defaultdict
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)

    train, val, test = [], [], []
    for cls_samples in by_class.values():
        arr = np.array(cls_samples, dtype=object)
        rng.shuffle(arr)
        n   = len(arr)
        nv  = max(1, int(n * val_ratio))
        nt  = max(1, int(n * test_ratio))
        test.extend(arr[:nt].tolist())
        val.extend(arr[nt:nt+nv].tolist())
        train.extend(arr[nt+nv:].tolist())
    return train, val, test


# ---------------------------------------------------------------------------
# HAM10000
# ---------------------------------------------------------------------------

HAM_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
HAM_LABEL   = {c: i for i, c in enumerate(HAM_CLASSES)}


def load_ham10000(
    root: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> LabelledImageDataset:
    """
    root/
        HAMDiagnosis.csv          (columns: image_id, dx)
        HAM10000_images_part1/    (*.jpg)
        HAM10000_images_part2/    (*.jpg)
    """
    root = Path(root)
    meta = pd.read_csv(root / 'HAMDiagnosis.csv')

    img_dirs = [
        root / 'HAM10000_images_part1',
        root / 'HAM10000_images_part2',
    ]

    id2path = {}
    for d in img_dirs:
        if d.exists():
            for f in d.glob('*.jpg'):
                id2path[f.stem] = str(f)

    samples = []
    for _, row in meta.iterrows():
        iid = row['image_id']
        if iid in id2path and row['dx'] in HAM_LABEL:
            samples.append((id2path[iid], HAM_LABEL[row['dx']]))

    train_s, val_s, test_s = _split_samples(samples, val_ratio, test_ratio, seed)
    split_map = {'train': train_s, 'val': val_s, 'test': test_s}
    return LabelledImageDataset(split_map[split], HAM_CLASSES, transform)


# ---------------------------------------------------------------------------
# ISIC 2019
# ---------------------------------------------------------------------------

ISIC19_CLASSES = ['AK','BCC','BKL','DF','MEL','NV','SCC','VASC']
ISIC19_LABEL   = {c: i for i, c in enumerate(ISIC19_CLASSES)}


def load_isic2019(
    root: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> LabelledImageDataset:
    """
    root/
        ISIC_2019_Training_GroundTruth.csv
        ISIC_2019_Training_Input/   (*.jpg)
    """
    root = Path(root)
    gt   = pd.read_csv(root / 'ISIC_2019_Training_GroundTruth.csv')
    img_dir = root / 'ISIC_2019_Training_Input'

    # Ground truth is one-hot; find the positive column
    label_cols = [c for c in gt.columns if c != 'image']
    gt['label'] = gt[label_cols].values.argmax(axis=1)
    # Remap to ISIC19_CLASSES order (columns may differ)
    col_to_idx = {col: ISIC19_LABEL.get(col, None) for col in label_cols}

    samples = []
    for _, row in gt.iterrows():
        path = img_dir / f"{row['image']}.jpg"
        if not path.exists():
            continue
        label_col = label_cols[int(row['label'])]
        lbl = ISIC19_LABEL.get(label_col)
        if lbl is not None:
            samples.append((str(path), lbl))

    train_s, val_s, test_s = _split_samples(samples, val_ratio, test_ratio, seed)
    split_map = {'train': train_s, 'val': val_s, 'test': test_s}
    return LabelledImageDataset(split_map[split], ISIC19_CLASSES, transform)


# ---------------------------------------------------------------------------
# Retinal OCT (Kermany 2018 — Kaggle folder structure)
# ---------------------------------------------------------------------------

OCT_CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']


def load_retinal_oct(
    root: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
) -> LabelledImageDataset:
    """
    root/
        train/{CNV,DME,DRUSEN,NORMAL}/*.jpeg
        val/{...}/*.jpeg
        test/{...}/*.jpeg
    """
    root     = Path(root)
    split_dir = root / split
    label_map = {c: i for i, c in enumerate(OCT_CLASSES)}

    samples = []
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        lbl = label_map.get(cls_dir.name.upper())
        if lbl is None:
            continue
        for img_path in cls_dir.glob('*.jpeg'):
            samples.append((str(img_path), lbl))
        for img_path in cls_dir.glob('*.jpg'):
            samples.append((str(img_path), lbl))

    return LabelledImageDataset(samples, OCT_CLASSES, transform)


# ---------------------------------------------------------------------------
# APTOS 2019
# ---------------------------------------------------------------------------

APTOS_CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']


def load_aptos(
    root: str,
    split: str = 'train',
    transform: Optional[Callable] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> LabelledImageDataset:
    """
    root/
        train.csv  (columns: id_code, diagnosis)
        train_images/*.png
    """
    root    = Path(root)
    meta    = pd.read_csv(root / 'train.csv')
    img_dir = root / 'train_images'

    samples = []
    for _, row in meta.iterrows():
        path = img_dir / f"{row['id_code']}.png"
        if path.exists():
            samples.append((str(path), int(row['diagnosis'])))

    train_s, val_s, test_s = _split_samples(samples, val_ratio, test_ratio, seed)
    split_map = {'train': train_s, 'val': val_s, 'test': test_s}
    return LabelledImageDataset(split_map[split], APTOS_CLASSES, transform)


# ---------------------------------------------------------------------------
# PH2 (OOD evaluation only — 3 classes)
# ---------------------------------------------------------------------------

PH2_CLASSES = ['common_nevi', 'atypical_nevi', 'melanoma']


def load_ph2(
    root: str,
    transform: Optional[Callable] = None,
) -> LabelledImageDataset:
    """
    root/
        PH2Dataset.txt  (metadata)
        PH2 Dataset images/{image_id}/{image_id}_Dermoscopic_Image/*.bmp
    Label mapping:  0 = common nevi, 1 = atypical nevi, 2 = melanoma
    """
    root    = Path(root)
    img_root = root / 'PH2 Dataset images'

    # PH2 metadata has fixed-width format; parse column 3 (lesion type)
    samples = []
    for img_dir in sorted(img_root.iterdir()):
        if not img_dir.is_dir():
            continue
        derm_dir = img_dir / f'{img_dir.name}_Dermoscopic_Image'
        if not derm_dir.exists():
            continue
        bmps = list(derm_dir.glob('*.bmp'))
        if not bmps:
            continue
        # No csv — label from directory name convention (IMD002–IMD168)
        # Common nevi: IMD002–IMD080 (79 images)
        # Atypical nevi: IMD101–IMD168 (80 images)
        # Melanoma: IMD197–IMD436 (41 images)
        num = int(img_dir.name[3:])
        if 2 <= num <= 80:
            lbl = 0
        elif 101 <= num <= 168:
            lbl = 1
        else:
            lbl = 2
        samples.append((str(bmps[0]), lbl))

    return LabelledImageDataset(samples, PH2_CLASSES, transform)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    'ham10000':   load_ham10000,
    'isic2019':   load_isic2019,
    'oct':        load_retinal_oct,
    'aptos':      load_aptos,
    'ph2':        load_ph2,
}


def build_dataset(name: str, root: str, split: str,
                  transform=None, seed: int = 42, **kwargs):
    fn = DATASET_REGISTRY[name]
    if name in ('ham10000', 'isic2019', 'aptos'):
        return fn(root, split, transform, seed=seed, **kwargs)
    elif name == 'oct':
        return fn(root, split, transform)
    elif name == 'ph2':
        return fn(root, transform)
    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Weighted sampler (class-balanced)
# ---------------------------------------------------------------------------

def make_weighted_sampler(dataset: LabelledImageDataset) -> WeightedRandomSampler:
    """Returns a WeightedRandomSampler that upsamples minority classes."""
    counts  = torch.tensor(dataset.class_counts, dtype=torch.float)
    weights = 1.0 / counts
    sample_weights = torch.tensor(
        [weights[lbl] for _, lbl in dataset.samples]
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset:      LabelledImageDataset,
    batch_size:   int  = 64,
    shuffle:      bool = False,
    use_sampler:  bool = False,    # weighted sampler for training
    num_workers:  int  = 4,
    pin_memory:   bool = True,
) -> DataLoader:
    sampler = make_weighted_sampler(dataset) if use_sampler else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(sampler is not None),
    )
