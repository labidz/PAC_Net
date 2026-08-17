"""Checkpoint utilities for PAC-Net training."""

from __future__ import annotations
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def save_checkpoint(
    state:   Dict[str, Any],
    path:    str,
    is_best: bool = False,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = str(path).replace('.pt', '_best.pt')
        torch.save(state, best_path)


def load_checkpoint(path: str, model, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt.get('epoch', 0), ckpt.get('best_val_metric', 0.0)


def make_checkpoint(model, optimizer, epoch, metrics, config):
    return {
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch':           epoch,
        'metrics':         metrics,
        'config':          config,
    }
