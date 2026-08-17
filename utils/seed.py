"""Reproducible seeding for all libraries."""

import os
import random
import numpy as np
import torch


SEEDS = [42, 43, 44, 45, 46]   # 5-seed protocol


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (slight speed cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
