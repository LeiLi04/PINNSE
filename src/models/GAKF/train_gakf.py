from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PARENTS = CURRENT_DIR.parents
PROJECT_ROOT = PARENTS[3] if len(PARENTS) >= 4 else CURRENT_DIR
SRC_ROOT = PARENTS[2] if len(PARENTS) >= 3 else CURRENT_DIR
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.models.GAKF.discriminator import GAKFDiscriminator
from src.models.GAKF.generator import GAKFGenerator
from src.models.GAKF import losses, utils

import numpy as np
import torch
from torch import nn, optim
from torch.cuda import amp
from tqdm import tqdm

