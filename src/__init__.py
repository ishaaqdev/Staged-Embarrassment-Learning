"""
SEL -- Staged Embarrassment Learning

A curriculum-based training method that applies dynamic gradient sparsity
to reduce computational overhead without sacrificing accuracy.

Modules:
    models      - ResNet-18 architecture adapted for CIFAR-10
    sel_engine  - Core math: Embarrassment (E), Confidence (C), Sparse Updates
    trainers    - Standard and SEL training loops
    data_utils  - CIFAR-10 loading and class-sorted pool construction
"""

__version__ = "1.0.0"
__author__ = "Ishaaq"

from .models import build_model
from .sel_engine import pc_embarrassment, sparse_update
from .trainers import run_dense_epoch, run_sel_epoch
from .data_utils import CLASS_NAMES, N_CLASSES
