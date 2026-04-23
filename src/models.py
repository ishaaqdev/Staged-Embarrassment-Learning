"""
ResNet-18 architecture adapted for CIFAR-10 (32x32 inputs).

Modifications from standard ImageNet ResNet-18:
    - conv1 kernel reduced from 7x7 to 3x3 (stride=1, padding=1)
    - maxpool layer replaced with Identity (no spatial downsampling on small inputs)
    - final fc layer outputs 10 classes instead of 1000
"""

import torch
import torch.nn as nn
from torchvision import models

from .data_utils import N_CLASSES, DEVICE


def build_model():
    """
    Construct a CIFAR-10 adapted ResNet-18 with pretrained ImageNet weights.

    Returns:
        nn.Module: ResNet-18 model moved to the active device.
    """
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Adapt for 32x32 CIFAR-10 inputs
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, N_CLASSES)

    return m.to(DEVICE)


def count_params(model):
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def count_flops(sparsity, n_samples, total_params):
    """
    Estimate forward+backward FLOPs for a batch.

    Args:
        sparsity:     fraction of parameters frozen (0.0 to 1.0)
        n_samples:    number of training samples in the batch
        total_params: total parameter count of the model

    Returns:
        float: estimated FLOPs
    """
    return 2 * total_params * (1.0 - sparsity) * n_samples
