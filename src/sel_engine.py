"""
SEL Engine -- Core mathematics for Staged Embarrassment Learning.

This module implements the two fundamental operations:

1. Per-Class Embarrassment (E_c):
   E_c = (1 / |N_c|) * sum( L(y_hat_i / T, y_i) )  for i in N_c

2. Sparse Gradient Update via Guilt Threshold:
   Mask = |grad(p)| > gamma
   grad(p) <- grad(p) * Mask

Where:
    T     = temperature parameter (default 1.5)
    gamma = guilt threshold (p40 percentile of gradient magnitudes)

The combination produces ~95% gradient sparsity, meaning only 5% of
parameters receive updates each step -- the ones the model is most
"embarrassed" about getting wrong.
"""

import numpy as np
import torch
import torch.nn as nn

from .data_utils import N_CLASSES, DEVICE


# ---------------------------------------------------------------------------
# Guilt Threshold Calibration
# ---------------------------------------------------------------------------
def calibrate_guilt_threshold(model, train_dataset, base_lr, percentile=40):
    """
    Compute the guilt threshold gamma from a single forward-backward pass.

    The threshold is set at the given percentile of absolute gradient
    magnitudes. At p40, approximately 60% of gradients exceed the
    threshold, producing ~95% sparsity after masking.

    Args:
        model:         nn.Module to calibrate
        train_dataset: training dataset for a sample batch
        base_lr:       base learning rate (used for optimizer setup)
        percentile:    percentile cutoff (default 40 = p40)

    Returns:
        float: the calibrated guilt threshold gamma
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader

    opt = optim.Adam(model.parameters(), lr=base_lr)
    loader = DataLoader(train_dataset, batch_size=128)
    imgs, labels = next(iter(loader))

    opt.zero_grad()
    loss = nn.CrossEntropyLoss()(model(imgs.to(DEVICE)), labels.to(DEVICE))
    loss.backward()

    grads = np.concatenate([
        p.grad.abs().cpu().numpy().flatten()
        for p in model.parameters() if p.grad is not None
    ])

    return float(np.percentile(grads, percentile))


# ---------------------------------------------------------------------------
# Per-Class Embarrassment
# ---------------------------------------------------------------------------
def pc_embarrassment(logits, labels, temperature=1.5):
    """
    Compute per-class Embarrassment (E) and Confidence (C) scores.

    E_c = mean cross-entropy loss for class c with temperature scaling.
    C_c = max(0, 1 - E_c)

    Args:
        logits:      model output logits [batch_size, n_classes]
        labels:      ground-truth labels [batch_size]
        temperature: softmax temperature for loss scaling

    Returns:
        (E, C): numpy arrays of shape [N_CLASSES]
    """
    losses = nn.CrossEntropyLoss(reduction="none")(logits / temperature, labels)
    E = np.zeros(N_CLASSES)
    C = np.zeros(N_CLASSES)

    for c in range(N_CLASSES):
        mask = (labels == c)
        if mask.sum() > 0:
            e_c = losses[mask].mean().item()
            E[c] = e_c
            C[c] = max(0.0, 1.0 - e_c)

    return E, C


# ---------------------------------------------------------------------------
# Sparse Gradient Update
# ---------------------------------------------------------------------------
def sparse_update(model, gamma):
    """
    Apply guilt threshold mask to all parameter gradients.

    For each parameter p with gradient grad(p):
        mask = |grad(p)| > gamma
        grad(p) <- grad(p) * mask

    Gradients below the guilt threshold are zeroed out, effectively
    "freezing" the knowledge those parameters encode.

    Args:
        model: nn.Module with computed gradients
        gamma: guilt threshold scalar

    Returns:
        float: sparsity fraction (proportion of gradients zeroed)
    """
    total = guilty = 0

    for p in model.parameters():
        if p.grad is not None:
            mask = (p.grad.abs() > gamma).float()
            p.grad.mul_(mask)
            total += mask.numel()
            guilty += mask.sum().item()

    return 1.0 - (guilty / max(total, 1))
