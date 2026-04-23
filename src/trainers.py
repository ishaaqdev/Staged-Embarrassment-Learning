"""
Training loops for standard (dense) and SEL (sparse) training.

Two training modes are provided:

1. run_dense_epoch  -- Standard full-gradient training (baseline / warmup)
2. run_sel_epoch    -- SEL training with embarrassment scoring and sparse updates

Both return per-epoch metrics compatible with the benchmark logging format.
"""

import numpy as np
import torch
import torch.nn as nn

from .data_utils import N_CLASSES, DEVICE
from .sel_engine import pc_embarrassment, sparse_update


# ---------------------------------------------------------------------------
# Dense Training (Baseline / Warmup)
# ---------------------------------------------------------------------------
def run_dense_epoch(model, loader, optimizer):
    """
    Standard training epoch with full gradient updates.

    Args:
        model:     nn.Module
        loader:    DataLoader for training data
        optimizer: torch.optim.Optimizer

    Returns:
        (avg_loss, accuracy): tuple of epoch-level metrics
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# SEL Training (Sparse Updates)
# ---------------------------------------------------------------------------
def run_sel_epoch(model, loader, optimizer, gamma, base_lr=1e-3):
    """
    SEL training epoch with embarrassment scoring and sparse gradient updates.

    For each batch:
        1. Forward pass and loss computation
        2. Per-class embarrassment (E) and confidence (C) measured
        3. Learning rate scaled by loss magnitude: lr = base_lr * (1 + 2*loss)
        4. Backward pass
        5. Guilt threshold mask applied to gradients
        6. Optimizer step with sparse gradients

    Args:
        model:     nn.Module
        loader:    DataLoader for the current curriculum stage
        optimizer: torch.optim.Optimizer
        gamma:     guilt threshold for gradient masking
        base_lr:   base learning rate before scaling

    Returns:
        (avg_loss, accuracy, avg_sparsity, avg_E, avg_C):
            avg_loss     - mean training loss
            accuracy     - training accuracy
            avg_sparsity - mean gradient sparsity across batches
            avg_E        - mean per-class embarrassment [N_CLASSES]
            avg_C        - mean per-class confidence [N_CLASSES]
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    E_sum = np.zeros(N_CLASSES)
    C_sum = np.zeros(N_CLASSES)
    sparsity_sum = 0
    n_batches = 0
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # Measure embarrassment before gradient update
        E_batch, C_batch = pc_embarrassment(outputs.detach(), labels)
        E_sum += E_batch
        C_sum += C_batch

        # Scale learning rate by loss magnitude
        for param_group in optimizer.param_groups:
            param_group["lr"] = base_lr * (1.0 + 2.0 * loss.item())

        loss.backward()

        # Apply guilt threshold mask
        sparsity = sparse_update(model, gamma)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        sparsity_sum += sparsity
        n_batches += 1

    return (
        total_loss / total,
        correct / total,
        sparsity_sum / n_batches,
        E_sum / n_batches,
        C_sum / n_batches,
    )


# ---------------------------------------------------------------------------
# Evaluation Utilities
# ---------------------------------------------------------------------------
def eval_full(model, test_loader):
    """
    Evaluate model on the full test set.

    Returns:
        float: accuracy on the full test set
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def eval_per_class(model, held_out_loader):
    """
    Evaluate model on the held-out test set (100 images per class).

    Returns:
        (per_class_acc, overall_acc):
            per_class_acc - numpy array of shape [N_CLASSES]
            overall_acc   - scalar accuracy across all held-out samples
    """
    model.eval()
    class_correct = [0] * N_CLASSES
    class_total = [0] * N_CLASSES

    with torch.no_grad():
        for imgs, labels in held_out_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (preds[i] == labels[i]).item()
                class_total[label] += 1

    per_class = np.array([
        class_correct[c] / max(class_total[c], 1) for c in range(N_CLASSES)
    ])
    overall = sum(class_correct) / max(sum(class_total), 1)

    return per_class, overall
