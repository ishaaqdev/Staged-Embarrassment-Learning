"""
CIFAR-10 data loading and per-class difficulty sorting.

This module provides:
    - Standard train/test transforms with augmentation
    - A held-out test set of 100 images per class (1000 total)
    - Per-class sorted pools ranked by loss (easy -> hard)
    - Stage-specific and warmup data loaders
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
N_CLASSES = 10
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
])


# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------
def load_cifar10(data_root="./data"):
    """
    Download and return CIFAR-10 train/test datasets.

    Args:
        data_root: directory to store/load cached data

    Returns:
        (full_train, full_test): CIFAR-10 Dataset objects
    """
    full_train = datasets.CIFAR10(
        data_root, train=True, download=True, transform=TRAIN_TRANSFORM
    )
    full_test = datasets.CIFAR10(
        data_root, train=False, download=True, transform=TEST_TRANSFORM
    )
    return full_train, full_test


def build_loaders(full_train, full_test, batch_size=BATCH_SIZE):
    """
    Build standard DataLoaders for full train and full test sets.

    Returns:
        (base_loader, full_test_loader): DataLoader pair
    """
    base_loader = DataLoader(
        full_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    full_test_loader = DataLoader(
        full_test, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return base_loader, full_test_loader


def build_held_out_loader(full_test, n_per_class=100, batch_size=BATCH_SIZE):
    """
    Construct a held-out evaluation loader with exactly n_per_class images
    per class. These images are never seen during training.

    Args:
        full_test:   CIFAR-10 test dataset
        n_per_class: number of samples to hold out per class

    Returns:
        DataLoader over the held-out subset
    """
    indices = []
    for c in range(N_CLASSES):
        class_idx = [i for i, (_, label) in enumerate(full_test) if label == c]
        indices.extend(class_idx[:n_per_class])
    return DataLoader(
        Subset(full_test, indices),
        batch_size=batch_size, shuffle=False, num_workers=2,
    )


# ---------------------------------------------------------------------------
# Per-Class Sorted Pools
# ---------------------------------------------------------------------------
def sort_classes_by_difficulty(full_train, scorer_model):
    """
    Score every training sample using a model and sort each class
    by loss (ascending = easy to hard).

    Args:
        full_train:    CIFAR-10 training dataset
        scorer_model:  nn.Module used to compute per-sample loss

    Returns:
        dict[int, list[int]]: class_id -> sorted list of dataset indices
    """
    scorer_model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_loss, all_labels = [], []

    with torch.no_grad():
        loader = DataLoader(full_train, batch_size=256, shuffle=False, num_workers=2)
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            losses = criterion(scorer_model(imgs), labels)
            all_loss.extend(losses.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_loss = np.array(all_loss)
    all_labels = np.array(all_labels)

    class_pools = {}
    for c in range(N_CLASSES):
        idx = np.where(all_labels == c)[0]
        order = np.argsort(all_loss[idx])
        class_pools[c] = idx[order].tolist()

    return class_pools


def get_stage_loader(stage_index, stage_config, class_pools, full_train,
                     samples_per_class=2000, batch_size=BATCH_SIZE):
    """
    Build a DataLoader for a specific curriculum stage.

    Each stage defines a difficulty window [pct_start, pct_end] within the
    sorted per-class pools.

    Args:
        stage_index:      index into stage_config
        stage_config:     list of (name, pct_start, pct_end, threshold, n_epochs)
        class_pools:      output of sort_classes_by_difficulty
        full_train:       CIFAR-10 training dataset
        samples_per_class: max samples per class per stage

    Returns:
        DataLoader for the stage
    """
    _, pct_start, pct_end, _, _ = stage_config[stage_index]
    indices = []
    for c in range(N_CLASSES):
        pool = class_pools[c]
        start = int(pct_start * len(pool))
        end = min(int(pct_end * len(pool)), start + samples_per_class)
        indices += pool[start:end]
    random.shuffle(indices)
    return DataLoader(
        Subset(full_train, indices),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )


def build_warmup_loader(class_pools, full_train, n_easy_per_class=1000,
                        batch_size=BATCH_SIZE):
    """
    Build a DataLoader containing only the easiest samples for warmup training.

    Args:
        class_pools:      output of sort_classes_by_difficulty
        full_train:       CIFAR-10 training dataset
        n_easy_per_class: number of easiest samples per class to include

    Returns:
        DataLoader for warmup phase
    """
    indices = []
    for c in range(N_CLASSES):
        indices += class_pools[c][:n_easy_per_class]
    return DataLoader(
        Subset(full_train, indices),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
