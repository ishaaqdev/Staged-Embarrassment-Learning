"""
run_benchmark.py -- Main entry point for the SEL 4-Experiment Benchmark.

Reproduces the full benchmark suite:
    Experiment 1: Baseline CNN (ResNet-18, full data, 30 epochs)
    Experiment 2: Lottery Ticket (80% global magnitude pruning + retrain)
    Experiment 3: SEL-95% (staged embarrassment, p40 guilt threshold)
    Experiment 4: Warmup+SEL (5 dense warmup epochs + staged embarrassment)

Usage:
    python experiments/run_benchmark.py

All results are saved to results/ as CSV files.
Runtime: ~45 minutes on a T4 GPU.
"""

import sys
import os
import time
import copy
import random
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import build_model, count_params, count_flops
from src.sel_engine import calibrate_guilt_threshold, sparse_update
from src.trainers import run_dense_epoch, run_sel_epoch, eval_full, eval_per_class
from src.data_utils import (
    CLASS_NAMES, N_CLASSES, DEVICE, BATCH_SIZE,
    load_cifar10, build_loaders, build_held_out_loader,
    sort_classes_by_difficulty, get_stage_loader, build_warmup_loader,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_LR = 1e-3
BASELINE_EPOCHS = 30
LT_PRUNE_PCT = 0.80
LT_EPOCHS = 30
WARMUP_EPOCHS = 5

STAGE_CONFIG = [
    ("Stage 1 Easy",    0.00, 0.25, 0.30, 10),
    ("Stage 2 Medium",  0.20, 0.50, 0.45, 10),
    ("Stage 3 Hard",    0.40, 0.65, 0.58, 10),
    ("Stage 4 Harder",  0.60, 0.82, 0.68, 10),
    ("Stage 5 Hardest", 0.78, 1.00, 0.76, 10),
]
SAMPLES_PER_CLASS = 2000
STAGED_EPOCHS = sum(s[4] for s in STAGE_CONFIG)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Load data
    full_train, full_test = load_cifar10()
    base_loader, full_test_loader = build_loaders(full_train, full_test)
    held_out_loader = build_held_out_loader(full_test)

    # Count params
    tmp = build_model()
    total_params = count_params(tmp)
    print(f"ResNet-18 params:  {total_params:,}")
    print(f"Dense FLOPs/epoch: {2 * total_params * 50000 / 1e12:.3f}T")

    # Sort classes by difficulty
    print("Sorting training data per class...")
    class_pools = sort_classes_by_difficulty(full_train, tmp)
    for c in range(N_CLASSES):
        print(f"  {CLASS_NAMES[c]:<12}: {len(class_pools[c])} samples")

    # Calibrate guilt threshold
    guilt_gamma = calibrate_guilt_threshold(tmp, full_train, BASE_LR, percentile=40)
    del tmp
    print(f"Guilt threshold (p40): {guilt_gamma:.6f}")

    # -----------------------------------------------------------------------
    # Experiment 1: Baseline CNN
    # -----------------------------------------------------------------------
    separator("EXP 1 -- BASELINE CNN")
    e1_model = build_model()
    e1_opt = optim.Adam(e1_model.parameters(), lr=BASE_LR)
    e1_hist = []
    e1_flops = 0.0
    e1_t0 = time.time()

    for ep in range(BASELINE_EPOCHS):
        tl, ta = run_dense_epoch(e1_model, base_loader, e1_opt)
        e1_flops += 2 * total_params * len(full_train)
        fa = eval_full(e1_model, full_test_loader)
        pc, t100 = eval_per_class(e1_model, held_out_loader)

        row = {
            "epoch": ep + 1, "train_acc": ta, "train_loss": tl,
            "test_full": fa, "test_100": t100,
            "flops_T": e1_flops / 1e12, "time_s": time.time() - e1_t0,
            "sparsity": 0.0,
        }
        for ci, cn in enumerate(CLASS_NAMES):
            row[f"acc_{cn}"] = pc[ci]
        e1_hist.append(row)
        print(f"  Ep{ep+1:2d}/{BASELINE_EPOCHS}  full={fa:.1%}  t100={t100:.1%}  train={ta:.1%}")

    e1_elapsed = time.time() - e1_t0
    e1_df = pd.DataFrame(e1_hist)
    E1_PC, E1_100 = eval_per_class(e1_model, held_out_loader)
    E1_FULL = eval_full(e1_model, full_test_loader)
    print(f"\nBaseline: full={E1_FULL:.1%}  t100={E1_100:.1%}  {e1_elapsed:.0f}s  {e1_flops/1e12:.1f}T")

    # -----------------------------------------------------------------------
    # Experiment 2: Lottery Ticket
    # -----------------------------------------------------------------------
    separator("EXP 2 -- LOTTERY TICKET")
    lt_model = build_model()
    lt_init = copy.deepcopy(lt_model.state_dict())
    lt_opt = optim.Adam(lt_model.parameters(), lr=BASE_LR)

    print("Phase 1: Initial training (15 epochs to find tickets)...")
    for ep in range(15):
        run_dense_epoch(lt_model, base_loader, lt_opt)
        if (ep + 1) % 5 == 0:
            print(f"  Phase1 Ep{ep+1}/15  test={eval_full(lt_model, full_test_loader):.1%}")

    # Find winning tickets
    all_weights = np.concatenate([
        p.data.abs().cpu().numpy().flatten() for p in lt_model.parameters()
    ])
    threshold = np.percentile(all_weights, LT_PRUNE_PCT * 100)
    lt_masks = [(p.data.abs() > threshold).float() for p in lt_model.parameters()]
    total_kept = sum(m.sum().item() for m in lt_masks)
    total_all = sum(m.numel() for m in lt_masks)
    print(f"\nTickets: {total_kept/total_all:.1%} kept ({LT_PRUNE_PCT:.0%} pruned)")

    # Reset and retrain
    lt_model.load_state_dict(lt_init)
    lt_opt = optim.Adam(lt_model.parameters(), lr=BASE_LR)
    e2_hist = []
    e2_flops = 0.0
    e2_t0 = time.time()

    print("\nPhase 2: Retrain winning tickets...")
    for ep in range(LT_EPOCHS):
        lt_model.train()
        crit = nn.CrossEntropyLoss()
        ls = cor = tot = sp_sum = nb = 0

        for imgs, labels in base_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            lt_opt.zero_grad()
            out = lt_model(imgs)
            loss = crit(out, labels)
            loss.backward()

            for p, mask in zip(lt_model.parameters(), lt_masks):
                if p.grad is not None:
                    p.grad.mul_(mask)
            lt_opt.step()
            with torch.no_grad():
                for p, mask in zip(lt_model.parameters(), lt_masks):
                    p.data.mul_(mask)

            active = sum(m.sum().item() for m in lt_masks)
            sp = 1.0 - active / total_all
            ls += loss.item() * labels.size(0)
            cor += (out.argmax(1) == labels).sum().item()
            tot += labels.size(0)
            sp_sum += sp
            nb += 1
            e2_flops += count_flops(sp, labels.size(0), total_params)

        fa = eval_full(lt_model, full_test_loader)
        pc, t100 = eval_per_class(lt_model, held_out_loader)

        row = {
            "epoch": ep + 1, "train_acc": cor / tot, "train_loss": ls / tot,
            "test_full": fa, "test_100": t100,
            "flops_T": e2_flops / 1e12, "time_s": time.time() - e2_t0,
            "sparsity": sp_sum / nb,
        }
        for ci, cn in enumerate(CLASS_NAMES):
            row[f"acc_{cn}"] = pc[ci]
        e2_hist.append(row)
        print(f"  Ep{ep+1:2d}/{LT_EPOCHS}  full={fa:.1%}  t100={t100:.1%}  sp={sp_sum/nb:.0%}")

    e2_elapsed = time.time() - e2_t0
    e2_df = pd.DataFrame(e2_hist)
    E2_PC, E2_100 = eval_per_class(lt_model, held_out_loader)
    E2_FULL = eval_full(lt_model, full_test_loader)
    print(f"\nLottery: full={E2_FULL:.1%}  t100={E2_100:.1%}  {e2_elapsed:.0f}s  {e2_flops/1e12:.2f}T")

    # -----------------------------------------------------------------------
    # Experiment 3: SEL-95%
    # -----------------------------------------------------------------------
    separator("EXP 3 -- SEL 95% SPARSITY")
    e3_model = build_model()
    e3_opt = optim.Adam(e3_model.parameters(), lr=BASE_LR)
    e3_hist = []
    e3_flops = 0.0
    ep_num = 0
    e3_t0 = time.time()

    for si, (sname, ps, pe, thresh, n_ep) in enumerate(STAGE_CONFIG):
        gamma = guilt_gamma * (1.0 - 0.3 * si / len(STAGE_CONFIG))
        loader = get_stage_loader(si, STAGE_CONFIG, class_pools, full_train, SAMPLES_PER_CLASS)
        print(f"\n  {sname}  gamma={gamma:.5f}")

        for ep in range(n_ep):
            tl, ta, sp, avg_E, avg_C = run_sel_epoch(e3_model, loader, e3_opt, gamma, BASE_LR)
            e3_flops += count_flops(sp, len(loader.dataset), total_params)
            fa = eval_full(e3_model, full_test_loader)
            pc, t100 = eval_per_class(e3_model, held_out_loader)
            ep_num += 1

            row = {
                "epoch": ep_num, "stage": si + 1, "train_acc": ta, "train_loss": tl,
                "test_full": fa, "test_100": t100,
                "flops_T": e3_flops / 1e12, "time_s": time.time() - e3_t0,
                "sparsity": sp,
            }
            for ci, cn in enumerate(CLASS_NAMES):
                row[f"E_{cn}"] = avg_E[ci]
                row[f"C_{cn}"] = avg_C[ci]
                row[f"acc_{cn}"] = pc[ci]
            e3_hist.append(row)
            print(f"  Ep{ep+1:2d}/{n_ep}  full={fa:.1%}  t100={t100:.1%}  Sp={sp:.0%}")

    e3_elapsed = time.time() - e3_t0
    e3_df = pd.DataFrame(e3_hist)
    E3_PC, E3_100 = eval_per_class(e3_model, held_out_loader)
    E3_FULL = eval_full(e3_model, full_test_loader)
    print(f"\nSEL-95: full={E3_FULL:.1%}  t100={E3_100:.1%}  {e3_elapsed:.0f}s  {e3_flops/1e12:.3f}T")

    # -----------------------------------------------------------------------
    # Experiment 4: Warmup + SEL
    # -----------------------------------------------------------------------
    separator("EXP 4 -- WARMUP + SEL")
    e4_model = build_model()
    e4_opt = optim.Adam(e4_model.parameters(), lr=BASE_LR)
    e4_hist = []
    e4_flops = 0.0
    ep_num = 0
    e4_t0 = time.time()

    warmup_loader = build_warmup_loader(class_pools, full_train)
    print(f"Phase 1: Dense warmup ({WARMUP_EPOCHS} epochs)...")

    for ep in range(WARMUP_EPOCHS):
        tl, ta = run_dense_epoch(e4_model, warmup_loader, e4_opt)
        e4_flops += 2 * total_params * len(warmup_loader.dataset)
        fa = eval_full(e4_model, full_test_loader)
        pc, t100 = eval_per_class(e4_model, held_out_loader)
        ep_num += 1

        row = {
            "epoch": ep_num, "stage": 0, "train_acc": ta, "train_loss": tl,
            "test_full": fa, "test_100": t100,
            "flops_T": e4_flops / 1e12, "time_s": time.time() - e4_t0,
            "sparsity": 0.0,
        }
        for ci, cn in enumerate(CLASS_NAMES):
            row[f"E_{cn}"] = 0.0
            row[f"C_{cn}"] = 0.0
            row[f"acc_{cn}"] = pc[ci]
        e4_hist.append(row)
        print(f"  Warmup Ep{ep+1}/{WARMUP_EPOCHS}  full={fa:.1%}  t100={t100:.1%}")

    print("\nPhase 2: SEL staged training...")
    for si, (sname, ps, pe, thresh, n_ep) in enumerate(STAGE_CONFIG):
        gamma = guilt_gamma * (1.0 - 0.3 * si / len(STAGE_CONFIG))
        loader = get_stage_loader(si, STAGE_CONFIG, class_pools, full_train, SAMPLES_PER_CLASS)
        print(f"\n  {sname}  gamma={gamma:.5f}")

        for ep in range(n_ep):
            tl, ta, sp, avg_E, avg_C = run_sel_epoch(e4_model, loader, e4_opt, gamma, BASE_LR)
            e4_flops += count_flops(sp, len(loader.dataset), total_params)
            fa = eval_full(e4_model, full_test_loader)
            pc, t100 = eval_per_class(e4_model, held_out_loader)
            ep_num += 1

            row = {
                "epoch": ep_num, "stage": si + 1, "train_acc": ta, "train_loss": tl,
                "test_full": fa, "test_100": t100,
                "flops_T": e4_flops / 1e12, "time_s": time.time() - e4_t0,
                "sparsity": sp,
            }
            for ci, cn in enumerate(CLASS_NAMES):
                row[f"E_{cn}"] = avg_E[ci]
                row[f"C_{cn}"] = avg_C[ci]
                row[f"acc_{cn}"] = pc[ci]
            e4_hist.append(row)
            print(f"  Ep{ep+1:2d}/{n_ep}  full={fa:.1%}  t100={t100:.1%}  Sp={sp:.0%}")

    e4_elapsed = time.time() - e4_t0
    e4_df = pd.DataFrame(e4_hist)
    E4_PC, E4_100 = eval_per_class(e4_model, held_out_loader)
    E4_FULL = eval_full(e4_model, full_test_loader)
    print(f"\nWarmup+SEL: full={E4_FULL:.1%}  t100={E4_100:.1%}  {e4_elapsed:.0f}s  {e4_flops/1e12:.3f}T")

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    bf = e1_flops / 1e12
    separator("BENCHMARK RESULTS -- HELD-OUT TEST (100 IMAGES PER CLASS)")
    print(f"  {'System':<18} {'Full Test':>10} {'100/class':>10} {'FLOPs':>9} {'Time':>8} {'Saved':>12}")
    print("-" * 72)
    for label, full, t100, fl, tm in [
        ("Baseline CNN",   E1_FULL, E1_100, e1_flops / 1e12, e1_elapsed),
        ("Lottery Ticket",  E2_FULL, E2_100, e2_flops / 1e12, e2_elapsed),
        ("SEL-95%",        E3_FULL, E3_100, e3_flops / 1e12, e3_elapsed),
        ("Warmup+SEL",     E4_FULL, E4_100, e4_flops / 1e12, e4_elapsed),
    ]:
        saved = (1 - fl / bf) * 100
        print(f"  {label:<18} {full:>9.1%} {t100:>9.1%} {fl:>8.2f}T {tm:>7.0f}s {saved:>11.0f}%")

    # Save results
    e1_df.to_csv(os.path.join(RESULTS_DIR, "hist_baseline.csv"), index=False)
    e2_df.to_csv(os.path.join(RESULTS_DIR, "hist_lottery.csv"), index=False)
    e3_df.to_csv(os.path.join(RESULTS_DIR, "hist_sel95.csv"), index=False)
    e4_df.to_csv(os.path.join(RESULTS_DIR, "hist_warmup.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
