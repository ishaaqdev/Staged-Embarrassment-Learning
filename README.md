# Staged Embarrassment Learning (SEL) -- Efficiency Benchmark

**A Comparative Analysis of Dynamic Gradient Sparsity vs. Magnitude Pruning on CIFAR-10**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-T4_GPU-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-FF6F00?style=flat-square)
![Architecture](https://img.shields.io/badge/Model-ResNet--18-4A90D9?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research_Complete-blueviolet?style=flat-square)
[![Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-222?style=flat-square&logo=github&logoColor=white)](https://ishaaqdev.github.io/Staged-Embarrassment-Learning/)

**[View Interactive Results Page](https://ishaaqdev.github.io/Staged-Embarrassment-Learning/)**

---

## Overview

This repository presents a complete, reproducible benchmark for **Staged Embarrassment Learning (SEL)**, a curriculum-based training method that applies dynamic gradient sparsity to reduce computational overhead without sacrificing accuracy.

We compare SEL against the **Lottery Ticket Hypothesis** and a **standard ResNet-18 Baseline** on CIFAR-10, using a held-out test set of 100 images per class (1000 total, never seen during training).

The core insight: instead of updating all parameters equally, SEL measures per-class *embarrassment* and suppresses gradient updates for parameters the model is already confident about. The result is **95--99% reduction in effective FLOPs** with a controlled accuracy trade-off.

---

## Origin

The idea for SEL came from observing a niece learning to catch a ball. She missed. Someone nearby laughed. She felt embarrassed -- and in that instant, something remarkable happened: she didn't just try harder, she **corrected her exact mistake** with a precision she hadn't shown before.

That emotional signal -- embarrassment -- triggered a **targeted, high-efficiency correction**. She wasn't recalibrating everything she knew. She was zeroing in on exactly what went wrong.

Traditional neural networks don't do this. They apply gradients uniformly, spending compute on easy samples they already know perfectly. SEL asks: **what if we only updated weights where the model is genuinely embarrassed?**

The result is a training algorithm that mirrors this human learning instinct -- suppressing gradient updates for confident, easy predictions, and concentrating all available compute on the samples the model finds hardest to explain.

---

## How It Works

```
Input --> Prediction --> Error Detection --> Embarrassment Signal --> Sparse Update --> Improved Output
                                |                    |
                                v                    v
                        Per-class loss       Guilt threshold mask
                        with temperature     zeros out confident
                        scaling (T=1.5)      gradients (p40)
```

**Step-by-step:**

1. A training sample passes through the network and a prediction is made.
2. Per-class embarrassment `E_c` is measured via temperature-scaled cross-entropy loss.
3. Gradients below the guilt threshold (`gamma`) are masked to zero.
4. Only the most significant gradients survive -- frozen knowledge stays frozen.
5. Training progresses through 5 stages from easy to hard samples, naturally escalating difficulty.

---

## Mathematical Foundation

### Per-Class Embarrassment

For a class `c` with temperature `T`:

```
E_c = (1 / |N_c|) * sum( L(y_hat_i / T, y_i) )   for i in N_c
```

Where `T = 1.5` is the temperature parameter, `N_c` is the set of samples from class `c`, and `L` is cross-entropy loss. Confidence is defined as `C_c = max(0, 1 - E_c)`.

### Sparse Gradient Update

Gradient sparsity is applied via a mask based on the guilt threshold (`gamma`):

```
Mask    = |grad(p)| > gamma
grad(p) <- grad(p) * Mask
```

`gamma` is set at the p40 percentile of gradient magnitudes, producing approximately 95% sparsity. The remaining 5% of gradients carry all the learning signal.

### Core Implementation

```python
def sparse_update(model, gamma):
    """Apply guilt threshold mask to gradients. Returns sparsity fraction."""
    tot = guilty = 0
    for p in model.parameters():
        if p.grad is not None:
            mask = (p.grad.abs() > gamma).float()
            p.grad.mul_(mask)
            tot    += mask.numel()
            guilty += mask.sum().item()
    return 1.0 - (guilty / max(tot, 1))
```

---

## Experiment Methodology

All experiments use ResNet-18 on CIFAR-10 with a held-out test set of 100 images per class (1000 total, never seen during training). All runs on a T4 GPU.

| # | Experiment | Description | Epochs |
|---|------------|-------------|--------|
| 1 | **Baseline CNN** | Standard ResNet-18 on full data | 30 |
| 2 | **Lottery Ticket** | Global magnitude pruning (80%) followed by retraining from scratch | 30 |
| 3 | **SEL-95%** | Staged sparsity with target 95% frozen gradients, 5 curriculum stages | 50 |
| 4 | **Warmup+SEL** | 5 epochs of dense warmup followed by SEL staged curriculum | 55 |

### Stage Configuration (SEL)

| Stage | Difficulty | Data Range | Guilt Decay | Epochs |
|-------|-----------|------------|-------------|--------|
| Stage 1 | Easy | 0--25% | 1.00x | 10 |
| Stage 2 | Medium | 20--50% | 0.94x | 10 |
| Stage 3 | Hard | 40--65% | 0.88x | 10 |
| Stage 4 | Harder | 60--82% | 0.82x | 10 |
| Stage 5 | Hardest | 78--100% | 0.76x | 10 |

---

## Results

| System | Test Acc (100/class) | FLOPs (T) | Time (s) | FLOPs Saved |
|:---|:---|:---|:---|:---|
| **Baseline CNN** | 93.2% | 33.5 | 1398 | 0% |
| **Lottery Ticket** | 93.0% | 6.7 | 1460 | 80% |
| **SEL-95%** | 77.5% | 0.084 | 835 | 99% |
| **Warmup+SEL** | 85.3% | 1.58 | 875 | 95% |

### Key Findings

- **SEL-95%** achieves **99% FLOPs reduction** at the cost of 15.7pp accuracy.
- **Warmup+SEL** recovers ~8pp of that gap while still saving **95% of compute**.
- Training wall-clock time drops by **38%** versus baseline.
- The embarrassment signal provides interpretable per-class difficulty tracking throughout training.
- Lottery Ticket achieves near-baseline accuracy with 80% savings, making it the strongest competitor for accuracy-critical use cases.

---

## Repository Structure

```
SEL-Benchmark/
|
|-- src/                               # Core Python library
|   |-- __init__.py                    # Package init with public API exports
|   |-- models.py                      # ResNet-18 architecture (CIFAR-10 adapted)
|   |-- sel_engine.py                  # Embarrassment (E), Confidence (C), sparse_update()
|   |-- trainers.py                    # Dense and SEL training loops + evaluation
|   |-- data_utils.py                  # CIFAR-10 loading, class sorting, stage loaders
|
|-- experiments/
|   |-- run_benchmark.py               # Main entry point -- reproduces the full 4-experiment suite
|
|-- notebooks/
|   |-- sel_4experiment_benchmark.ipynb # Original Colab notebook with all visualizations
|
|-- results/                           # Experimental output (CSV data from actual runs)
|   |-- hist_baseline.csv              # Epoch-by-epoch metrics: Baseline CNN
|   |-- hist_lottery.csv               # Epoch-by-epoch metrics: Lottery Ticket
|   |-- hist_sel95.csv                 # Epoch-by-epoch metrics: SEL-95%
|   |-- hist_warmup.csv               # Epoch-by-epoch metrics: Warmup+SEL
|
|-- assets/                            # Images and visual outputs
|   |-- benchmark_results.png          # Full 14-panel benchmark visualization
|
|-- docs/                              # Interactive results page (GitHub Pages)
|   |-- index.html                     # Self-contained HTML with Chart.js visualizations
|
|-- .gitignore                         # Python-specific ignore rules
|-- LICENSE                            # MIT License
|-- requirements.txt                   # Dependency versions
|-- README.md                          # This file
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** (T4 or better recommended; CPU will work but ~10x slower)
- **pip** package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ishaaqdev/Staged-Embarrassment-Learning.git
cd Staged-Embarrassment-Learning

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Full Benchmark (Python)

The `experiments/run_benchmark.py` script reproduces all 4 experiments from scratch. This is the recommended way to validate results independently.

```bash
# Run the complete benchmark suite (~45 min on T4 GPU)
python experiments/run_benchmark.py
```

**What this does:**
1. Downloads CIFAR-10 automatically to `./data/`
2. Builds a ResNet-18 adapted for 32x32 inputs
3. Sorts all training data per class by difficulty (easy to hard)
4. Calibrates the guilt threshold `gamma` at the p40 percentile
5. Runs all 4 experiments sequentially:
   - Experiment 1: Baseline CNN (30 epochs)
   - Experiment 2: Lottery Ticket (15 pretrain + 30 retrain epochs)
   - Experiment 3: SEL-95% (50 epochs across 5 stages)
   - Experiment 4: Warmup+SEL (5 warmup + 50 staged epochs)
6. Prints a final summary table
7. Saves epoch-by-epoch CSV results to `results/`

**Output:**
```
============================================================
  BENCHMARK RESULTS -- HELD-OUT TEST (100 IMAGES PER CLASS)
============================================================
  System              Full Test   100/class     FLOPs     Time        Saved
------------------------------------------------------------------------
  Baseline CNN          92.5%      93.2%     33.52T    1398s          0%
  Lottery Ticket        93.0%      93.0%      6.70T    1460s         80%
  SEL-95%               77.5%      77.5%      0.08T     835s         99%
  Warmup+SEL            85.3%      85.3%      1.58T     875s         95%
```

### Running the Notebook (Google Colab)

The original Colab notebook contains all experiments, visualizations, per-class breakdowns, and the 14-panel benchmark figure.

1. Open `notebooks/sel_4experiment_benchmark.ipynb`
2. Upload to [Google Colab](https://colab.research.google.com/)
3. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
4. Run all cells sequentially (Runtime > Run all)
5. The notebook will:
   - Train all 4 models
   - Generate per-class embarrassment/confidence curves
   - Produce the full benchmark visualization (`benchmark_results.png`)
   - Download all CSV result files automatically

**Estimated runtime:** ~45 minutes on a T4 GPU.

### Using the Source Modules Directly

The `src/` package is designed for reuse. You can import individual components for custom experiments:

```python
# Import the core library
from src.models import build_model, count_params
from src.sel_engine import pc_embarrassment, sparse_update, calibrate_guilt_threshold
from src.trainers import run_dense_epoch, run_sel_epoch, eval_full, eval_per_class
from src.data_utils import (
    load_cifar10, build_loaders, build_held_out_loader,
    sort_classes_by_difficulty, get_stage_loader, CLASS_NAMES
)

# Build model
model = build_model()
print(f"Parameters: {count_params(model):,}")

# Load data
train_data, test_data = load_cifar10()
train_loader, test_loader = build_loaders(train_data, test_data)
held_out = build_held_out_loader(test_data)

# Sort by difficulty
class_pools = sort_classes_by_difficulty(train_data, model)

# Calibrate guilt threshold
gamma = calibrate_guilt_threshold(model, train_data, base_lr=1e-3, percentile=40)

# Run a single SEL epoch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
stage_loader = get_stage_loader(0, STAGE_CONFIG, class_pools, train_data)
loss, acc, sparsity, E, C = run_sel_epoch(model, stage_loader, optimizer, gamma)

# Evaluate
per_class_acc, overall_acc = eval_per_class(model, held_out)
```

### Viewing the Interactive Results Page

The `docs/index.html` file is a self-contained, zero-dependency page (uses Chart.js from CDN) that visualizes all benchmark results interactively.

```bash
# Option 1: Open directly in browser
# Just double-click docs/index.html

# Option 2: Serve locally
python -m http.server 8765 --directory docs
# Then open http://localhost:8765 in your browser
```

**Features:**
- Accuracy curves across all 4 experiments (test + train)
- Cumulative FLOPs and training time comparisons
- Accuracy vs. FLOPs Saved Pareto frontier (bubble chart)
- Per-class accuracy breakdown (Warmup+SEL vs Baseline)
- Interactive tooltips with exact values on hover
- Tab switching between chart views
- Fully responsive (works on mobile and desktop)

---

## Applications

- **Edge Devices** -- 99% FLOPs reduction makes on-device fine-tuning feasible on microcontrollers with extreme power constraints.
- **Real-Time Robotics** -- Robots can continuously learn from novel situations without replaying entire datasets.
- **Adaptive Education AI** -- Personalized tutoring systems that track per-concept embarrassment scores and focus practice on knowledge gaps.
- **Federated Learning** -- Sparse gradient updates dramatically reduce communication overhead in distributed training.
- **Continual Learning** -- By freezing confident knowledge and updating only on embarrassing samples, SEL naturally resists catastrophic forgetting.
- **Foundation Model Fine-Tuning** -- Apply SEL's sparse update logic to LoRA-style fine-tuning, cutting the cost of adapting large models to new domains.

---

## Trade-offs and Limitations

### What Works

- Massive FLOPs reduction (95--99%) with manageable accuracy cost
- Warmup+SEL recovers ~8pp accuracy vs pure SEL-95 at minimal extra cost
- Training wall-clock time drops by 38% vs baseline
- Natural curriculum prevents early overfitting on easy samples
- Embarrassment signal provides interpretable per-class difficulty tracking

### Known Limitations

- **Accuracy gap:** SEL-95 achieves only 77.5% vs 93.2% baseline -- a 15.7pp accuracy gap. When accuracy is critical, the Lottery Ticket method (93.0% with 80% savings) is a stronger choice.
- **Threshold sensitivity:** The guilt threshold `gamma` requires careful calibration. Setting it too high kills convergence entirely; too low removes the sparsity benefit. Currently uses a fixed p40 percentile.
- **Stage transition instability:** Stage boundaries can cause temporary accuracy dips as the model encounters a new difficulty distribution. This is visible in the training curves.
- **Limited evaluation scope:** All experiments are on CIFAR-10 with ResNet-18. Generalization to larger datasets (ImageNet), deeper architectures (ResNet-50, ViT), or different domains (NLP, audio) is untested.
- **No adaptive gamma:** The guilt threshold does not auto-adjust during training. A per-class adaptive threshold could significantly improve results.
- **Single-run results:** Reported numbers are from a single seed (42). Variance across seeds has not been characterized.
- **Samples per class cap:** The current stage loader caps at 2000 samples per class per stage, which may underrepresent certain difficulty distributions.

---

## Future Scope

| Timeline | Direction | Description |
|----------|-----------|-------------|
| **Near-term** | Adaptive gamma | Auto-tuning guilt threshold per class, removing manual calibration |
| **Medium-term** | Transformer SEL | Per-head embarrassment scoring as an attention-aware sparsity method |
| **Long-term** | Universal signal | Combining embarrassment with reinforcement learning for agents that prioritize surprising state transitions |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | Model definition, training, GPU acceleration |
| Augmentation | ![TorchVision](https://img.shields.io/badge/TorchVision-0.15+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) | Transforms, pretrained weights, CIFAR-10 |
| Numerics | ![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white) | Percentile calculations, array operations |
| Data | ![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white) | CSV export, result logging |
| Visualization | ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=flat-square) | Benchmark figure generation (notebook) |
| Web | ![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?style=flat-square&logo=chartdotjs&logoColor=white) | Interactive results page |
| Font | ![Fira Code](https://img.shields.io/badge/Font-Fira_Code-333?style=flat-square) | Monospace typography (web page) |

---

## CSV Data Format

Each CSV in `results/` contains per-epoch metrics:

| Column | Type | Description |
|--------|------|-------------|
| `epoch` | int | Epoch number |
| `train_acc` | float | Training set accuracy |
| `train_loss` | float | Training set loss |
| `test_full` | float | Full test set accuracy (10,000 images) |
| `test_100` | float | Held-out test accuracy (100/class) |
| `flops_T` | float | Cumulative FLOPs in teraflops |
| `time_s` | float | Wall-clock time in seconds |
| `sparsity` | float | Gradient sparsity fraction |
| `acc_{class}` | float | Per-class accuracy (10 columns) |
| `E_{class}` | float | Per-class embarrassment (SEL only) |
| `C_{class}` | float | Per-class confidence (SEL only) |

---

## Links

- [LinkedIn](https://www.linkedin.com/in/ishaaq42/)
- [GitHub](https://github.com/ishaaqdev)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
