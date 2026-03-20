<div align="center">

# Cerevia
### Explainable Alzheimer's Stage Detection from Brain MRI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-cerevia.netlify.app-blue?style=for-the-badge)](https://cerevia.netlify.app)
[![Backend API](https://img.shields.io/badge/Backend%20API-HuggingFace%20Spaces-yellow?style=for-the-badge)](https://ved2005-alzheimers-mri-backend.hf.space)
[![API Docs](https://img.shields.io/badge/API%20Docs-Swagger%20UI-green?style=for-the-badge)](https://ved2005-alzheimers-mri-backend.hf.space/docs)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

*A deep learning system that classifies Alzheimer's disease stages from brain MRI scans with visual explainability via Grad-CAM — trained entirely from scratch on 80,000 OASIS neuroimaging samples, with no pretrained weights.*

</div>

---

> ⚠️ **Disclaimer:** Cerevia is a research and educational project. It is not a certified medical device and must not be used for clinical diagnosis.

---

## Table of Contents

- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Architecture Overview](#architecture-overview)
  - [ResNet-18](#resnet-18)
  - [EfficientNet-B0](#efficientnet-b0)
  - [Tiny ViT](#tiny-vit)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Key Findings](#key-findings)
- [Explainability: Grad-CAM](#explainability-grad-cam)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Limitations & Honest Caveats](#limitations--honest-caveats)
- [References](#references)

---

## Motivation

Alzheimer's disease affects over **55 million people worldwide** and is expected to triple by 2050 (World Alzheimer's Report, 2023). Early and accurate staging is critical — intervention during the MCI or Very Mild stage can meaningfully slow progression. Yet access to specialist neuroimaging interpretation remains limited in much of the world.

Current deep learning approaches for Alzheimer's MRI classification predominantly rely on **transfer learning from ImageNet-pretrained weights** — a domain fundamentally different from brain MRI. This project investigates whether architectures trained **entirely from scratch** on neuroimaging data can achieve competitive performance, and specifically whether the CNN inductive bias advantage over pure attention-based models holds at the scale of 80,000 samples.

---

## Problem Statement

Given a 2D axial brain MRI slice, classify the scan into one of four Alzheimer's progression stages:

| Label | Description |
|---|---|
| `Non Demented` | No signs of dementia |
| `Very Mild Dementia` | Earliest detectable stage — clinically the hardest to distinguish |
| `Mild Dementia` | Moderate cognitive decline present |
| `Moderate Dementia` | Significant cognitive and functional impairment |

The **Very Mild / Mild boundary** is the clinically most important and most challenging distinction — misclassification here has direct impact on treatment decisions.

---

## Dataset

**Source:** OASIS (Open Access Series of Imaging Studies) — Marcus et al., 2007, Journal of Cognitive Neuroscience  
**Kaggle mirror:** `ninadaithal/imagesoasis`

| Property | Value |
|---|---|
| Total images | ~86,000 |
| Image type | T1-weighted axial brain MRI slices |
| Resolution | 128 × 128 px |
| Format | JPEG / PNG |
| Split | 70% train / 15% val / 15% test (stratified) |

**Class distribution (pre-split):**

| Class | Count | % |
|---|---|---|
| Non Demented | ~67,200 | 78% |
| Very Mild Dementia | ~13,700 | 16% |
| Mild Dementia | ~5,000 | 6% |
| Moderate Dementia | ~488 | <1% |

Class imbalance was handled via **weighted random sampling** during training — each class receives equal expected representation per batch regardless of dataset frequency.

**Important caveat:** The Kaggle mirror of OASIS contains pre-augmented images. Near-duplicate slices may exist across train/test splits, which inflates test accuracy. Metrics should be interpreted as performance on this dataset configuration, not as a claim of clinical generalisation. Independent validation on held-out institutional data would be required for clinical use.

---

## Architecture Overview

All three architectures were implemented **from scratch in PyTorch** — no pretrained ImageNet weights were used at any stage. This was a deliberate design choice to study the contribution of architectural inductive bias in isolation from pretraining effects.

### ResNet-18

The canonical residual network architecture introduced by He et al. (2016). Consists of **BasicBlocks** — pairs of 3×3 convolutions with identity skip connections — arranged in four stages with progressively increasing channel depth.

```
Input (3×128×128)
  └─ Stem: Conv7×7 → BN → ReLU → MaxPool
  └─ Layer 1: 2× BasicBlock(64→64)
  └─ Layer 2: 2× BasicBlock(64→128, stride=2)
  └─ Layer 3: 2× BasicBlock(128→256, stride=2)
  └─ Layer 4: 2× BasicBlock(256→512, stride=2)
  └─ AdaptiveAvgPool(1×1)
  └─ FC(512→4)
```

| Property | Value |
|---|---|
| Parameters | ~11.2M |
| Activation | ReLU (inplace) |
| Normalisation | BatchNorm2d after every conv |
| Weight init | Kaiming Normal (fan_out) |
| Skip connections | Projection shortcut when spatial dims or channels change |
| Stem modification | Stride reduced for 128×128 input to preserve spatial resolution |

---

### EfficientNet-B0

The compound-scaled architecture introduced by Tan & Le (2019). Uses **Mobile Inverted Bottleneck (MBConv)** blocks with depthwise separable convolutions and **Squeeze-and-Excitation (SE)** channel attention. Stochastic depth (drop-path) is applied for regularisation.

```
Input (3×128×128)
  └─ Stem: Conv3×3(3→32, stride=2) → BN → Swish
  └─ Stage 1: 1× MBConv(expand=1, k=3, out=16)
  └─ Stage 2: 2× MBConv(expand=6, k=3, out=24, stride=2)
  └─ Stage 3: 2× MBConv(expand=6, k=5, out=40, stride=2)
  └─ Stage 4: 3× MBConv(expand=6, k=3, out=80, stride=2)
  └─ Stage 5: 3× MBConv(expand=6, k=5, out=112)
  └─ Stage 6: 4× MBConv(expand=6, k=5, out=192, stride=2)
  └─ Stage 7: 1× MBConv(expand=6, k=3, out=320)
  └─ Head: Conv1×1(320→1280) → BN → Swish
  └─ AdaptiveAvgPool → Dropout(0.2) → FC(1280→4)
```

**MBConv block internals:**
```
x → [Expansion Conv1×1] → DepthwiseConv → SE(ratio=0.25) → Projection Conv1×1 → (+x if residual)
```

| Property | Value |
|---|---|
| Parameters | ~5.3M |
| Activation | Swish (x · sigmoid(x)) |
| Attention | Squeeze-Excitation per block |
| Regularisation | Stochastic depth (linearly scaled, max rate 0.2) |
| Drop-path schedule | Linear from 0 to 0.2 across all blocks |
| Normalisation | BatchNorm2d (momentum=0.01, eps=1e-3) |
| Weight init | Kaiming Normal for conv, zero bias |

---

### Tiny ViT

A pure attention-based Vision Transformer (Dosovitskiy et al., 2020) implemented from scratch. The image is split into non-overlapping 16×16 patches, linearly projected to an embedding dimension of 256, and processed through 6 transformer encoder blocks.

```
Input (3×128×128)
  └─ PatchEmbedding: Conv2d(3→256, kernel=16, stride=16) → (B, 64, 256)
  └─ Prepend [CLS] token → (B, 65, 256)
  └─ Add learnable positional embedding
  └─ 6× TransformerBlock:
       LayerNorm → MultiHeadSelfAttention(heads=8) → residual
       LayerNorm → MLP(256→512→256, GELU) → residual
  └─ LayerNorm
  └─ CLS token → FC(256→4)
```

| Property | Value |
|---|---|
| Parameters | ~6.1M |
| Patch size | 16×16 → 64 patches per image |
| Embedding dim | 256 |
| Transformer depth | 6 blocks |
| Attention heads | 8 (head dim = 32) |
| MLP expansion | 2× (512) |
| Positional embedding | Learnable, truncated normal init (σ=0.02) |
| Dropout | 0.1 throughout |

---

## Training Strategy

All three models were trained with identical hyperparameters and data pipelines to ensure a fair architectural comparison.

### Optimiser & Scheduler

| Component | Choice | Rationale |
|---|---|---|
| Optimiser | AdamW (lr=1e-3, wd=1e-4) | Better weight decay decoupling than Adam |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) | Periodic restarts escape local minima; addresses noisy val loss |
| Gradient clipping | max_norm=1.0 | Stabilises ViT training specifically |
| Early stopping | patience=8 epochs | Prevents overfitting; triggers on val loss plateau |

### Loss Function

**Label Smoothing Cross-Entropy** (ε=0.1):

```
L = (1-ε) · CE(logits, y_hard) + ε · mean(-log_softmax(logits))
```

Label smoothing reduces overconfidence and improves calibration — important for a medical decision-support context where well-calibrated probability estimates matter as much as accuracy.

### Data Augmentation (training only)

```python
T.RandomHorizontalFlip(p=0.5)
T.RandomVerticalFlip(p=0.2)
T.RandomRotation(degrees=15)
T.ColorJitter(brightness=0.3, contrast=0.3)
T.RandomAffine(degrees=0, translate=(0.1, 0.1))
T.ToTensor()
T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
T.RandomErasing(p=0.2, scale=(0.02, 0.1))
```

**Mixup augmentation** (α=0.2) was applied at the batch level during training:
```
x_mixed = λ·xᵢ + (1-λ)·xⱼ,   λ ~ Beta(0.2, 0.2)
L_mixed = λ·L(yᵢ) + (1-λ)·L(yⱼ)
```

### Hardware

- GPU: NVIDIA RTX 4050 (6GB VRAM)
- Batch size: 64
- Training time per model: ~25–35 minutes

---

## Results

### Test Set Performance

| Model | Accuracy | F1 Macro | F1 Weighted | ROC-AUC |
|---|---|---|---|---|
| ResNet-18 | 99.76% | 99.71% | 99.76% | 1.0000 |
| **EfficientNet-B0** | **99.97%** | **99.97%** | **99.97%** | **1.0000** |
| Tiny ViT | 67.07% | 44.22% | 71.62% | 0.8744 |

### Per-Class F1 — EfficientNet-B0 (Deployed Model)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non Demented | 1.00 | 1.00 | 1.00 | 10,084 |
| Very Mild Dementia | 1.00 | 1.00 | 0.999 | 2,060 |
| Mild Dementia | 1.00 | 1.00 | 1.00 | 751 |
| Moderate Dementia | 1.00 | 1.00 | 1.00 | 74 |

### Per-Class F1 — ResNet-18

| Class | F1 |
|---|---|
| Non Demented | 0.9985 |
| Very Mild Dementia | 0.9940 |
| Mild Dementia | 0.9960 |
| Moderate Dementia | 1.0000 |

### Tiny ViT Confusion Matrix Summary

ViT showed significant cross-class confusion — 2,934 of 10,084 Non Demented samples were misclassified, and Moderate Dementia achieved only 0.08 precision (collapsing to predict Moderate for nearly all uncertain samples). This is consistent with class-imbalance collapse behaviour in undertrained attention models.

---

## Key Findings

### 1. CNNs with inductive spatial bias significantly outperform ViT from scratch on this dataset

EfficientNet-B0 outperformed Tiny ViT by **32.9 percentage points in accuracy** and **55.5 points in F1 Macro**. This is consistent with the broader literature:

- The original ViT paper (Dosovitskiy et al., 2020) explicitly noted that ViTs underperform CNNs when trained on datasets smaller than JFT-300M without pretraining
- A 2024 systematic review in *Journal of Medical Systems* (Matsoukas et al.) confirmed that pretraining is critical for ViT performance in medical imaging applications
- A 2025 study in *Soft Computing* (Springer) benchmarking EfficientNet-B0 and lightweight ViTs on the same OASIS dataset found attention-based ensemble methods required significantly more parameters to match CNN performance

The finding that **a ~5M parameter CNN trained from scratch outperforms a ~6M parameter transformer by over 30%** on this task has practical implications for resource-constrained medical AI deployment.

### 2. EfficientNet-B0 achieves higher accuracy but ResNet-18 is better calibrated

EfficientNet's calibration curve shows sharp step-function confidence — the model is either near-certain or near-zero with little probabilistic nuance at intermediate confidence levels. ResNet-18 produces smoother, more graded probability estimates — clinically preferable for a decision-support tool where expressing uncertainty honestly matters.

This echoes findings from Ali et al. (2024, *Frontiers in Cardiovascular Medicine*) that architectures with SE attention tend toward overconfidence relative to simpler residual networks.

### 3. The Very Mild / Mild boundary is the hardest classification pair

Across all three models, Very Mild Dementia consistently produced the lowest per-class metrics. This is clinically expected — the structural MRI differences between Very Mild and Mild Alzheimer's are subtle, and even trained radiologists show inter-rater variability at this boundary. ResNet-18's F1 of 0.994 on this class is the most clinically meaningful individual result in this study.

---

## Explainability: Grad-CAM

Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017) was implemented to visualise which brain regions drove each prediction.

```
weights  = ∂score_c / ∂A^k  averaged over spatial dims
CAM      = ReLU(Σ_k  weights_k · A^k)
```

Where `A^k` are the feature maps of the target convolutional layer and `score_c` is the class logit before softmax.

For EfficientNet-B0, Grad-CAM is applied to the final 1×1 projection conv (`head_conv`). For ResNet-18, to the last conv of `layer4`.

Qualitatively, activation maps for Moderate and Mild Dementia cases show high activation over **central and temporal brain structures** — anatomically consistent with the hippocampal atrophy and ventricular enlargement characteristic of Alzheimer's neurodegeneration. This provides face validity that the model is responding to clinically meaningful structural features rather than image artefacts.

---

## System Architecture

```
┌─────────────────────────────────────────────┐
│          cerevia.netlify.app                │
│         (Plain HTML/CSS/JS)                 │
│  • MRI upload interface                     │
│  • Animated probability bars                │
│  • Grad-CAM side-by-side display           │
└──────────────────┬──────────────────────────┘
                   │  POST /predict (multipart/form-data)
                   │  ← JSON: class, confidence, probs, gradcam_b64
                   ▼
┌─────────────────────────────────────────────┐
│   ved2005-alzheimers-mri-backend.hf.space   │
│         (FastAPI + Uvicorn, Docker)         │
│  • EfficientNet-B0 inference (CPU)          │
│  • Grad-CAM generation                      │
│  • CORS configured for Netlify origin       │
└─────────────────────────────────────────────┘
```

**API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves built-in HTML frontend |
| `POST` | `/predict` | Accepts image, returns prediction + Grad-CAM |
| `GET` | `/health` | Liveness check (used by cron to prevent sleep) |
| `GET` | `/models` | Lists available checkpoints |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

## Project Structure

```
cerevia/
├── config.py                   # All hyperparameters — single source of truth
├── compare.py                  # Post-training model comparison script
├── requirements.txt
│
├── models/
│   ├── __init__.py             # MODEL_REGISTRY + get_model()
│   ├── resnet18.py             # ResNet-18 from scratch
│   ├── efficientnet.py         # EfficientNet-B0 from scratch
│   └── vit_tiny.py             # Tiny ViT from scratch
│
├── training/
│   ├── losses.py               # Label smoothing CE + Mixup
│   ├── evaluate.py             # Metrics, confusion matrix, calibration plots
│   └── train.py                # Training loop (all 3 models or single)
│
├── explainability/
│   └── gradcam.py              # Grad-CAM with forward/backward hooks
│
├── app/
│   ├── main.py                 # FastAPI backend (CORS, lifespan, /predict)
│   ├── static/                 # CSS + JS
│   └── templates/              # Jinja2 HTML (built-in HuggingFace UI)
│
├── alzheimer-frontend/
│   └── index.html              # Standalone Netlify frontend (Cerevia)
│
└── outputs/
    └── reports/                # Training curves, confusion matrices, metrics JSON
```

---

## Local Setup

### Prerequisites

- Python 3.11 or 3.13
- NVIDIA GPU with CUDA 12.1+ (CPU training is possible but slow)
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Installation

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/cerevia.git
cd cerevia

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
kaggle datasets download -d ninadaithal/imagesoasis -p data/raw --unzip
```

### Create Splits

```bash
python -c "from data.dataset import create_splits; create_splits()"
```

### Train

```bash
# Train all 3 models sequentially (~90 min on RTX 4050)
python training/train.py

# Train a single model
python training/train.py --model efficientnet_b0
```

### Compare

```bash
python compare.py
```

### Run App Locally

```bash
python app/main.py
# Open http://localhost:8000
```

---

## Limitations & Honest Caveats

These are stated explicitly because research integrity matters more than inflated claims:

**1. Dataset overlap inflates test metrics**  
The OASIS Kaggle mirror contains pre-augmented images. Near-duplicate slices are likely present across train and test splits. The 99.97% accuracy should be understood as performance on this specific dataset configuration — not as a generalisation claim to unseen clinical data.

**2. No external validation**  
The model has not been validated on any independent institutional dataset (e.g., ADNI, clinical hospital data). External validation is a prerequisite for any clinical use claim.

**3. 2D slice classification is a simplification**  
Clinical Alzheimer's diagnosis uses volumetric 3D MRI, multimodal data (PET, CSF biomarkers, cognitive tests), and longitudinal comparison. A single axial slice is a heavily simplified input.

**4. Class imbalance in rare stages**  
Moderate Dementia has only 74 test samples. Metrics for this class should be interpreted cautiously given the small support.

**5. CPU inference latency**  
The HuggingFace free tier runs on CPU. First inference after cold start takes 15–30 seconds. Not suitable for high-throughput use.

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016.*
2. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML 2019.*
3. Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021.*
4. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017.*
5. Marcus, D. S., et al. (2007). Open access series of imaging studies (OASIS). *Journal of Cognitive Neuroscience, 19(9).*
6. Matsoukas, C., et al. (2024). Comparison of Vision Transformers and CNNs in medical image analysis: A systematic review. *Journal of Medical Systems.*
7. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *CVPR 2018.*
8. Zhang, H., et al. (2017). mixup: Beyond empirical risk minimization. *ICLR 2018.*
9. Huang, G., et al. (2016). Deep networks with stochastic depth. *ECCV 2016.*
10. Müller, R., Kornblith, S., & Hinton, G. (2019). When does label smoothing help? *NeurIPS 2019.*
11. Chakraborty et al. (2025). Novel diagnostic framework with optimized ensemble of ViTs and CNNs for Alzheimer's detection. *Diagnostics, 15(6):789.*
12. Shahid et al. (2025). Novel deep learning for multi-class classification of Alzheimer's using MRI. *Frontiers in Bioinformatics.*

---

<div align="center">

**Cerevia** · Built with PyTorch · FastAPI · Grad-CAM · OASIS Dataset
*For research and educational purposes only · Not for clinical diagnosis*

</div>
