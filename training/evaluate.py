# training/evaluate.py
# Computes accuracy, per-class F1, ROC-AUC, calibration + saves plots.

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLASS_NAMES, REPORT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str = "test",
) -> dict:
    """
    Run model on loader, return dict of metrics.
    """
    model.eval()
    all_logits, all_labels = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)          # (N, C)
    all_labels = torch.cat(all_labels)          # (N,)
    all_probs  = F.softmax(all_logits, dim=-1)  # (N, C)
    all_preds  = all_logits.argmax(dim=-1)      # (N,)

    y_true  = all_labels.numpy()
    y_pred  = all_preds.numpy()
    y_probs = all_probs.numpy()

    acc     = (y_pred == y_true).mean()
    f1_mac  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wtd  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (one-vs-rest, macro)
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    # Per-class F1
    per_class_f1 = f1_score(
        y_true, y_pred, average=None,
        labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES, zero_division=0,
    )

    metrics = {
        "accuracy":        float(acc),
        "f1_macro":        float(f1_mac),
        "f1_weighted":     float(f1_wtd),
        "roc_auc_macro":   float(auc),
        "per_class_f1":    {CLASS_NAMES[i]: float(v)
                            for i, v in enumerate(per_class_f1)},
        "classification_report": report,
        # Keep raw arrays for plots
        "_y_true":  y_true,
        "_y_pred":  y_pred,
        "_y_probs": y_probs,
    }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im  = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)

    plt.tight_layout()
    out = REPORT_DIR / f"confusion_{model_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_training_curves(
    train_losses: list[float],
    val_losses:   list[float],
    train_accs:   list[float],
    val_accs:     list[float],
    model_name:   str,
):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_title(f"Loss — {model_name}")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc")
    ax2.set_title(f"Accuracy — {model_name}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.tight_layout()
    out = REPORT_DIR / f"training_curves_{model_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_calibration(
    y_true:  np.ndarray,
    y_probs: np.ndarray,
    model_name: str,
):
    """
    Reliability diagram: does 80% confidence mean 80% correct?
    Plotted for the positive class of each one-vs-rest problem.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for i, cls_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == i).astype(int)
        prob_true, prob_pred = calibration_curve(
            binary_true, y_probs[:, i], n_bins=10, strategy="uniform"
        )
        ax.plot(prob_pred, prob_true, marker="o", label=cls_name[:10])

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration — {model_name}")
    ax.legend(fontsize=7)
    plt.tight_layout()
    out = REPORT_DIR / f"calibration_{model_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")
