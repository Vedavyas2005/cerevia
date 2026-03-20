# training/train.py
# Trains all models defined in config.MODELS_TO_TRAIN sequentially.
# Each model gets identical data splits and augmentation — clean comparison.
#
# Usage:
#   python training/train.py
#   python training/train.py --model resnet18   # train one model only

import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHECKPOINT_DIR, REPORT_DIR,
    MODELS_TO_TRAIN, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY,
    T_0, T_MULT, ETA_MIN,
    EARLY_STOP_PATIENCE, MIXUP_ALPHA,
    NUM_CLASSES,
)
from models       import get_model
from data.dataset import get_dataloaders
from training.losses   import LabelSmoothingCrossEntropy, mixup_data, mixup_criterion
from training.evaluate import (
    evaluate,
    plot_confusion_matrix,
    plot_training_curves,
    plot_calibration,
)


# ─────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_mixup: bool = True,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_mixup and MIXUP_ALPHA > 0:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, device=device)
            logits = model(imgs)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping — helps ViT stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == y_a if (use_mixup and MIXUP_ALPHA > 0) else preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Train one model end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model_name: str, loaders: dict) -> dict:
    """
    Full training pipeline for a single model.
    Returns final test metrics dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}  |  device: {device}")
    print(f"{'='*60}")

    model     = get_model(model_name).to(device)
    criterion = LabelSmoothingCrossEntropy()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT, eta_min=ETA_MIN
    )

    best_val_loss  = float("inf")
    patience_count = 0
    ckpt_path      = CHECKPOINT_DIR / f"{model_name}_best.pth"

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        vl_loss, vl_acc = validate(
            model, loaders["val"], criterion, device
        )
        scheduler.step(epoch)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{NUM_EPOCHS}  "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  |  "
            f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}  "
            f"[{elapsed:.1f}s]"
        )

        # ── Checkpoint & early stopping ───────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save({
                "epoch":      epoch,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "val_loss":   vl_loss,
                "val_acc":    vl_acc,
            }, ckpt_path)
            print(f"    ✓ Best model saved → {ckpt_path.name}")
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # ── Reload best weights for test evaluation ───────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\n  Best checkpoint: epoch {ckpt['epoch']}  "
          f"val_loss: {ckpt['val_loss']:.4f}  val_acc: {ckpt['val_acc']:.4f}")

    # ── Test metrics ──────────────────────────────────────────────────────
    print("  Evaluating on test set ...")
    metrics = evaluate(model, loaders["test"], device, split="test")

    print(f"\n  Test Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Macro      : {metrics['f1_macro']:.4f}")
    print(f"  ROC-AUC Macro : {metrics['roc_auc_macro']:.4f}")
    print(f"\n{metrics['classification_report']}")

    # ── Save plots ────────────────────────────────────────────────────────
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name)
    plot_confusion_matrix(metrics["_y_true"], metrics["_y_pred"], model_name)
    plot_calibration(metrics["_y_true"], metrics["_y_probs"], model_name)

    # Save metrics JSON (exclude raw arrays)
    save_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    json_path = REPORT_DIR / f"metrics_{model_name}.json"
    with open(json_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"  Metrics saved → {json_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default=None,
        help="Train a specific model. Options: resnet18, efficientnet_b0, vit_tiny"
    )
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODELS_TO_TRAIN

    print("Loading data ...")
    loaders = get_dataloaders()

    all_results = {}
    for name in models_to_run:
        all_results[name] = train_model(name, loaders)

    # ── Final comparison summary ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'F1 Macro':>10} {'AUC':>10}")
    print(f"  {'-'*52}")
    for name, m in all_results.items():
        print(
            f"  {name:<20} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_macro']:>10.4f} "
            f"{m['roc_auc_macro']:>10.4f}"
        )


if __name__ == "__main__":
    main()
