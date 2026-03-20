# compare.py
# Loads all trained model checkpoints, evaluates on the test set,
# and prints a clean comparison table + saves a combined bar chart.

import sys
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import CHECKPOINT_DIR, REPORT_DIR, MODELS_TO_TRAIN
from models import get_model
from data.dataset import get_dataloaders
from training.evaluate import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name: str) -> torch.nn.Module | None:
    ckpt_path = CHECKPOINT_DIR / f"{name}_best.pth"
    if not ckpt_path.exists():
        print(f"  SKIP {name}: checkpoint not found at {ckpt_path}")
        return None
    model = get_model(name).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main():
    print("Loading test data ...")
    loaders = get_dataloaders()
    test_loader = loaders["test"]

    results = {}
    for name in MODELS_TO_TRAIN:
        print(f"\nEvaluating {name} ...")
        model = load_model(name)
        if model is None:
            continue
        metrics = evaluate(model, test_loader, DEVICE)
        results[name] = metrics

        # Also read saved JSON in case we want to compare with training runs
        json_path = REPORT_DIR / f"metrics_{name}.json"
        with open(json_path, "w") as f:
            save = {k: v for k, v in metrics.items() if not k.startswith("_")}
            json.dump(save, f, indent=2)

    if not results:
        print("\nNo models evaluated. Train them first: python training/train.py")
        return

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL MODEL COMPARISON")
    print(f"{'='*70}")
    header = f"  {'Model':<22} {'Accuracy':>10} {'F1 Macro':>10} {'AUC':>10}"
    print(header)
    print(f"  {'-'*66}")
    for name, m in results.items():
        print(
            f"  {name:<22} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_macro']:>10.4f} "
            f"{m['roc_auc_macro']:>10.4f}"
        )

    best = max(results, key=lambda n: results[n]["roc_auc_macro"])
    print(f"\n  ✓ Best model by AUC: {best}")
    print(f"    Update SERVE_MODEL in config.py to '{best}' before running the app.\n")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    names   = list(results.keys())
    metrics = ["accuracy", "f1_macro", "roc_auc_macro"]
    labels  = ["Accuracy", "F1 Macro", "ROC-AUC"]
    x       = np.arange(len(names))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = REPORT_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Comparison chart saved → {out}")


if __name__ == "__main__":
    main()
