# training/losses.py
# Label-smoothing cross-entropy loss + Mixup augmentation helper

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_CLASSES, LABEL_SMOOTHING, MIXUP_ALPHA


# ─────────────────────────────────────────────────────────────────────────────
# Label-smoothing cross-entropy
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    Soft targets: (1 - ε) for the true class, ε / (C-1) for others.
    Reduces overconfidence and improves calibration.
    """

    def __init__(
        self,
        smoothing: float = LABEL_SMOOTHING,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing   = smoothing
        self.num_classes = num_classes
        self.confidence  = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,   # (B, C)  raw model output
        targets: torch.Tensor,  # (B,)    integer class indices
    ) -> torch.Tensor:

        log_probs = F.log_softmax(logits, dim=-1)

        # Hard-target NLL
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Uniform smoothing term
        smooth_loss = -log_probs.mean(dim=-1)

        loss = self.confidence * nll + self.smoothing * smooth_loss
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Mixup augmentation
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = MIXUP_ALPHA,
    device: torch.device = torch.device("cpu"),
):
    """
    Apply Mixup to a batch.

    Returns:
        mixed_x  — linearly interpolated input
        y_a, y_b — original and shuffled labels
        lam      — mixing coefficient λ ∈ (0, 1]
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam   = float(np.random.beta(alpha, alpha))
    batch = x.shape[0]
    index = torch.randperm(batch, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixed loss: λ * L(y_a) + (1-λ) * L(y_b)."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
