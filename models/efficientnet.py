# models/efficientnet.py
# EfficientNet-B0 implemented from scratch in PyTorch.
# No pretrained weights.

import math
import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_CLASSES, IMAGE_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Utility layers
# ─────────────────────────────────────────────────────────────────────────────

def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round v up to the nearest multiple of divisor."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.
    Recalibrates channel-wise feature responses adaptively.
    """

    def __init__(self, in_ch: int, se_ratio: float = 0.25):
        super().__init__()
        squeezed_ch = max(1, int(in_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, squeezed_ch, kernel_size=1, bias=True),
            Swish(),
            nn.Conv2d(squeezed_ch, in_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class ConvBnAct(nn.Sequential):
    """Conv2d → BatchNorm2d → Activation (default: Swish)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        padding = (kernel_size - 1) // 2
        layers  = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
        ]
        if act:
            layers.append(Swish())
        super().__init__(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# MBConv (Mobile Inverted Bottleneck with SE)
# ─────────────────────────────────────────────────────────────────────────────

class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck block with Squeeze-Excitation.
    Used in every stage of EfficientNet.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        self.drop_path_rate = drop_path_rate
        expanded_ch = _make_divisible(in_ch * expand_ratio)

        layers: list[nn.Module] = []

        # Expansion phase (skip when expand_ratio == 1)
        if expand_ratio != 1:
            layers.append(ConvBnAct(in_ch, expanded_ch, kernel_size=1))

        # Depthwise conv
        layers.append(
            ConvBnAct(expanded_ch, expanded_ch,
                      kernel_size=kernel_size, stride=stride,
                      groups=expanded_ch)
        )

        # Squeeze-excitation
        layers.append(SqueezeExcitation(expanded_ch, se_ratio=se_ratio))

        # Projection (no activation)
        layers.append(ConvBnAct(expanded_ch, out_ch, kernel_size=1, act=False))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)

        if self.use_residual:
            if self.training and self.drop_path_rate > 0:
                # Stochastic depth / drop-path
                keep = torch.rand(x.shape[0], 1, 1, 1,
                                  device=x.device) >= self.drop_path_rate
                out  = out * keep.float() / (1 - self.drop_path_rate + 1e-8)
            out = out + x

        return out


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet-B0
# ─────────────────────────────────────────────────────────────────────────────

# (expand_ratio, channels, num_layers, stride, kernel_size)
_B0_STAGES = [
    (1,  16, 1, 1, 3),
    (6,  24, 2, 2, 3),
    (6,  40, 2, 2, 5),
    (6,  80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
]


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 from scratch.
    Stochastic depth (drop-path) applied to MBConv blocks.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()

        # Stem
        self.stem = ConvBnAct(3, 32, kernel_size=3, stride=2)

        # Count total blocks for linearly increasing drop-path rate
        total_blocks = sum(s[2] for s in _B0_STAGES)
        block_idx    = 0

        stages: list[nn.Module] = []
        in_ch = 32

        for expand_ratio, out_ch, num_layers, stride, ks in _B0_STAGES:
            stage_layers: list[nn.Module] = []
            for i in range(num_layers):
                dp = drop_path_rate * block_idx / total_blocks
                stage_layers.append(
                    MBConv(
                        in_ch        = in_ch,
                        out_ch       = out_ch,
                        kernel_size  = ks,
                        stride       = stride if i == 0 else 1,
                        expand_ratio = expand_ratio,
                        drop_path_rate = dp,
                    )
                )
                in_ch      = out_ch
                block_idx += 1
            stages.append(nn.Sequential(*stage_layers))

        self.stages = nn.Sequential(*stages)

        # Head
        self.head_conv = ConvBnAct(in_ch, 1280, kernel_size=1)
        self.avgpool   = nn.AdaptiveAvgPool2d(1)
        self.dropout   = nn.Dropout(p=dropout)
        self.fc        = nn.Linear(1280, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def get_gradcam_layer(self) -> nn.Module:
        """Return last conv layer for Grad-CAM (last stage's projection conv)."""
        last_stage  = self.stages[-1]
        last_mbconv = last_stage[-1]
        # projection conv is the last sub-layer in .block
        return self.head_conv[0]  # ConvBnAct → Conv2d


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EfficientNetB0()
    x     = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out   = model(x)
    total = sum(p.numel() for p in model.parameters())
    print(f"EfficientNet-B0  |  output: {out.shape}  |  params: {total:,}")
