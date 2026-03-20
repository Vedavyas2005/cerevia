# models/resnet18.py
# ResNet-18 implemented from scratch in PyTorch.
# No pretrained weights — all parameters initialised randomly.

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_CLASSES, IMAGE_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    ResNet BasicBlock: two 3×3 conv layers with a skip connection.
    Used in ResNet-18 and ResNet-34.
    """
    expansion = 1   # output channels == planes (no bottleneck)

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2   = nn.BatchNorm2d(planes)

        # Shortcut: project if spatial size or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18(nn.Module):
    """
    ResNet-18 from scratch.
    Architecture: [2, 2, 2, 2] BasicBlocks with channels [64, 128, 256, 512].
    Adapted for 128×128 input (stem stride reduced from 4 to 2).
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.in_planes = 64

        # Stem — reduced stride for 128×128 inputs
        # (original ResNet uses stride=2 conv + stride=2 maxpool = /4 total)
        # We keep maxpool but use a single stride-1 conv to preserve spatial res
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    # ── Grad-CAM hook helpers ─────────────────────────────────────────────────
    def get_gradcam_layer(self) -> nn.Module:
        """Return the last conv layer used for Grad-CAM."""
        return self.layer4[-1].conv2


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ResNet18()
    x     = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out   = model(x)
    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet-18  |  output: {out.shape}  |  params: {total:,}")
