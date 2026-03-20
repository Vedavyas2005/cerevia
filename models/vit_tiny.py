# models/vit_tiny.py
# Tiny Vision Transformer (ViT) implemented from scratch.
# No pretrained weights.

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NUM_CLASSES, IMAGE_SIZE,
    VIT_PATCH_SIZE, VIT_DIM, VIT_DEPTH,
    VIT_HEADS, VIT_MLP_DIM, VIT_DROPOUT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project to VIT_DIM.
    Input:  (B, 3, H, W)
    Output: (B, num_patches, VIT_DIM)
    """

    def __init__(
        self,
        image_size: int  = IMAGE_SIZE,
        patch_size: int  = VIT_PATCH_SIZE,
        in_channels: int = 3,
        embed_dim: int   = VIT_DIM,
    ):
        super().__init__()
        assert image_size % patch_size == 0, \
            f"image_size {image_size} must be divisible by patch_size {patch_size}"

        self.num_patches = (image_size // patch_size) ** 2
        # A single Conv2d with stride=patch_size implements patch extraction + linear projection
        self.projection  = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, num_patches, embed_dim)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.qkv        = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj       = nn.Linear(embed_dim, embed_dim)
        self.attn_drop  = nn.Dropout(dropout)
        self.proj_drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class TransformerBlock(nn.Module):
    """
    ViT Transformer encoder block:
    LayerNorm → MHSA → residual → LayerNorm → MLP → residual
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim:   int,
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Tiny ViT
# ─────────────────────────────────────────────────────────────────────────────

class ViTTiny(nn.Module):
    """
    Tiny Vision Transformer from scratch.

    Config (from config.py):
      image_size   = 128
      patch_size   = 16   →  (128/16)² = 64 patches
      embed_dim    = 256
      depth        = 6    transformer blocks
      num_heads    = 8
      mlp_dim      = 512
      dropout      = 0.1
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.patch_embed = PatchEmbedding()
        num_patches      = self.patch_embed.num_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, VIT_DIM))
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, num_patches + 1, VIT_DIM)
        )
        self.pos_drop    = nn.Dropout(VIT_DROPOUT)

        self.blocks = nn.Sequential(*[
            TransformerBlock(VIT_DIM, VIT_HEADS, VIT_MLP_DIM, VIT_DROPOUT)
            for _ in range(VIT_DEPTH)
        ])

        self.norm = nn.LayerNorm(VIT_DIM)
        self.head = nn.Linear(VIT_DIM, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Positional embeddings: sinusoidal init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding: (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Classification head on CLS token
        return self.head(x[:, 0])

    def get_gradcam_layer(self) -> nn.Module:
        """
        For ViT, Grad-CAM is applied differently (attention rollout is ideal),
        but we expose the last transformer block's MLP norm for compatibility.
        """
        return self.blocks[-1].norm2


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ViTTiny()
    x     = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out   = model(x)
    total = sum(p.numel() for p in model.parameters())
    print(f"ViT-Tiny  |  output: {out.shape}  |  params: {total:,}")
