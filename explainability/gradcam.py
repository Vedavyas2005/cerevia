# explainability/gradcam.py
# Grad-CAM implementation that works with ResNet-18 and EfficientNet-B0.
# For ViT, falls back to attention-weight visualisation.

import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMAGE_SIZE, CLASS_NAMES


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Usage:
        cam    = GradCAM(model)
        heatmap, pred_class, confidence = cam(image_tensor)
        overlay = cam.overlay(original_pil_image, heatmap)
    """

    def __init__(self, model: nn.Module):
        self.model      = model
        self.device     = next(model.parameters()).device
        self._gradients = None
        self._activations = None
        self._hook_handles: list = []
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward and backward hooks to the target layer."""
        try:
            target_layer = self.model.get_gradcam_layer()
        except AttributeError:
            # Fallback: last Conv2d in the network
            target_layer = self._find_last_conv()

        def _save_activation(_, __, output):
            self._activations = output.detach()

        def _save_gradient(_, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hook_handles.append(
            target_layer.register_forward_hook(_save_activation)
        )
        self._hook_handles.append(
            target_layer.register_full_backward_hook(_save_gradient)
        )

    def _find_last_conv(self) -> nn.Module:
        last_conv = None
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d found in model.")
        return last_conv

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

    def __call__(
        self,
        image_tensor: torch.Tensor,   # (1, 3, H, W) already normalised
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Returns:
            heatmap      — (H, W) float32 array in [0, 1]
            pred_class   — predicted class index
            confidence   — softmax probability of predicted class
        """
        self.model.eval()
        image_tensor = image_tensor.to(self.device)

        # Forward
        logits = self.model(image_tensor)
        probs  = F.softmax(logits, dim=-1)

        pred_class = int(probs.argmax(dim=-1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        # Backward on target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Pool gradients over spatial dims (global average pooling)
        # gradients: (1, C, H, W)
        gradients   = self._gradients          # (1, C, H', W')
        activations = self._activations        # (1, C, H', W')

        if gradients is None or activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. "
                "Check that the model has a Conv2d layer."
            )

        weights  = gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam      = (weights * activations).sum(dim=1, keepdim=True)   # (1, 1, H', W')
        cam      = F.relu(cam)

        # Resize to input resolution
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), pred_class, confidence

    # ── Overlay helper ─────────────────────────────────────────────────────────

    @staticmethod
    def overlay(
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Blend a Grad-CAM heatmap onto the original PIL image.

        Args:
            pil_image  — original RGB PIL image (any size)
            heatmap    — (H, W) float array in [0, 1]
            alpha      — opacity of heatmap overlay

        Returns:
            PIL image with heatmap blended in
        """
        img_np = np.array(pil_image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ))

        # Colourmap: jet gives clinically intuitive red=high, blue=low
        heat_uint8 = (heatmap * 255).astype(np.uint8)
        jet        = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        jet_rgb    = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

        blended = (alpha * jet_rgb + (1 - alpha) * img_np).astype(np.uint8)
        return Image.fromarray(blended)
