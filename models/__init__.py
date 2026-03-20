# models/__init__.py
from .resnet18      import ResNet18
from .efficientnet  import EfficientNetB0
from .vit_tiny      import ViTTiny

MODEL_REGISTRY = {
    "resnet18":        ResNet18,
    "efficientnet_b0": EfficientNetB0,
    "vit_tiny":        ViTTiny,
}

def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]()
