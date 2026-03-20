# config.py — single source of truth for all hyperparameters
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data" / "raw" /"Data"
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR  = BASE_DIR / "outputs" / "checkpoints"
REPORT_DIR      = BASE_DIR / "outputs" / "reports"

for d in [DATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
KAGGLE_DATASET  = "ninadaithal/imagesoasis"   # kaggle datasets download -d ...
IMAGE_SIZE      = 128                          # native OASIS slice size
NUM_CLASSES     = 4
CLASS_NAMES     = [
    "Non Demented",
    "Very mild Dementia",
    "Mild Dementia",
    "Moderate Dementia",
]

# Train / Val / Test split ratios
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15   # must sum to 1.0

RANDOM_SEED     = 42

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
NUM_EPOCHS      = 50
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1    # CrossEntropy label smoothing factor
MIXUP_ALPHA     = 0.2    # Mixup alpha; set 0.0 to disable
EARLY_STOP_PATIENCE = 8  # stop if val loss doesn't improve for N epochs

# Cosine Annealing Warm Restarts
T_0             = 10     # restart every T_0 epochs
T_MULT          = 2      # period doubles after each restart
ETA_MIN         = 1e-6   # minimum LR

# ── Model ──────────────────────────────────────────────────────────────────────
# Options: "resnet18" | "efficientnet_b0" | "vit_tiny"
MODELS_TO_TRAIN = ["resnet18", "efficientnet_b0", "vit_tiny"]

# Tiny ViT specific
VIT_PATCH_SIZE  = 16
VIT_DIM         = 256
VIT_DEPTH       = 6
VIT_HEADS       = 8
VIT_MLP_DIM     = 512
VIT_DROPOUT     = 0.1

# ── Hardware ───────────────────────────────────────────────────────────────────
NUM_WORKERS     = 0     # DataLoader workers; set 0 on Windows if errors
PIN_MEMORY      = False   # faster GPU transfer

# ── App ────────────────────────────────────────────────────────────────────────
APP_HOST        = "0.0.0.0"
APP_PORT        = 8000
# Which model the app serves (best one after comparison)
SERVE_MODEL     = "efficientnet_b0"
