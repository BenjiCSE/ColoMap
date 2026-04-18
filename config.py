# config.py

import torch

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DATA_DIR   = "data/raw/NCT-CRC-HE-100K"
TRAIN_DIR      = "data/train"
VAL_DIR        = "data/val"
TEST_DIR       = "data/test"
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR     = "outputs"

# ── Classes ──────────────────────────────────────────────────────────────────
# Must be alphabetical — matches how ImageFolder assigns indices automatically
CLASSES     = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
NUM_CLASSES = len(CLASSES)   # 9

# ── Dataset ──────────────────────────────────────────────────────────────────
MAX_PER_CLASS = 500          # randomly select this many images per class
VAL_SPLIT     = 0.2          # 20% of selected images go to val/
RANDOM_SEED   = 42

# ── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY  = 1e-4

# ── Image Settings ───────────────────────────────────────────────────────────
IMAGE_SIZE    = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── DataLoader ───────────────────────────────────────────────────────────────
NUM_WORKERS = 0              # CPU training — parallel workers cause overhead
PIN_MEMORY  = False          # GPU only — not applicable

# ── Learning Rate Scheduler ───────────────────────────────────────────────────
LR_PATIENCE = 3              # reduce LR after 3 epochs of no improvement
LR_FACTOR   = 0.5            # multiply LR by 0.5 when triggered
