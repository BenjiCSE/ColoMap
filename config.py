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

# ── WSI Inference (Phase 2) ──────────────────────────────────────────────────
TARGET_MPP                 = 0.5    # μm/pixel of NCT-CRC-HE training patches —
                                    # we must extract WSI patches at this
                                    # resolution so the model sees tissue at
                                    # the magnification it was trained on
WSI_BATCH_SIZE             = 64     # bigger than training batch; no gradients
                                    # means less memory needed per sample
TISSUE_SATURATION_THRESHOLD = 0.07  # HSV saturation cutoff (0-1). H&E stain
                                    # is strongly saturated; glass is near 0
TISSUE_COVERAGE_THRESHOLD  = 0.10   # min fraction of tissue pixels in a patch
                                    # for it to be worth classifying
HEATMAP_ALPHA              = 0.55   # opacity of colour overlay on thumbnail
WSI_THUMBNAIL_MAX_DIM      = 2048   # longest side of the thumbnail used for
                                    # tissue mask + final visualization
