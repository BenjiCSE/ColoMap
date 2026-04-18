# Project Context File — Colorectal Cancer Tissue Classifier
### To be read by Claude Opus at the start of every coding session

---

## IMPORTANT INSTRUCTIONS FOR CLAUDE OPUS

- Read this entire file before writing any code or making any suggestions
- All decisions documented in this file are final — do not suggest alternatives unless explicitly asked
- All scripts must import settings from `config.py` — never hardcode values directly in scripts
- This project is being built by two people who are learning as they go — explain what your code does as you write it
- If something is unclear or a decision needs to be made that isn't covered here, ask before proceeding
- Update the **Current Status** section at the bottom of this file whenever a step is completed

---

## 1. Project Goal

Build a multi-class image classifier using ResNet-50 to classify colorectal histology slide patches into 9 tissue types. The model will be trained on the NCT-CRC-HE-100K dataset and evaluated on the CRC-VAL-HE-7K test set.

The long-term goal beyond this training pipeline is to eventually run the trained model over full Whole Slide Images (WSIs) using OpenSlide and generate tumor heatmaps showing where cancer tissue is detected across the slide. That WSI inference pipeline is a future step — focus only on training for now.

Target performance: approximately 97% accuracy on the CRC-VAL-HE-7K test set.

---

## 2. Background and Context

Colorectal histology slides are stained with H&E (hematoxylin and eosin) and scanned into enormous gigapixel images called Whole Slide Images. Because these images are too large to feed directly into a neural network, they are cut into small fixed-size patches (224×224 pixels). Each patch shows one type of tissue. The model's job is to look at a patch and classify which of the 9 tissue types it belongs to.

ResNet-50 is being used because it is a well-established architecture with pretrained ImageNet weights. Transfer learning means the model already knows how to detect edges, textures, and shapes — we are fine-tuning it to apply those skills to histology patches rather than training from scratch.

---

## 3. The 9 Tissue Classes

These are the exact folder names in the dataset. They are case-sensitive and must be listed in alphabetical order in config.py because PyTorch's ImageFolder assigns class indices alphabetically.

| Abbreviation | Full Name | Description |
|---|---|---|
| ADI | Adipose | Fat tissue — large empty circular vacuoles |
| BACK | Background | Empty slide — no tissue present |
| DEB | Debris | Necrotic material, dead cells |
| LYM | Lymphocytes | Dense clusters of small dark immune cell nuclei |
| MUC | Mucus | Luminal mucin, glandular secretions |
| MUS | Muscularis | Smooth muscle tissue |
| NORM | Normal mucosa | Healthy colon epithelium |
| STR | Stroma | Fibrous connective tissue |
| TUM | Tumor epithelium | Malignant glandular cells — the cancer |

---

## 4. Dataset Details

### Training Dataset — NCT-CRC-HE-100K
- Source: Zenodo — https://zenodo.org/record/1214456
- Full dataset contains ~10,000–12,000 images per class (100,000 total)
- We are only using 500 images per class (4,500 total)
- Images are 224×224 pixels, .tif format, H&E stained
- Downloaded and unzipped to: `data/raw/NCT-CRC-HE-100K/`

### Test Dataset — CRC-VAL-HE-7K
- Source: Same Zenodo page, separate download
- Contains 7,180 images total (~800 per class)
- Prepared by the original dataset creators as a held-out benchmark
- Comes from different patients and scanners than the training data
- Placed directly into: `data/test/`
- Never used during training — only touched during final evaluation

---

## 5. All Locked-In Decisions

Do not suggest changes to any of the following unless explicitly asked:

| Setting | Value | Reason |
|---|---|---|
| Model | ResNet-50 | Established architecture, strong pretrained weights |
| Pretrained weights | ImageNet1K V2 | Best available pretrained weights for ResNet-50 |
| Images per class | 500 | Manageable size for CPU training |
| Train/val split | 80/20 | 400 train, 100 val per class |
| Total training images | 3,600 | 400 × 9 classes |
| Total validation images | 900 | 100 × 9 classes |
| Test set | CRC-VAL-HE-7K | 7,180 images, separate download |
| Image size | 224×224 | ResNet-50 standard input size |
| Batch size | 32 | Standard for this task |
| Epochs | 30 | Sufficient for convergence with pretrained weights |
| Learning rate | 0.0001 | Conservative starting point for fine-tuning |
| Optimizer | Adam | Standard choice |
| Weight decay | 1e-4 | Light regularization |
| Loss function | CrossEntropyLoss | Standard for multi-class classification |
| Device | CPU | No GPU available — training will be slow but correct |
| Random seed | 42 | For reproducibility |
| LR scheduler patience | 3 epochs | Reduce LR after 3 epochs of no improvement |
| LR scheduler factor | 0.5 | Halve the learning rate when triggered |
| num_workers | 0 | CPU training — parallel loading causes overhead |
| pin_memory | False | GPU only feature — not applicable here |

### Normalization Values
These are the exact ImageNet normalization statistics. Required because the model was pretrained on ImageNet:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

### Training Augmentations (applied to train/ images only)
- RandomHorizontalFlip — valid because histology has no natural left/right orientation
- RandomVerticalFlip — valid because histology has no natural up/down orientation
- RandomRotation(90) — tissue can appear at any rotation
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) — simulates stain variation between labs

---

## 6. Project Folder Structure

```
colorectal_cancer_detector/
├── data/
│   ├── raw/
│   │   ├── NCT-CRC-HE-100K/        ← full dataset (9 subfolders, one per class)
│   │   │   ├── ADI/
│   │   │   ├── BACK/
│   │   │   ├── DEB/
│   │   │   ├── LYM/
│   │   │   ├── MUC/
│   │   │   ├── MUS/
│   │   │   ├── NORM/
│   │   │   ├── STR/
│   │   │   └── TUM/
│   │   └── CRC-VAL-HE-7K/          ← test set raw download (can be ignored after copying)
│   ├── train/                       ← 400 images per class after split_dataset.py runs
│   │   ├── ADI/  (400 images)
│   │   ├── BACK/ (400 images)
│   │   ├── DEB/  (400 images)
│   │   ├── LYM/  (400 images)
│   │   ├── MUC/  (400 images)
│   │   ├── MUS/  (400 images)
│   │   ├── NORM/ (400 images)
│   │   ├── STR/  (400 images)
│   │   └── TUM/  (400 images)
│   ├── val/                         ← 100 images per class after split_dataset.py runs
│   │   ├── ADI/  (100 images)
│   │   ├── BACK/ (100 images)
│   │   ├── DEB/  (100 images)
│   │   ├── LYM/  (100 images)
│   │   ├── MUC/  (100 images)
│   │   ├── MUS/  (100 images)
│   │   ├── NORM/ (100 images)
│   │   ├── STR/  (100 images)
│   │   └── TUM/  (100 images)
│   └── test/                        ← CRC-VAL-HE-7K contents placed here (7,180 images total)
│       ├── ADI/
│       ├── BACK/
│       └── ... (same 9 subfolders)
├── checkpoints/                     ← best_model.pth saved here during training
├── outputs/                         ← training_curves.png, confusion_matrix.png
├── scripts/
│   ├── split_dataset.py             ← randomly selects 500/class, splits 80/20 into train/ and val/
│   ├── train.py                     ← main training loop
│   ├── evaluate.py                  ← runs model on test set, produces metrics and confusion matrix
│   └── inference.py                 ← classifies a single image file, for debugging
├── requirements.txt
└── config.py                        ← single source of truth for all settings
```

---

## 7. config.py — Full Contents

Every script imports from this file. Never hardcode values in individual scripts. Variable names must match exactly as written here:

```python
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
```

---

## 8. What Each Script Does

### split_dataset.py
Goes into `data/raw/NCT-CRC-HE-100K/`, reads each of the 9 tissue class folders, randomly shuffles all images in that folder, selects the first 500, then copies 400 into `data/train/<classname>/` and 100 into `data/val/<classname>/`. Run once before training. Does not touch `data/test/`.

### train.py
The main training script. Loads images from `data/train/` and `data/val/` using PyTorch's ImageFolder and DataLoader. Applies augmentation transforms to training images only. Loads ResNet-50 with pretrained ImageNet weights and replaces the final layer (originally 1000 outputs) with a new layer outputting 9 values. Trains for 30 epochs — each epoch does a full forward pass, loss calculation, backpropagation, and weight update across all 3,600 training images in batches of 32. After each epoch, evaluates on all 900 val images without updating weights. Saves the model to `checkpoints/best_model.pth` whenever validation accuracy improves. Saves training curves to `outputs/training_curves.png` after training finishes.

### evaluate.py
Loads `checkpoints/best_model.pth` and runs it against the 7,180 images in `data/test/`. No weight updates. Prints a full classification report (precision, recall, F1 per class) and overall accuracy. Saves a normalized confusion matrix to `outputs/confusion_matrix.png`. Run only once after training is complete.

### inference.py
Loads the saved model and classifies a single image file specified via command line argument. Prints the predicted class and the probability assigned to each of the 9 classes. Used for debugging and spot-checking model behavior.

---

## 9. Key Concepts to Know

**Epoch** — one complete pass through all 3,600 training images. Each epoch consists of ~113 batches of 32 images each.

**Batch** — a group of 32 images processed together. The average loss across the batch is used for one weight update step.

**Overfitting** — when training accuracy keeps climbing but validation accuracy plateaus or drops. Means the model is memorizing training images rather than learning general patterns. Watch for a widening gap between the two numbers.

**Transfer learning** — using ResNet-50's pretrained ImageNet weights as a starting point. The model already knows how to detect visual features; we are teaching it to apply those to histology.

**ImageFolder** — PyTorch utility that reads a folder of subfolders and automatically assigns class labels based on subfolder names. The label for each image is just the name of the folder it lives in.

**DataLoader** — feeds images to the model in batches. Shuffles training data at the start of every epoch so the model never sees images in the same order twice. Does not shuffle validation data.

**Loss** — a number measuring how wrong the model's predictions are. Low loss = confident and correct. High loss = wrong or uncertain. CrossEntropyLoss is used here.

**Backpropagation** — after calculating loss, the error is sent backwards through all 50 layers of the network, computing how much each weight contributed to the mistake.

**Optimizer (Adam)** — uses the gradients from backpropagation to update every weight in the network by a small amount. The learning rate (0.0001) controls how large each step is.

**Learning rate scheduler** — automatically reduces the learning rate by half if validation loss doesn't improve for 3 consecutive epochs.

**Confusion matrix** — a 9×9 grid showing exactly which classes the model confuses with each other. Rows are true labels, columns are predicted labels. The diagonal is correct predictions. Off-diagonal cells are mistakes.

---

## 10. Execution Order

```bash
# Step 1 — install dependencies
pip install -r requirements.txt

# Step 2 — download datasets from https://zenodo.org/record/1214456
#           place NCT-CRC-HE-100K/ inside data/raw/
#           place CRC-VAL-HE-7K/ contents inside data/test/

# Step 3 — split dataset into train/ and val/
python scripts/split_dataset.py

# Step 4 — train the model (will take several hours on CPU)
python scripts/train.py

# Step 5 — evaluate on test set
python scripts/evaluate.py

# Step 6 — test on a single image (optional, for debugging)
python scripts/inference.py --image data/test/TUM/some_image.tif
```

---

## 11. Expected Outputs

### During training (printed each epoch):
```
Epoch 1/30  (lr=0.000100)
  Train — Loss: 1.2847  Acc: 61.32%
  Val   — Loss: 0.9231  Acc: 71.20%
  ✓ New best model saved! (71.20%)
```

### After evaluate.py runs:
```
Classification Report:
              precision    recall  f1-score   support

         ADI       0.99      0.99      0.99       ...
        BACK       1.00      1.00      1.00       ...
         DEB       0.96      0.97      0.96       ...
         LYM       0.98      0.98      0.98       ...
         MUC       0.97      0.96      0.97       ...
         MUS       0.97      0.97      0.97       ...
        NORM       0.96      0.97      0.96       ...
         STR       0.96      0.95      0.96       ...
         TUM       0.97      0.97      0.97       ...

Overall Test Accuracy: 97.xx%
```

### Files produced:
```
checkpoints/best_model.pth       ← model weights from best epoch
outputs/training_curves.png      ← loss and accuracy plotted over 30 epochs
outputs/confusion_matrix.png     ← 9×9 normalized grid of predictions vs true labels
```

---

## 12. Important Technical Notes

- **Class order in CLASSES must be alphabetical** — ImageFolder assigns indices alphabetically, so if CLASSES is in a different order the model will predict the wrong labels
- **pin_memory and num_workers are both set for CPU** — pin_memory=False and num_workers=0. Do not change these
- **The checkpoint saves more than just weights** — it saves epoch number, optimizer state, val accuracy, val loss, and class list. This allows training to be resumed and audited
- **Augmentation is only applied to training images** — val and test images only get resized and normalized, never flipped or color-jittered
- **The model's final layer (model.fc) is replaced** — ResNet-50 originally outputs 1000 values for ImageNet classes. We replace it with a Linear layer outputting 9 values. The in_features for ResNet-50's fc layer is 2048
- **No stain normalization is implemented** — Macenko or Vahadane stain normalization would improve real-world WSI inference but is not included in this pipeline. It can be added later as a preprocessing step
- **All layers are trainable** — the entire network including the pretrained backbone is fine-tuned, not just the final layer

---

## 13. Future Steps (Not Part of Current Work)

These are planned for after the training pipeline is complete. Do not implement these yet:

- WSI inference pipeline using OpenSlide to slice a whole slide image into 224×224 patches and run the model on each one
- Tumor heatmap generation overlaid on the full WSI
- Gemma 4 integration as a reasoning layer on top of flagged tumor patches
- Stain normalization preprocessing for WSI inference

---

## 14. Current Status

**[Update this section at the start and end of every coding session]**

```
Dataset:         [ ] Downloaded NCT-CRC-HE-100K to data/raw/
                 [ ] Downloaded CRC-VAL-HE-7K to data/test/

Scripts:         [ ] requirements.txt written
                 [ ] config.py written
                 [ ] split_dataset.py written
                 [ ] train.py written
                 [ ] evaluate.py written
                 [ ] inference.py written

Execution:       [ ] split_dataset.py successfully run
                       train/ has 400 images × 9 classes = 3,600 total
                       val/   has 100 images × 9 classes =   900 total
                 [ ] train.py successfully run
                       Best val accuracy achieved: _____%
                       Epoch of best model: _____
                 [ ] evaluate.py successfully run
                       Final test accuracy: _____%
                 [ ] Training curves saved to outputs/
                 [ ] Confusion matrix saved to outputs/

Notes from last session:
[Add any errors encountered, decisions made, or things to follow up on]
```
