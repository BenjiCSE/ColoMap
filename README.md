# CRC Tissue Analyser

End-to-end colorectal cancer tissue classification from 224x224 histology
patches up to whole-slide inference with a clinical-style web dashboard.

- **Model:** ResNet-50, ImageNet1K V2 pretrained, fine-tuned on
  NCT-CRC-HE-100K for 9-way colorectal tissue classification.
- **Test accuracy:** 95.49% on CRC-VAL-HE-7K (7,180 held-out images).
- **Val accuracy:** 98.22% (best epoch 14).
- **Pipelines:** training, single-image inference, whole-slide (WSI)
  inference with OpenSlide, and a Flask + vanilla-JS web app.

> This project is for research and educational purposes only. It is **not**
> a certified medical device and must not be used for clinical diagnosis,
> treatment planning, or patient care. All results require verification by
> a qualified pathologist.

---

## Table of contents

1. [Tissue classes](#tissue-classes)
2. [Project structure](#project-structure)
3. [Setup](#setup)
4. [Training pipeline](#training-pipeline)
5. [Evaluation](#evaluation)
6. [Whole-slide inference (CLI)](#whole-slide-inference-cli)
7. [Web app](#web-app)
8. [Model and training details](#model-and-training-details)
9. [Results](#results)

---

## Tissue classes

Nine classes, in the alphabetical order expected by `torchvision.datasets.ImageFolder`:

| Abbrev | Full name             | Description                              |
|--------|-----------------------|------------------------------------------|
| ADI    | Adipose               | Fat tissue                               |
| BACK   | Background            | Empty slide (no tissue)                  |
| DEB    | Debris                | Necrotic material / dead cells           |
| LYM    | Lymphocytes           | Immune cell clusters                     |
| MUC    | Mucus                 | Luminal mucin                            |
| MUS    | Muscularis            | Smooth muscle                            |
| NORM   | Normal mucosa         | Healthy colon epithelium                 |
| STR    | Stroma                | Fibrous connective tissue                |
| TUM    | Tumor epithelium      | Malignant glandular cells (the cancer)   |

---

## Project structure

```
early-cancer-detection/
├── config.py                     single source of truth for hyperparameters
├── requirements.txt
├── Context.md                    design doc / session notes
├── data/
│   ├── raw/NCT-CRC-HE-100K/      ~100k training images (9 class subfolders)
│   ├── train/   (9 x 400 images) populated by split_dataset.py
│   ├── val/     (9 x 100 images) populated by split_dataset.py
│   ├── test/    (CRC-VAL-HE-7K)  held-out evaluation set
│   └── wsi/                      temporary storage for uploaded WSIs
├── checkpoints/best_model.pth    trained model weights + optimiser state
├── outputs/                      training_curves.png, confusion_matrix.png,
│                                 wsi_heatmap_<slide>.png
├── frontend/index.html           single-file web UI (Chart.js via CDN)
└── scripts/
    ├── split_dataset.py          build 80/20 train/val split from raw/
    ├── train.py                  fine-tune ResNet-50 for 30 epochs
    ├── evaluate.py               classification report + confusion matrix
    ├── wsi_inference.py          CLI WSI heatmap generator
    └── app.py                    Flask backend for the web app
```

All scripts import settings from `config.py`; nothing is hardcoded.

---

## Setup

### Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins every dependency. Key packages:

- `torch`, `torchvision` — the model and data pipeline
- `openslide-python` and `openslide-bin` — WSI reading
  (`openslide-bin` bundles the native C library, so no Homebrew is needed)
- `Flask`, `Flask-Cors` — the web backend
- `scikit-learn`, `matplotlib`, `tqdm` — metrics, plots, progress bars

If you see `ModuleNotFoundError` for something just installed, your `pip`
and `python` likely point at different environments — check with
`which pip` and `which python` and use the correct pip
(e.g. `/opt/anaconda3/bin/pip` on macOS with Anaconda).

### Datasets

Both datasets are available from [Zenodo 1214456](https://zenodo.org/record/1214456):

1. **NCT-CRC-HE-100K** (training) — unzip to `data/raw/NCT-CRC-HE-100K/`
   so it contains the 9 class subfolders.
2. **CRC-VAL-HE-7K** (test) — unzip straight into `data/test/` so it
   contains the same 9 class subfolders with ~800 images each.

---

## Training pipeline

Run in this order:

```bash
# 1. Build train/val split (400 train + 100 val per class = 4,500 images)
python scripts/split_dataset.py

# 2. Fine-tune ResNet-50 for 30 epochs
#    Runs on CPU out of the box; uses GPU automatically if available.
python scripts/train.py
```

`train.py` produces:

- `checkpoints/best_model.pth` — best-val-accuracy weights, optimiser state,
  epoch number, val metrics, and the class list (for auditability and
  resumable training).
- `outputs/training_curves.png` — loss and accuracy curves for train + val.

Per-epoch progress is printed like:

```
Epoch 14/30  (lr=0.000100)
  Train - Loss: 0.0531  Acc: 98.22%
  Val   - Loss: 0.0821  Acc: 98.22%
  ✓ New best model saved! (98.22%)
```

---

## Evaluation

```bash
python scripts/evaluate.py
```

Loads `checkpoints/best_model.pth` and runs it over the full CRC-VAL-HE-7K
test set (7,180 images). Prints a per-class classification report, overall
accuracy, and saves `outputs/confusion_matrix.png` (row-normalised so each
cell is per-class recall).

---

## Whole-slide inference (CLI)

The WSI pipeline extracts 224x224 patches from a gigapixel `.svs` (or other
OpenSlide-supported format), classifies each one, and renders a colour-coded
heatmap overlaid on the slide thumbnail.

```bash
python scripts/wsi_inference.py --slide /path/to/slide.svs
```

Options:

| Flag | Default | Purpose |
|------|---------|---------|
| `--slide`              | *(required)* | Path to the WSI (.svs, .ndpi, .tif, .mrxs, ...) |
| `--checkpoint`         | `checkpoints/best_model.pth` | Trained weights |
| `--output-dir`         | `outputs/` | Where the heatmap PNG is saved |
| `--no-tissue-filter`   | off | Classify every grid cell, including glass/background |

How the pipeline works:

1. **Magnification matching** — WSIs can be scanned at 0.25 or 0.5 um/pixel;
   the model was trained at 0.5 um/pixel. The script reads the slide's
   microns-per-pixel metadata and picks the pyramid level + read size that
   produces patches at the training magnification.
2. **Tissue masking** — a low-resolution thumbnail is converted to HSV and
   thresholded on saturation to tell stained tissue from clear glass.
3. **Grid walk** — a non-overlapping grid of 224x224-equivalent patches is
   stepped across the slide; tissue-poor cells are skipped.
4. **Batched inference** — patches are batched (size 64 by default) and run
   through the trained ResNet-50 on the CPU.
5. **Heatmap rendering** — a 3-panel PNG is saved: raw thumbnail,
   multi-class colour overlay with a legend, and a tumour-only view.

Tunable parameters live in `config.py` under "WSI Inference (Phase 2)":
`TARGET_MPP`, `WSI_BATCH_SIZE`, `TISSUE_SATURATION_THRESHOLD`,
`TISSUE_COVERAGE_THRESHOLD`, `HEATMAP_ALPHA`, `WSI_THUMBNAIL_MAX_DIM`.

Output: `outputs/wsi_heatmap_<slidename>.png`.

---

## Web app

A clinical-style dashboard on top of the WSI pipeline. Users upload an SVS
slide, wait a few minutes, and get a heatmap, a tumour-percentage risk
summary, a donut-chart tissue composition, per-class tiles, and an
exportable plain-text report.

### Running it

1. **Start the Flask backend** (must be running before the frontend is used):

   ```bash
   python scripts/app.py
   ```

   Expected startup output:

   ```
   CRC Tissue Analyser — Backend Server
   Loaded checkpoint: checkpoints/best_model.pth
     Epoch: 14   Val acc: 98.22%
   Model checkpoint: checkpoints/best_model.pth  ✓
   Running at: http://localhost:5001
   Ready to accept SVS uploads.
   ```

   > Port 5001 (not 5000) because macOS AirPlay Receiver reserves 5000 on
   > modern macOS. Change the port in both `scripts/app.py` and
   > `frontend/index.html` (the `API_URL` constant) if needed.

2. **Open the frontend**:

   ```bash
   open frontend/index.html
   ```

   Or double-click the file in Finder. CORS is fully open, so the `file://`
   origin works without any local webserver.

### Endpoints

| Method | Route                       | Purpose                                       |
|--------|-----------------------------|-----------------------------------------------|
| POST   | `/analyze`                  | Accept a `.svs` upload; returns analysis JSON |
| GET    | `/outputs/<filename>`       | Serve generated heatmap PNGs                  |
| GET    | `/health`                   | Liveness probe (model loaded, epoch, val acc) |

Response shape from `/analyze`:

```json
{
  "heatmap_url":             "/outputs/wsi_heatmap_slide1.png",
  "patch_counts":            {"ADI": 6, "BACK": 18, "...": "..."},
  "total_patches":           1242,
  "tissue_coverage":         5.3,
  "processing_time_minutes": 1.9,
  "slide_dimensions":        "85656 x 42044 px",
  "best_epoch":              14,
  "val_accuracy":            98.22
}
```

Risk thresholds in the UI (computed client-side from `patch_counts.TUM`):

- `< 5%` TUM — **LOW RISK** (green)
- `5-15%` TUM — **MODERATE RISK** (amber)
- `> 15%` TUM — **HIGH RISK** (red)

---

## Model and training details

| Setting              | Value                                  |
|----------------------|----------------------------------------|
| Architecture         | ResNet-50                              |
| Pretrained weights   | ImageNet1K V2                          |
| Input size           | 224 x 224                              |
| Final layer          | `nn.Linear(2048, 9)` (replaces fc)     |
| All layers trainable | yes (full fine-tuning, not feature-extraction) |
| Loss                 | Cross-entropy                          |
| Optimiser            | Adam, lr=1e-4, weight_decay=1e-4       |
| LR schedule          | ReduceLROnPlateau, factor=0.5, patience=3 |
| Batch size           | 32                                     |
| Epochs               | 30                                     |
| Seed                 | 42                                     |

Augmentations (training only; val and test are deterministic):

- `RandomHorizontalFlip`, `RandomVerticalFlip`
- `RandomRotation(90)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

Checkpoints (`checkpoints/best_model.pth`) save more than weights:
`epoch`, `model_state_dict`, `optimizer_state_dict`, `val_acc`, `val_loss`,
and the class list. This supports resumable training and lets every
downstream script verify that the class order used for training matches
the class order the script expects.

---

## Results

- **Test accuracy (CRC-VAL-HE-7K):** 95.49%
- **Best val accuracy:** 98.22% (epoch 14)
- **Training set:** 3,600 images (400 per class, 500 per class sampled from
  NCT-CRC-HE-100K and split 80/20)

Plots generated by the pipeline:

- `outputs/training_curves.png` — loss and accuracy per epoch, train + val
- `outputs/confusion_matrix.png` — row-normalised 9x9 confusion matrix

---

## Disclaimers

CRC Tissue Analyser is intended for research and educational purposes
only. It is **not** a certified medical device and has not been approved
for clinical diagnosis, treatment planning, or patient care. Risk levels
reported by the web dashboard are based solely on the proportion of
predicted tumour-epithelium patches and must not be interpreted as a
diagnosis. All results must be verified by a qualified pathologist.
