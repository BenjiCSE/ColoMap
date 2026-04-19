"""
evaluate.py
───────────
Loads checkpoints/best_model.pth and measures its performance on the held-out
CRC-VAL-HE-7K test set (data/test/, ~7,180 images).

Produces:
  - A per-class classification report (precision, recall, F1, support)
  - Overall test accuracy
  - A normalised 9×9 confusion matrix saved to outputs/confusion_matrix.png

This script is run exactly once, after training is finished.  It never
updates weights.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# Make `import config` work from any launch directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config


# ── Data ─────────────────────────────────────────────────────────────────────

def build_test_loader() -> tuple[DataLoader, list[str]]:
    """Build the test DataLoader.

    Uses the same deterministic preprocessing as val in train.py: resize to
    224, convert to tensor, normalise with ImageNet stats.  No augmentation
    — evaluation must be repeatable.
    """
    test_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    test_ds = ImageFolder(config.TEST_DIR, transform=test_tf)

    if test_ds.classes != config.CLASSES:
        raise RuntimeError(
            f"Class order mismatch in {config.TEST_DIR}!\n"
            f"  ImageFolder:    {test_ds.classes}\n"
            f"  config.CLASSES: {config.CLASSES}\n"
            f"Fix folder names or config.CLASSES so they match."
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    return test_loader, test_ds.classes


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> nn.Module:
    """Rebuild the ResNet-50 + 9-way head and load trained weights.

    We build the architecture *without* pretrained weights here — the
    checkpoint we're about to load contains the fine-tuned weights that
    replace them.  Downloading ImageNet weights again would just waste
    bandwidth.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Run scripts/train.py first."
        )

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)

    # `map_location` lets a GPU-trained checkpoint load on CPU and vice versa.
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    # Extra safety: verify the class list baked into the checkpoint matches
    # what we think the labels are.  If someone retrained with a different
    # class order, catching it here prevents a silently wrong report.
    ckpt_classes = checkpoint.get("classes")
    if ckpt_classes is not None and ckpt_classes != config.CLASSES:
        raise RuntimeError(
            f"Checkpoint class order doesn't match config.CLASSES!\n"
            f"  Checkpoint: {ckpt_classes}\n"
            f"  Config:     {config.CLASSES}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}   "
          f"Val acc: {checkpoint.get('val_acc', float('nan')):.2f}%")
    return model


# ── Inference loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on every test image and collect predictions.

    Returns two parallel numpy arrays of length N_test:
      y_true[i] = integer index of the correct class for image i
      y_pred[i] = integer index the model predicted for image i
    """
    all_true = []
    all_pred = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(config.DEVICE)

        logits = model(images)
        preds = logits.argmax(dim=1).cpu()

        all_true.append(labels)
        all_pred.append(preds)

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    return y_true, y_pred


# ── Confusion matrix plot ────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, classes, path) -> None:
    """Compute a row-normalised confusion matrix and save it as a heatmap.

    Rows are true labels, columns are predicted labels.  `normalize="true"`
    divides each row by its row sum, so each cell is the fraction of images
    of that true class that were predicted as the column's class.  The
    diagonal is per-class recall.  Support (raw counts per class) varies,
    so row-normalisation makes classes visually comparable.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        colorbar=True,
        xticks_rotation=45,
    )
    ax.set_title("Confusion Matrix (normalised by true class)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Device: {config.DEVICE}")

    test_loader, classes = build_test_loader()
    print(f"Test samples: {len(test_loader.dataset)}   "
          f"Classes: {len(classes)}")
    print("-" * 60)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model = load_model(checkpoint_path)
    print("-" * 60)

    y_true, y_pred = run_inference(model, test_loader)

    # Per-class precision / recall / F1 / support.  digits=4 gives us
    # 4-decimal reporting to match the ~97% target granularity.
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
    )
    accuracy = 100.0 * (y_true == y_pred).mean()

    print("\nClassification Report:")
    print(report)
    print(f"Overall Test Accuracy: {accuracy:.2f}%")

    cm_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    save_confusion_matrix(y_true, y_pred, classes, cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
