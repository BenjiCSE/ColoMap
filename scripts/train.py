"""
train.py
────────
Fine-tunes ResNet-50 (ImageNet1K V2 pretrained) to classify 224×224 H&E
histology patches into 9 colorectal tissue classes.

Pipeline:
  1. Build train + val DataLoaders from data/train/ and data/val/.
     Training images get random flips, rotations, and colour jitter; val
     images are only resized + normalised.
  2. Load ResNet-50 with pretrained weights, swap its final 1000-way
     classifier for a new 9-way Linear head, keep every layer trainable.
  3. For NUM_EPOCHS:
        - one full pass over the training set (forward / loss / backprop /
          optimiser step)
        - evaluate on the val set with gradients disabled
        - ReduceLROnPlateau halves the LR if val loss stalls for
          LR_PATIENCE epochs
        - whenever val accuracy hits a new best, save a full checkpoint
          (model weights + optimiser state + metadata) to
          checkpoints/best_model.pth
  4. After the loop, save a train/val loss-and-accuracy plot to
     outputs/training_curves.png.
"""

import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt

# Make `import config` work no matter which folder the script is run from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Seed every RNG that can influence training order or initialisation.

    PyTorch uses three separate RNGs under the hood (Python's `random`,
    NumPy, and its own CPU/GPU RNGs).  Seeding all of them makes the
    shuffle order in DataLoader and the new Linear layer's init weights
    reproducible across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data ─────────────────────────────────────────────────────────────────────

def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, val_transform).

    The train pipeline applies random flips/rotations/colour jitter so each
    epoch sees slightly different versions of every image — that combats
    overfitting on only 400 images/class.  The val pipeline is deterministic:
    just resize to 224 and normalise with the same ImageNet stats ResNet-50
    was originally trained on.
    """
    train_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    return train_tf, val_tf


def build_dataloaders() -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train + val DataLoaders and return them alongside the class list.

    We also verify that ImageFolder's discovered class order matches
    config.CLASSES — if someone ever reorders CLASSES the integer labels in
    the tensors would silently correspond to the wrong tissue names, which
    is the exact bug section 12 of Context.md warns about.
    """
    train_tf, val_tf = build_transforms()

    train_ds = ImageFolder(config.TRAIN_DIR, transform=train_tf)
    val_ds   = ImageFolder(config.VAL_DIR,   transform=val_tf)

    if train_ds.classes != config.CLASSES:
        raise RuntimeError(
            f"Class order mismatch!\n"
            f"  ImageFolder:  {train_ds.classes}\n"
            f"  config.CLASSES: {config.CLASSES}\n"
            f"Fix config.CLASSES so it matches alphabetical folder order."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,                  # reshuffle each epoch
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,                 # val order doesn't matter, keep it stable
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, train_ds.classes


# ── Model ────────────────────────────────────────────────────────────────────

def build_model() -> nn.Module:
    """Load ResNet-50 with ImageNet1K V2 weights and swap the final layer.

    ResNet-50's original `fc` layer maps its 2048-dim feature vector to 1000
    ImageNet classes.  We replace it with a fresh Linear(2048, 9).  Every
    other layer is left as-is and stays trainable — full fine-tuning rather
    than feature extraction.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    return model.to(config.DEVICE)


# ── Train / validate one epoch ───────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, epoch_idx):
    """One full pass over the training set with gradient updates.

    Returns (average_loss, accuracy_percent) across all samples.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx} train", leave=False)
    for images, labels in pbar:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # Standard PyTorch training step:
        #   1. zero any leftover gradients from the previous batch
        #   2. forward pass → logits of shape (batch, 9)
        #   3. cross-entropy loss against integer class labels
        #   4. backprop to populate .grad on every trainable parameter
        #   5. optimiser nudges each parameter in the direction that
        #      reduces the loss
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Track metrics.  loss.item() * batch_size undoes the mean so we
        # can compute a true sample-weighted average at the end.
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, epoch_idx):
    """Run the val set without updating weights.

    `@torch.no_grad()` disables gradient tracking for the whole function —
    saves memory and time since we never call .backward() here.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx} val  ", leave=False)
    for images, labels in pbar:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += batch_size

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── Checkpoint + plot helpers ────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, epoch, val_acc, val_loss, classes):
    """Save a full checkpoint (not just weights).

    Section 12 of Context.md requires that checkpoints include epoch,
    optimiser state, val metrics, and the class list so training can be
    resumed or audited later.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":              val_acc,
        "val_loss":             val_loss,
        "classes":              classes,
    }, path)


def save_training_curves(history, path):
    """Two side-by-side plots: loss over epochs and accuracy over epochs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"],   label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"],   label="val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(config.RANDOM_SEED)

    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}   Epochs: {config.NUM_EPOCHS}")
    print(f"Initial LR: {config.LEARNING_RATE}   Weight decay: {config.WEIGHT_DECAY}")
    print("-" * 60)

    train_loader, val_loader, classes = build_dataloaders()
    print(f"Train samples: {len(train_loader.dataset)}   "
          f"Val samples: {len(val_loader.dataset)}")
    print(f"Classes ({len(classes)}): {classes}")
    print("-" * 60)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",                   # watch val loss; lower is better
        factor=config.LR_FACTOR,      # multiply LR by this when triggered
        patience=config.LR_PATIENCE,  # epochs of no improvement before firing
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}  (lr={current_lr:.6f})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train — Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")
        print(f"  Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

        # ReduceLROnPlateau needs the metric it's watching.  We're watching
        # val loss, so pass val_loss in.  Step happens AFTER the epoch's
        # metrics are logged so the LR printed next epoch is the post-step LR.
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path, model, optimizer,
                epoch=epoch, val_acc=val_acc, val_loss=val_loss,
                classes=classes,
            )
            print(f"  ✓ New best model saved! ({val_acc:.2f}%)")

    total_minutes = (time.time() - start_time) / 60
    print("-" * 60)
    print(f"Training complete in {total_minutes:.1f} min.")
    print(f"Best val accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"Checkpoint: {checkpoint_path}")

    curves_path = os.path.join(config.OUTPUT_DIR, "training_curves.png")
    save_training_curves(history, curves_path)
    print(f"Training curves: {curves_path}")


if __name__ == "__main__":
    main()
