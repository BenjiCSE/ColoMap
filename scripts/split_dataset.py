"""
split_dataset.py
────────────────
Takes the raw NCT-CRC-HE-100K dataset and builds the train/ and val/
directories that the training pipeline expects.

For each of the 9 tissue classes:
  1. Find every .tif in data/raw/NCT-CRC-HE-100K/<class>/
  2. Shuffle deterministically with RANDOM_SEED
  3. Keep the first MAX_PER_CLASS (500) images
  4. Copy the first 80% to data/train/<class>/   (400 files)
     Copy the last  20% to data/val/<class>/     (100 files)

Does not touch data/test/.  Safe to re-run: destination folders are cleared
first so counts stay consistent.
"""

import os
import random
import shutil
import sys

# Make the project root importable so `import config` works no matter where
# the script is launched from (repo root, scripts/ folder, etc.).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config


# ── Helpers ──────────────────────────────────────────────────────────────────

def list_tif_images(folder: str) -> list[str]:
    """Return a sorted list of .tif filenames in `folder`.

    Sorting first guarantees the starting order is deterministic across
    machines and filesystems — then we shuffle with a fixed seed on top.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Expected class folder not found: {folder}")

    filenames = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".tif") and not f.startswith(".")
    ]
    filenames.sort()
    return filenames


def clear_and_make(folder: str) -> None:
    """Remove everything inside `folder` and recreate it empty.

    Used on the train/ and val/ class folders before copying, so re-running
    the script doesn't accumulate stale files from a previous run.
    """
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def copy_files(filenames: list[str], src_dir: str, dst_dir: str) -> None:
    """Copy each file in `filenames` from src_dir to dst_dir, preserving metadata."""
    for name in filenames:
        shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Seed Python's RNG once.  Every shuffle below uses this same stream, so
    # the split is 100% reproducible as long as the raw files are unchanged.
    random.seed(config.RANDOM_SEED)

    # Verify the raw dataset is actually in place before doing anything else.
    if not os.path.isdir(config.RAW_DATA_DIR):
        raise FileNotFoundError(
            f"Raw dataset not found at {config.RAW_DATA_DIR}. "
            f"Download NCT-CRC-HE-100K and place it there before running."
        )

    n_train_target = int(config.MAX_PER_CLASS * (1 - config.VAL_SPLIT))  # 400
    n_val_target   = config.MAX_PER_CLASS - n_train_target               # 100

    print(f"Splitting dataset from: {config.RAW_DATA_DIR}")
    print(f"  {config.MAX_PER_CLASS} images/class → "
          f"{n_train_target} train + {n_val_target} val "
          f"(seed={config.RANDOM_SEED})")
    print("-" * 60)

    total_train, total_val = 0, 0

    for cls in config.CLASSES:
        src_dir   = os.path.join(config.RAW_DATA_DIR, cls)
        train_dst = os.path.join(config.TRAIN_DIR, cls)
        val_dst   = os.path.join(config.VAL_DIR, cls)

        # List all .tif images for this class and sanity-check the count.
        all_files = list_tif_images(src_dir)
        if len(all_files) < config.MAX_PER_CLASS:
            raise RuntimeError(
                f"Class {cls} has only {len(all_files)} images in {src_dir}, "
                f"but MAX_PER_CLASS={config.MAX_PER_CLASS} was requested."
            )

        # Deterministic shuffle, then take exactly MAX_PER_CLASS images.
        random.shuffle(all_files)
        selected = all_files[:config.MAX_PER_CLASS]

        # First 400 → train, last 100 → val.
        train_files = selected[:n_train_target]
        val_files   = selected[n_train_target:]

        # Reset destinations so re-running gives a clean split.
        clear_and_make(train_dst)
        clear_and_make(val_dst)

        copy_files(train_files, src_dir, train_dst)
        copy_files(val_files,   src_dir, val_dst)

        total_train += len(train_files)
        total_val   += len(val_files)

        print(f"  {cls:<5}  train: {len(train_files):>4}   val: {len(val_files):>4}")

    print("-" * 60)
    print(f"Done. Total train: {total_train}   Total val: {total_val}")
    print(f"Expected:     train: {n_train_target * config.NUM_CLASSES}   "
          f"val: {n_val_target * config.NUM_CLASSES}")


if __name__ == "__main__":
    main()
