"""
wsi_inference.py
────────────────
Phase 2: run the trained ResNet-50 over a whole-slide image (WSI) and produce
a colour-coded tissue-type heatmap overlaid on the slide's thumbnail.

Pipeline:
  1. Open the slide with OpenSlide and read its metadata (dimensions,
     microns-per-pixel, pyramid levels).
  2. Decide which pyramid level to read patches from so that each patch's
     physical size matches the 0.5 μm/pixel training resolution.
  3. Build a thumbnail + tissue mask (HSV saturation threshold) so we can
     skip glass/background and only classify real tissue.
  4. Walk a regular grid across the slide.  For each cell that passes the
     tissue filter, read a patch, preprocess it identically to the val
     pipeline from train.py, and batch it for the model.
  5. Run the model in batches, recording predicted class per patch.
  6. Render a 9-colour heatmap, blend it over the thumbnail with a legend,
     and save to outputs/.

Usage:
    python scripts/wsi_inference.py --slide /path/to/slide.svs
    python scripts/wsi_inference.py --slide slide.svs --output-dir outputs
    python scripts/wsi_inference.py --slide slide.svs --no-tissue-filter
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import openslide
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Make `import config` work regardless of launch directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config


# ── Colour palette for the 9 tissue classes ─────────────────────────────────
#
# Order matches config.CLASSES (alphabetical).  TUM (tumour) is a saturated
# red so malignant regions jump out.  BACK is transparent-ish grey since it
# carries no clinical information.
#
CLASS_COLOURS = {
    "ADI":  (1.00, 0.92, 0.55),   # adipose          - pale yellow
    "BACK": (0.80, 0.80, 0.80),   # background       - light grey
    "DEB":  (0.55, 0.35, 0.20),   # debris           - brown
    "LYM":  (0.20, 0.40, 0.85),   # lymphocytes      - blue
    "MUC":  (0.35, 0.80, 0.80),   # mucus            - cyan
    "MUS":  (0.55, 0.30, 0.65),   # muscularis       - purple
    "NORM": (0.35, 0.75, 0.40),   # normal mucosa    - green
    "STR":  (0.95, 0.60, 0.25),   # stroma           - orange
    "TUM":  (0.90, 0.10, 0.15),   # tumour           - red
}


# ── Model loading (mirrors evaluate.py) ─────────────────────────────────────

def load_model(checkpoint_path: str) -> nn.Module:
    """Rebuild architecture and load fine-tuned weights.  No gradient tracking."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run scripts/train.py first."
        )

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)

    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    if ckpt.get("classes") and ckpt["classes"] != config.CLASSES:
        raise RuntimeError(
            f"Checkpoint class order doesn't match config.CLASSES!\n"
            f"  Checkpoint: {ckpt['classes']}\n"
            f"  Config:     {config.CLASSES}"
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}   "
          f"Val acc: {ckpt.get('val_acc', float('nan')):.2f}%")
    return model


# ── Slide geometry ───────────────────────────────────────────────────────────

def plan_patch_extraction(slide: openslide.OpenSlide) -> dict:
    """Work out at which pyramid level to read patches, and how big.

    ResNet-50 was trained on 224x224 patches at ~0.5 μm/pixel.  A WSI can be
    natively 0.25 (40x objective) or 0.5 μm/pixel (20x) or something else.
    We scale accordingly:

      target_downsample = TARGET_MPP / mpp_level0
        - If the slide is 0.25 μm/pixel, target_downsample = 2.0   (use 2x less detail)
        - If the slide is 0.5  μm/pixel, target_downsample = 1.0   (use as-is)

    OpenSlide stores multiple pyramid levels; get_best_level_for_downsample
    picks the highest-resolution level whose downsample is <= target.  Then
    we compute the read size at that level and the step between patches in
    level-0 coordinates (OpenSlide's read_region takes level-0 (x,y)).
    """
    # Microns-per-pixel at level 0.  Fallback to TARGET_MPP if the slide
    # doesn't advertise one, accompanied by a clear warning.
    mpp_prop = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp_prop is None:
        print(f"  ⚠ Slide has no MPP metadata; assuming {config.TARGET_MPP} μm/pixel.")
        mpp_level0 = config.TARGET_MPP
    else:
        mpp_level0 = float(mpp_prop)

    target_downsample = config.TARGET_MPP / mpp_level0
    level = slide.get_best_level_for_downsample(target_downsample)
    level_downsample = slide.level_downsamples[level]

    # Read this many pixels at `level` per patch, then resize to 224.  If
    # target_downsample divides evenly into level_downsample we won't need
    # to resize at all.
    read_size = int(round(config.IMAGE_SIZE * target_downsample / level_downsample))

    # Step between patch origins in level-0 coordinates (non-overlapping).
    step_level0 = int(round(config.IMAGE_SIZE * target_downsample))

    width_l0, height_l0 = slide.level_dimensions[0]
    n_cols = width_l0 // step_level0
    n_rows = height_l0 // step_level0

    print(f"  Level-0 size:        {width_l0} x {height_l0} px")
    print(f"  MPP (level 0):       {mpp_level0:.4f} μm/pixel")
    print(f"  Target MPP:          {config.TARGET_MPP} μm/pixel")
    print(f"  Chosen pyramid lvl:  {level}  (downsample={level_downsample:.2f}x)")
    print(f"  Patch read size:     {read_size}x{read_size} px at level {level}")
    print(f"  Level-0 step:        {step_level0} px")
    print(f"  Grid:                {n_cols} cols x {n_rows} rows "
          f"= {n_cols * n_rows:,} patches")

    return {
        "level":           level,
        "level_downsample": level_downsample,
        "read_size":       read_size,
        "step_level0":     step_level0,
        "n_cols":          n_cols,
        "n_rows":          n_rows,
        "width_l0":        width_l0,
        "height_l0":       height_l0,
    }


# ── Thumbnail + tissue mask ──────────────────────────────────────────────────

def build_thumbnail_and_mask(
    slide: openslide.OpenSlide, max_dim: int
) -> tuple[Image.Image, np.ndarray, float]:
    """Return (thumbnail PIL image, tissue mask, thumb_scale).

    `thumb_scale` is thumbnail-pixels-per-level0-pixel.  With that number we
    can convert any level-0 (x, y) into the matching thumbnail pixel and
    vice versa.

    The mask uses HSV saturation: H&E stain has strong purple/pink hues and
    high saturation; clear glass is low saturation.  Much more robust than
    brightness thresholding, which confuses yellow fat with background.
    """
    # get_thumbnail keeps aspect ratio and makes the longest side == max_dim.
    thumb = slide.get_thumbnail((max_dim, max_dim)).convert("RGB")
    thumb_w, thumb_h = thumb.size

    width_l0 = slide.level_dimensions[0][0]
    thumb_scale = thumb_w / width_l0  # same ratio in both dimensions

    # PIL's HSV mode: S channel is 0-255.
    hsv = np.asarray(thumb.convert("HSV"))
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    tissue_mask = saturation > config.TISSUE_SATURATION_THRESHOLD

    tissue_pct = 100.0 * tissue_mask.mean()
    print(f"  Thumbnail:           {thumb_w} x {thumb_h} px "
          f"(scale={thumb_scale:.5f} thumb/level0)")
    print(f"  Tissue coverage:     {tissue_pct:.1f}% of thumbnail")
    return thumb, tissue_mask, thumb_scale


def patch_has_tissue(
    mask: np.ndarray, thumb_scale: float,
    x_l0: int, y_l0: int, step_l0: int,
    coverage_threshold: float,
) -> bool:
    """True if the thumbnail region matching a level-0 patch is mostly tissue."""
    x_t = int(x_l0 * thumb_scale)
    y_t = int(y_l0 * thumb_scale)
    s_t = max(1, int(step_l0 * thumb_scale))  # at least 1 pixel

    # Clip to thumbnail bounds so the slice doesn't fall off the edge.
    x_end = min(x_t + s_t, mask.shape[1])
    y_end = min(y_t + s_t, mask.shape[0])
    if x_end <= x_t or y_end <= y_t:
        return False

    region = mask[y_t:y_end, x_t:x_end]
    return region.mean() >= coverage_threshold


# ── Heatmap rendering ───────────────────────────────────────────────────────

def render_overlay(
    thumb: Image.Image,
    pred_grid: np.ndarray,         # int16, -1 = no prediction
    thumb_scale: float,
    step_level0: int,
    output_path: str,
    slide_name: str,
) -> None:
    """Blend the per-patch class-colour map over the thumbnail and save.

    The prediction grid (n_rows x n_cols) is upscaled to thumbnail resolution
    using nearest-neighbour (so each patch stays a solid block of colour),
    converted to an RGBA image with transparency where no prediction was
    made, and alpha-blended on top of the thumbnail.  A legend is drawn
    alongside, and a second panel shows just the TUM-probability-like view:
    every non-tumour cell dimmed so tumour hotspots pop out.
    """
    thumb_np = np.asarray(thumb).astype(np.float32) / 255.0
    thumb_h, thumb_w = thumb_np.shape[:2]
    n_rows, n_cols = pred_grid.shape

    # Build an RGBA colour image at grid resolution, then upscale.
    rgba_grid = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
    for idx, cls in enumerate(config.CLASSES):
        r, g, b = CLASS_COLOURS[cls]
        cell_mask = pred_grid == idx
        rgba_grid[cell_mask, 0] = r
        rgba_grid[cell_mask, 1] = g
        rgba_grid[cell_mask, 2] = b
        rgba_grid[cell_mask, 3] = config.HEATMAP_ALPHA
    # Cells that were never classified stay fully transparent (alpha=0).

    # Nearest-neighbour upscale to thumbnail pixel size using PIL.
    rgba_u8 = (rgba_grid * 255).astype(np.uint8)
    rgba_img = Image.fromarray(rgba_u8, mode="RGBA").resize(
        (thumb_w, thumb_h), resample=Image.NEAREST
    )
    overlay = np.asarray(rgba_img).astype(np.float32) / 255.0

    # Alpha composite: out = thumb * (1-alpha) + overlay_rgb * alpha.
    alpha = overlay[..., 3:4]
    blended = thumb_np * (1 - alpha) + overlay[..., :3] * alpha
    blended = np.clip(blended, 0, 1)

    # Tumour-only view: everything that isn't TUM turns transparent.
    tum_idx = config.CLASSES.index("TUM")
    tum_only = np.zeros_like(rgba_grid)
    tum_cells = pred_grid == tum_idx
    r, g, b = CLASS_COLOURS["TUM"]
    tum_only[tum_cells] = [r, g, b, config.HEATMAP_ALPHA]
    tum_u8 = (tum_only * 255).astype(np.uint8)
    tum_img = Image.fromarray(tum_u8, mode="RGBA").resize(
        (thumb_w, thumb_h), resample=Image.NEAREST
    )
    tum_overlay = np.asarray(tum_img).astype(np.float32) / 255.0
    tum_alpha = tum_overlay[..., 3:4]
    tum_blend = thumb_np * (1 - tum_alpha) + tum_overlay[..., :3] * tum_alpha
    tum_blend = np.clip(tum_blend, 0, 1)

    # Legend patches, one per class.
    legend_handles = [
        Patch(facecolor=CLASS_COLOURS[cls], edgecolor="black",
              label=cls, linewidth=0.5)
        for cls in config.CLASSES
    ]

    # Three-panel figure: original thumbnail, multi-class overlay, tumour-only.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(thumb_np)
    axes[0].set_title("Thumbnail")
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title("All tissue types (colour-coded)")
    axes[1].axis("off")
    axes[1].legend(
        handles=legend_handles, loc="center left",
        bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9,
    )

    axes[2].imshow(tum_blend)
    axes[2].set_title("Tumour (TUM) regions only")
    axes[2].axis("off")

    fig.suptitle(f"WSI tissue classification — {slide_name}", fontsize=12)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the trained ResNet-50 over a WSI and render a "
                    "tissue-type heatmap."
    )
    p.add_argument("--slide", required=True, help="Path to the WSI file (.svs, .tif, .ndpi, ...)")
    p.add_argument("--checkpoint", default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
                   help="Path to trained checkpoint (default: checkpoints/best_model.pth)")
    p.add_argument("--output-dir", default=config.OUTPUT_DIR,
                   help="Where to save the heatmap image (default: outputs/)")
    p.add_argument("--no-tissue-filter", action="store_true",
                   help="Classify every grid cell, including glass/background. "
                        "Much slower; useful for debugging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.slide):
        raise FileNotFoundError(f"Slide not found: {args.slide}")

    slide_name = os.path.splitext(os.path.basename(args.slide))[0]
    print(f"Slide: {args.slide}")
    print(f"Device: {config.DEVICE}")
    print("-" * 60)

    # 1. Open slide and plan geometry.
    slide = openslide.OpenSlide(args.slide)
    geom = plan_patch_extraction(slide)
    print("-" * 60)

    # 2. Thumbnail + tissue mask.
    thumb, tissue_mask, thumb_scale = build_thumbnail_and_mask(
        slide, config.WSI_THUMBNAIL_MAX_DIM
    )
    print("-" * 60)

    # 3. Load model + build preprocessing transform (same as val/test).
    model = load_model(args.checkpoint)
    preprocess_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])
    print("-" * 60)

    # 4. Walk the grid, read patches, batch, classify.
    #
    # pred_grid stores the predicted class index per cell; -1 means "never
    # classified" (skipped by the tissue filter) and shows as transparent.
    pred_grid = np.full((geom["n_rows"], geom["n_cols"]), -1, dtype=np.int16)

    batch_tensors: list[torch.Tensor] = []
    batch_coords: list[tuple[int, int]] = []   # (row, col) for each tensor
    batch_size = config.WSI_BATCH_SIZE
    total_cells = geom["n_rows"] * geom["n_cols"]

    start_time = time.time()
    total_classified = 0
    skipped = 0

    # Flatten the 2D grid into a single tqdm-able sequence.
    grid_iter = (
        (row, col)
        for row in range(geom["n_rows"])
        for col in range(geom["n_cols"])
    )

    with torch.no_grad():
        for row, col in tqdm(grid_iter, total=total_cells, desc="Classifying"):
            x_l0 = col * geom["step_level0"]
            y_l0 = row * geom["step_level0"]

            if not args.no_tissue_filter and not patch_has_tissue(
                tissue_mask, thumb_scale, x_l0, y_l0, geom["step_level0"],
                config.TISSUE_COVERAGE_THRESHOLD,
            ):
                skipped += 1
                continue

            # read_region returns RGBA; convert to RGB for the model.
            patch = slide.read_region(
                (x_l0, y_l0), geom["level"],
                (geom["read_size"], geom["read_size"]),
            ).convert("RGB")

            batch_tensors.append(preprocess_tf(patch))
            batch_coords.append((row, col))

            if len(batch_tensors) >= batch_size:
                batch = torch.stack(batch_tensors).to(config.DEVICE)
                preds = model(batch).argmax(dim=1).cpu().tolist()
                for (r, c), p in zip(batch_coords, preds):
                    pred_grid[r, c] = p
                total_classified += len(preds)
                batch_tensors.clear()
                batch_coords.clear()

        # Flush the final partial batch.
        if batch_tensors:
            batch = torch.stack(batch_tensors).to(config.DEVICE)
            preds = model(batch).argmax(dim=1).cpu().tolist()
            for (r, c), p in zip(batch_coords, preds):
                pred_grid[r, c] = p
            total_classified += len(preds)

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Classified:   {total_classified:,} patches")
    print(f"Skipped:      {skipped:,} (below tissue threshold)")
    print(f"Time:         {elapsed/60:.1f} min "
          f"({(total_classified / max(elapsed, 1)):.1f} patches/sec)")

    # 5. Class-count summary.
    print("\nPer-class patch counts:")
    for idx, cls in enumerate(config.CLASSES):
        count = int((pred_grid == idx).sum())
        pct = 100.0 * count / max(total_classified, 1)
        print(f"  {cls:<5} {count:>7,}  ({pct:5.2f}%)")

    # 6. Render + save.
    output_path = os.path.join(args.output_dir, f"wsi_heatmap_{slide_name}.png")
    render_overlay(
        thumb, pred_grid, thumb_scale,
        geom["step_level0"], output_path, slide_name,
    )
    print(f"\nHeatmap saved to: {output_path}")

    slide.close()


if __name__ == "__main__":
    main()
