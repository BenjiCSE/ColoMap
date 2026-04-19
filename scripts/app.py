"""
app.py
──────
Flask backend for the CRC Tissue Analyser web app.

Responsibilities:
  1. Accept an uploaded .svs slide via POST /analyze
  2. Reuse the existing WSI inference pipeline (scripts/wsi_inference.py)
     by importing its building-block functions — no subprocess, no shell.
  3. Return a JSON summary of per-class patch counts, tissue coverage,
     processing time, slide dimensions, and model metadata.
  4. Serve generated heatmap PNGs from outputs/ via GET /outputs/<filename>

Designed to run on localhost:5001 with CORS fully open so the frontend
HTML file can be opened directly in the browser via file:// and still
hit the API.  (Port 5001 instead of 5000 because macOS AirPlay Receiver
already occupies 5000 on modern macOS.)

Dependencies (all already in requirements.txt):
    Flask, Flask-Cors, torch, torchvision, openslide-python, openslide-bin,
    numpy, Pillow

Install with:
    pip install flask flask-cors

Run with:
    python scripts/app.py
"""

import os
import sys
import time
import traceback
from typing import Any

import numpy as np
import openslide
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Path setup — make both `config` (at project root) and `wsi_inference`
# (in this same scripts/ directory) importable regardless of where we launch
# Python from.
# ─────────────────────────────────────────────────────────────────────────────
SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import config  # noqa: E402  (import-after-sys.path tweak is intentional)

# We import the helper functions from wsi_inference.py rather than running
# it as a script.  This keeps the existing CLI behaviour untouched while
# letting the web API reuse the same code paths for tissue masking, patch
# geometry planning, model loading, and heatmap rendering.
from wsi_inference import (  # noqa: E402
    load_model,
    plan_patch_extraction,
    build_thumbnail_and_mask,
    patch_has_tissue,
    render_overlay,
)


# ─────────────────────────────────────────────────────────────────────────────
# Flask app setup
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# CORS wide open so the frontend HTML file works when opened via file:// in
# the browser (no origin header), or served from any other origin during dev.
CORS(app, resources={r"/*": {"origins": "*"}})

# SVS files can be hundreds of MB; the default Flask upload size limit (None
# in dev, small in production) will reject large slides.  Allow up to 4 GB.
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024

# Directory where uploaded slides are saved temporarily.
UPLOAD_DIR  = os.path.join(PROJECT_ROOT, "data", "wsi")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, config.OUTPUT_DIR)
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, config.CHECKPOINT_DIR, "best_model.pth")

os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading at startup.
# We load the ResNet-50 checkpoint exactly once (it's ~100 MB, a second or
# two on CPU) and keep it in memory for the life of the server.  That way
# every /analyze request reuses the same model instance instead of paying
# the load cost on each upload.
# ─────────────────────────────────────────────────────────────────────────────
MODEL: torch.nn.Module | None = None
CHECKPOINT_META: dict[str, Any] = {}

# Preprocessing pipeline — identical to the val/test transform from
# train.py and wsi_inference.py.  Resize to 224, tensor, normalise with
# ImageNet statistics.  Built once and reused for every patch.
PREPROCESS_TF = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])


def init_model_and_meta() -> None:
    """Load the trained ResNet-50 and record its checkpoint metadata.

    Called once at server startup.  Populates the module-level MODEL and
    CHECKPOINT_META globals.  Raises FileNotFoundError if the checkpoint
    is missing — the server can't do anything useful without it.
    """
    global MODEL, CHECKPOINT_META

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            f"Train the model first (scripts/train.py)."
        )

    MODEL = load_model(CHECKPOINT_PATH)

    # Second load just to pull metadata (epoch number, val accuracy).
    # Cheap — the weights are already in memory; this extra load just reads
    # the small header + metadata fields.  Saves us from refactoring
    # load_model to return both model and metadata.
    ckpt = torch.load(CHECKPOINT_PATH, map_location=config.DEVICE)
    CHECKPOINT_META = {
        "epoch":   ckpt.get("epoch"),
        "val_acc": ckpt.get("val_acc"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core inference orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_wsi_analysis(slide_path: str) -> dict[str, Any]:
    """Run the full inference pipeline on a single slide and collect metrics.

    This reproduces the body of wsi_inference.main() but without argparse
    and without printing to stdout as its primary output channel — instead
    we return a dictionary that Flask can serialise to JSON.

    Returns keys: heatmap_url, patch_counts, total_patches,
                  tissue_coverage, processing_time_minutes,
                  slide_dimensions, best_epoch, val_accuracy.

    Raises any underlying exception up to the caller.
    """
    if MODEL is None:
        raise RuntimeError("Model not loaded. init_model_and_meta() must run first.")

    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    t_start = time.time()

    # 1. Open the slide and plan patch geometry at the target MPP.
    slide = openslide.OpenSlide(slide_path)
    geom = plan_patch_extraction(slide)
    width_l0, height_l0 = slide.level_dimensions[0]

    # 2. Thumbnail + tissue mask (HSV saturation threshold).
    thumb, tissue_mask, thumb_scale = build_thumbnail_and_mask(
        slide, config.WSI_THUMBNAIL_MAX_DIM
    )
    tissue_coverage_pct = float(tissue_mask.mean() * 100.0)

    # 3. Grid walk with batched inference.  Skip patches whose thumbnail
    #    region falls below TISSUE_COVERAGE_THRESHOLD.
    n_rows = geom["n_rows"]
    n_cols = geom["n_cols"]
    pred_grid = np.full((n_rows, n_cols), -1, dtype=np.int16)

    batch_tensors: list[torch.Tensor] = []
    batch_coords:  list[tuple[int, int]] = []
    batch_size = config.WSI_BATCH_SIZE
    total_classified = 0

    with torch.no_grad():
        for row in range(n_rows):
            for col in range(n_cols):
                x_l0 = col * geom["step_level0"]
                y_l0 = row * geom["step_level0"]

                if not patch_has_tissue(
                    tissue_mask, thumb_scale, x_l0, y_l0,
                    geom["step_level0"], config.TISSUE_COVERAGE_THRESHOLD,
                ):
                    continue

                patch = slide.read_region(
                    (x_l0, y_l0), geom["level"],
                    (geom["read_size"], geom["read_size"]),
                ).convert("RGB")

                batch_tensors.append(PREPROCESS_TF(patch))
                batch_coords.append((row, col))

                if len(batch_tensors) >= batch_size:
                    batch = torch.stack(batch_tensors).to(config.DEVICE)
                    preds = MODEL(batch).argmax(dim=1).cpu().tolist()
                    for (r, c), p in zip(batch_coords, preds):
                        pred_grid[r, c] = p
                    total_classified += len(preds)
                    batch_tensors.clear()
                    batch_coords.clear()

        # Flush the final partial batch.
        if batch_tensors:
            batch = torch.stack(batch_tensors).to(config.DEVICE)
            preds = MODEL(batch).argmax(dim=1).cpu().tolist()
            for (r, c), p in zip(batch_coords, preds):
                pred_grid[r, c] = p
            total_classified += len(preds)

    # 4. Compute per-class patch counts.
    patch_counts: dict[str, int] = {}
    for idx, cls in enumerate(config.CLASSES):
        patch_counts[cls] = int((pred_grid == idx).sum())

    # 5. Render the heatmap PNG.  render_overlay writes the file to disk and
    #    returns None, so we construct the expected path ourselves (mirrors
    #    wsi_inference's naming convention).
    heatmap_filename = f"wsi_heatmap_{slide_name}.png"
    output_path = os.path.join(OUTPUTS_DIR, heatmap_filename)
    render_overlay(
        thumb, pred_grid, thumb_scale,
        geom["step_level0"], output_path, slide_name,
    )

    slide.close()
    processing_minutes = (time.time() - t_start) / 60.0

    # 6. Assemble the response dict that matches the frontend contract.
    return {
        "heatmap_url":              f"/outputs/{heatmap_filename}",
        "patch_counts":             patch_counts,
        "total_patches":            total_classified,
        "tissue_coverage":          round(tissue_coverage_pct, 2),
        "processing_time_minutes":  round(processing_minutes, 2),
        "slide_dimensions":         f"{width_l0} x {height_l0} px",
        "best_epoch":               int(CHECKPOINT_META.get("epoch") or 0),
        "val_accuracy":             round(
            float(CHECKPOINT_META.get("val_acc") or 0.0), 2
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    """Accept an uploaded SVS slide and return a full analysis summary.

    Expects multipart/form-data with a single field named ``file``.  The
    file extension must be exactly ``.svs`` (case-insensitive) or the
    request is rejected with HTTP 400 and a clear error message.

    On success returns HTTP 200 with a JSON payload containing the
    heatmap URL, per-class patch counts, tissue coverage, processing
    time, slide dimensions, and model metadata.  Any unexpected error
    during inference returns HTTP 500 with a human-readable message.
    """
    # 1. Basic payload validation.
    if "file" not in request.files:
        return (
            jsonify({"error": "No file provided. Upload an SVS file with form field 'file'."}),
            400,
        )

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename. Please select an SVS file."}), 400

    # 2. Extension check — EXACTLY .svs, case-insensitive.  Any other
    #    extension gets the exact error message the spec requires.
    if not uploaded.filename.lower().endswith(".svs"):
        return (
            jsonify({
                "error": "Invalid file type. Please upload an SVS file (.svs). "
                         "Other formats are not currently supported.",
            }),
            400,
        )

    # 3. Save to data/wsi/ with original filename, overwriting any prior
    #    upload of the same name.
    safe_name = os.path.basename(uploaded.filename)
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    try:
        uploaded.save(save_path)
    except Exception as exc:
        return (
            jsonify({"error": f"Failed to save uploaded file: {exc}"}),
            500,
        )

    print(f"\n── New upload ──")
    print(f"  File:   {save_path}")
    print(f"  Size:   {os.path.getsize(save_path) / (1024 * 1024):.1f} MB")

    # 4. Run the pipeline.  Any exception becomes a 500 with a readable
    #    message; the full traceback still goes to server logs for debugging.
    try:
        result = run_wsi_analysis(save_path)
    except Exception as exc:
        traceback.print_exc()
        return (
            jsonify({
                "error": f"Analysis failed: {type(exc).__name__}: {exc}",
            }),
            500,
        )

    print(f"  Done.   {result['total_patches']:,} patches classified "
          f"in {result['processing_time_minutes']:.2f} min")
    print(f"  Heatmap: {result['heatmap_url']}")

    return jsonify(result), 200


@app.route("/outputs/<path:filename>", methods=["GET"])
def serve_output(filename: str):
    """Serve generated heatmap PNGs as static files.

    The frontend receives a URL like ``/outputs/wsi_heatmap_slide1.png``
    from /analyze and renders it as an ``<img src="...">``.  This route
    reads those files from the ``outputs/`` directory on disk.

    404 is returned automatically by Flask if the file doesn't exist.
    """
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    """Small liveness probe the frontend can use to check the server is up."""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "val_accuracy": CHECKPOINT_META.get("val_acc"),
        "best_epoch":   CHECKPOINT_META.get("epoch"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Print startup banner, load the model, then start the Flask dev server."""
    print("CRC Tissue Analyser — Backend Server")
    init_model_and_meta()
    print(f"Model checkpoint: {os.path.relpath(CHECKPOINT_PATH, PROJECT_ROOT)}  \u2713")
    print(f"Running at: http://localhost:5001")
    print(f"Ready to accept SVS uploads.\n")

    # debug=False so the model isn't reloaded twice on Flask's auto-reload.
    # Port 5001 avoids macOS AirPlay Receiver which squats on 5000.
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)


if __name__ == "__main__":
    main()
