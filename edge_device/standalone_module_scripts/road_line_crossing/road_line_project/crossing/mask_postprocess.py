"""Step 2 - clean the model's solid-line mask and keep only the near-field band.

The model output is noisy and fragmented (solid recall ~0.43). Before any geometry
we:
  1. take only the SOLID (class 1) pixels,
  2. morphologically close small gaps so a broken line reads as one line,
  3. drop tiny connected components (specks / mistakes),
  4. keep only the near-field band (bottom slice of the frame), because far away
     all lines converge to the vanishing point and would false-trigger.

OpenCV here is pure geometry/cleanup; it never decides solid vs dashed.

Run directly for a visual check (raw vs cleaned + band) on a BDD still:

    py crossing/mask_postprocess.py
    py crossing/mask_postprocess.py --image path/to/frame.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config_crossing as cfg
from infer import RoadLineSegmenter


def class_binary(mask: np.ndarray, class_id: int) -> np.ndarray:
    """Return a uint8 0/255 binary mask for one class."""
    return np.where(mask == class_id, 255, 0).astype(np.uint8)


def clean_binary(binary: np.ndarray, min_area: int = cfg.MIN_BLOB_AREA, close_kernel: int = cfg.MORPH_CLOSE_KERNEL) -> np.ndarray:
    """Bridge small gaps, then remove connected components smaller than min_area."""
    out = binary
    if close_kernel and close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        keep = np.zeros_like(out)
        for label in range(1, num):  # 0 is background
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == label] = 255
        out = keep
    return out


def near_field_band_start(height: int, band_fraction: float = cfg.NEAR_FIELD_BAND_FRACTION) -> int:
    """Row index where the near-field band begins (rows below this are kept)."""
    band_fraction = float(np.clip(band_fraction, 0.05, 1.0))
    return int(round(height * (1.0 - band_fraction)))


def apply_band(binary: np.ndarray, band_fraction: float = cfg.NEAR_FIELD_BAND_FRACTION) -> tuple[np.ndarray, int]:
    """Zero out everything above the near-field band. Returns (banded, band_y_start)."""
    height = binary.shape[0]
    y_start = near_field_band_start(height, band_fraction)
    banded = binary.copy()
    banded[:y_start, :] = 0
    return banded, y_start


def postprocess_solid(
    mask: np.ndarray,
    min_area: int = cfg.MIN_BLOB_AREA,
    close_kernel: int = cfg.MORPH_CLOSE_KERNEL,
    band_fraction: float = cfg.NEAR_FIELD_BAND_FRACTION,
) -> dict:
    """Full Step-2 pipeline for the solid class.

    Returns raw/cleaned/banded binary masks (uint8 0/255) and the band start row.
    """
    raw = class_binary(mask, cfg.CLASS_SOLID)
    cleaned = clean_binary(raw, min_area=min_area, close_kernel=close_kernel)
    banded, band_y_start = apply_band(cleaned, band_fraction=band_fraction)
    return {
        "raw_solid": raw,
        "cleaned_solid": cleaned,
        "banded_solid": banded,
        "band_y_start": band_y_start,
    }


# --- visual debugging ------------------------------------------------------
def _tint(base_rgb: np.ndarray, binary: np.ndarray, color, alpha: float = cfg.OVERLAY_ALPHA) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    sel = binary > 0
    if sel.any():
        out[sel] = (1.0 - alpha) * out[sel] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_debug_panel(model_input: Image.Image, result: dict) -> Image.Image:
    """Side-by-side: raw solid | cleaned solid | cleaned + near-field band."""
    base = np.asarray(model_input.convert("RGB"))
    solid_color = cfg.OVERLAY_COLORS[cfg.CLASS_SOLID]
    band_color = (60, 220, 60)

    raw_panel = _label(_tint(base, result["raw_solid"], solid_color), "raw solid")
    clean_panel = _label(_tint(base, result["cleaned_solid"], solid_color), "cleaned")

    band_panel = _tint(base, result["banded_solid"], solid_color)
    y = result["band_y_start"]
    band_panel[y : y + 2, :] = band_color  # band boundary line
    cx = int(round(cfg.EGO_CENTER_X * band_panel.shape[1]))
    band_panel[y:, cx : cx + 1] = band_color  # ego-center reference in the band
    band_panel = _label(band_panel, "cleaned + band")

    panel = np.concatenate([raw_panel, clean_panel, band_panel], axis=1)
    return Image.fromarray(panel)


def _pick_default_image() -> Path:
    images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in cfg.TEST_IMAGE_DIR.glob(ext))
    if not images:
        raise FileNotFoundError(f"No test images found in {cfg.TEST_IMAGE_DIR}")
    return images[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2 visual check: clean solid mask + near-field band.")
    parser.add_argument("--checkpoint", default=str(cfg.DEFAULT_CHECKPOINT))
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-area", type=int, default=cfg.MIN_BLOB_AREA)
    parser.add_argument("--close-kernel", type=int, default=cfg.MORPH_CLOSE_KERNEL)
    parser.add_argument("--band-fraction", type=float, default=cfg.NEAR_FIELD_BAND_FRACTION)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else _pick_default_image()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    segmenter = RoadLineSegmenter(checkpoint=args.checkpoint, device=args.device)
    frame = Image.open(image_path)
    pred = segmenter.predict(frame)
    result = postprocess_solid(
        pred["mask"],
        min_area=args.min_area,
        close_kernel=args.close_kernel,
        band_fraction=args.band_fraction,
    )

    print(f"Image: {image_path.name}")
    print(f"  raw solid pixels     : {int((result['raw_solid'] > 0).sum())}")
    print(f"  cleaned solid pixels : {int((result['cleaned_solid'] > 0).sum())}")
    print(f"  solid pixels in band : {int((result['banded_solid'] > 0).sum())}  (band starts at row {result['band_y_start']}/{pred['mask'].shape[0]})")

    cfg.DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else cfg.DEBUG_OUTPUT_DIR / f"step2_panel_{image_path.stem}.jpg"
    make_debug_panel(pred["model_input"], result).save(output_path, quality=92)
    print(f"Saved panel: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
