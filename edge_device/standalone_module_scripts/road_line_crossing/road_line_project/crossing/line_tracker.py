"""Step 3 - measure the solid line's horizontal position in the near-field band.

Reduces the cleaned, banded solid mask to ONE number per frame:

    position in [0.0, 1.0]   (0 = far left of frame, 1 = far right), or None.

If several solid blobs survive in the band (e.g. a left lane line and a right
margin line), we report the one **nearest the ego center**, because that is the
line the vehicle is closest to crossing. Side/margin lines that stay near the
edges are therefore naturally ignored until the car actually drifts toward them.

This number is the input that the Step-4 crossing state machine watches over time.

Run directly for a visual check on a BDD still:

    py crossing/line_tracker.py
    py crossing/line_tracker.py --image path/to/frame.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config_crossing as cfg
from infer import RoadLineSegmenter
from mask_postprocess import postprocess_solid


def side_of(position: float | None) -> str:
    if position is None:
        return "none"
    if position < cfg.HYSTERESIS_LEFT:
        return "left"
    if position > cfg.HYSTERESIS_RIGHT:
        return "right"
    return "center"


def track_solid_line(banded_solid: np.ndarray, ego_center_x: float = cfg.EGO_CENTER_X) -> dict:
    """Return the tracked solid-line position (nearest the ego center) + all blobs.

    position is normalized horizontal location in [0,1], or None if no solid blob
    is present in the band.
    """
    height, width = banded_solid.shape
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(banded_solid, connectivity=8)

    blobs = []
    for label in range(1, num):  # skip background label 0
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx_norm = float(centroids[label, 0]) / width
        cy_norm = float(centroids[label, 1]) / height
        blobs.append({"x": cx_norm, "y": cy_norm, "area": area})

    if not blobs:
        return {"position": None, "side": "none", "n_lines": 0, "blobs": [], "chosen": None}

    chosen = min(blobs, key=lambda b: abs(b["x"] - ego_center_x))
    return {
        "position": chosen["x"],
        "side": side_of(chosen["x"]),
        "n_lines": len(blobs),
        "blobs": blobs,
        "chosen": chosen,
    }


# --- visual debugging ------------------------------------------------------
def make_debug_image(model_input: Image.Image, banded_solid: np.ndarray, band_y_start: int, track: dict) -> Image.Image:
    base = np.asarray(model_input.convert("RGB")).astype(np.float32).copy()
    height, width, _ = base.shape

    # tint solid pixels in band red
    sel = banded_solid > 0
    if sel.any():
        base[sel] = 0.5 * base[sel] + 0.5 * np.asarray(cfg.OVERLAY_COLORS[cfg.CLASS_SOLID], dtype=np.float32)
    out = np.clip(base, 0, 255).astype(np.uint8)

    green, yellow, magenta = (60, 220, 60), (255, 220, 40), (255, 0, 255)

    def vline(x_norm, color, thick=1):
        x = int(round(x_norm * width))
        out[band_y_start:, max(0, x - thick) : x + thick + 1] = color

    out[band_y_start : band_y_start + 2, :] = green          # band top
    vline(cfg.HYSTERESIS_LEFT, yellow)                        # hysteresis zones
    vline(cfg.HYSTERESIS_RIGHT, yellow)
    vline(cfg.EGO_CENTER_X, green)                            # ego center

    if track["position"] is not None:
        vline(track["position"], magenta, thick=1)
        cy = int(round(track["chosen"]["y"] * height))
        cx = int(round(track["position"] * width))
        cv2.circle(out, (cx, cy), 5, magenta, -1, cv2.LINE_AA)

    pos_txt = "none" if track["position"] is None else f"{track['position']:.3f}"
    cv2.putText(out, f"pos={pos_txt}  side={track['side']}  lines={track['n_lines']}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(out)


def _pick_default_image() -> Path:
    images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in cfg.TEST_IMAGE_DIR.glob(ext))
    if not images:
        raise FileNotFoundError(f"No test images found in {cfg.TEST_IMAGE_DIR}")
    return images[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3 visual check: solid line position in near-field band.")
    parser.add_argument("--checkpoint", default=str(cfg.DEFAULT_CHECKPOINT))
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else _pick_default_image()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    segmenter = RoadLineSegmenter(checkpoint=args.checkpoint, device=args.device)
    pred = segmenter.predict(Image.open(image_path))
    post = postprocess_solid(pred["mask"])
    track = track_solid_line(post["banded_solid"])

    print(f"Image: {image_path.name}")
    pos = "none" if track["position"] is None else f"{track['position']:.3f}"
    print(f"  tracked position : {pos}  (side={track['side']}, lines in band={track['n_lines']})")
    for i, b in enumerate(sorted(track["blobs"], key=lambda b: b["x"])):
        print(f"    blob {i}: x={b['x']:.3f} area={b['area']}")

    cfg.DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else cfg.DEBUG_OUTPUT_DIR / f"step3_track_{image_path.stem}.jpg"
    make_debug_image(pred["model_input"], post["banded_solid"], post["band_y_start"], track).save(output_path, quality=92)
    print(f"Saved debug image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
