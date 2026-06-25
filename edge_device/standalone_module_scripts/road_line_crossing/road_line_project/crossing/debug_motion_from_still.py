"""Step 0 - fake a dashcam video from a single still, to smoke-test the pipeline.

A still photo does not move, but our crossing logic needs motion. So we slide the
whole image sideways (a constant-scale translation), frame by frame, which makes
any line in the photo drift across the frame - just like a car gradually crossing
it. Translation (not crop-zoom) is used so the line's scale stays constant and the
model keeps detecting it.

This proves ONLY that the whole pipeline runs end to end and fires an event. It
says NOTHING about real-world accuracy (the motion is artificial). Replace it with
real footage (run_video.py --video ...) for actual validation.

    py crossing/debug_motion_from_still.py
    py crossing/debug_motion_from_still.py --image path/to/frame.jpg --direction rtl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import config_crossing as cfg
from run_video import process_frames


def pan_frames(image_bgr: np.ndarray, num_frames: int, max_shift_fraction: float, direction: str):
    """Yield horizontally translated copies of the image to fake lateral drift.

    'ltr' sweeps content left-to-right (shift 0 -> +max); 'rtl' is the reverse.
    The exposed edge is filled by replicating the border (smear), which is fine
    for a smoke test - the line region itself stays clean.
    """
    h, w = image_bgr.shape[:2]
    max_shift = int(round(max_shift_fraction * w))
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)          # 0 -> 1
        if direction == "rtl":
            t = 1.0 - t                          # play the sweep backwards
        shift = int(round(t * max_shift))
        matrix = np.float32([[1, 0, shift], [0, 1, 0]])
        yield cv2.warpAffine(image_bgr, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _pick_default_image() -> Path:
    # A daytime solid-line test image pans nicely.
    preferred = cfg.TEST_IMAGE_DIR / "001585_train_0168e22b-7f034886.jpg"
    if preferred.is_file():
        return preferred
    images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in cfg.TEST_IMAGE_DIR.glob(ext))
    if not images:
        raise FileNotFoundError(f"No test images found in {cfg.TEST_IMAGE_DIR}")
    return images[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 0: fake-motion smoke test of the crossing pipeline.")
    parser.add_argument("--image", default=None)
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--max-shift-fraction", type=float, default=0.45, help="Max horizontal translation as a fraction of width.")
    parser.add_argument("--direction", choices=("ltr", "rtl"), default="ltr")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else _pick_default_image()
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    output_dir = Path(args.output_dir) if args.output_dir else cfg.DEBUG_OUTPUT_DIR / "run_fake_motion"
    print(f"Fake-motion source: {image_path.name}  frames={args.frames}  direction={args.direction}")

    frames = list(pan_frames(image_bgr, args.frames, args.max_shift_fraction, args.direction))
    # BDD stills are already wide and the model should use its own top-crop, so opt
    # out of the square-clip source crop.
    process_frames(frames, args.fps, output_dir, source_name=f"fake_motion[{image_path.name}]",
                   source_crop=None, model_crop_top=None)
    print("\nNote: artificial motion - smoke test only, not a real-accuracy result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
