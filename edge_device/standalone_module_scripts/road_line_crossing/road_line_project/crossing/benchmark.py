"""Measure inference / pipeline speed of the crossing module (e.g. on a Raspberry Pi).

Reports how fast the model and the full per-frame pipeline run, so you can see the
real FPS without the video-writing overhead of run_video.py. Does not change the
flow; it just times it.

    python3 benchmark.py                 # synthetic frame, 60 iterations
    python3 benchmark.py --video clip.mp4 --frames 100
    python3 benchmark.py --threads 4     # set CPU inference threads

Run from inside this crossing/ folder.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

try:
    import torch
except ImportError:  # ONNX deployment on the Pi intentionally does not install torch.
    torch = None

import config_crossing as cfg
from infer import RoadLineSegmenter
from mask_postprocess import postprocess_solid
from line_tracker import track_solid_line
from run_video import _apply_source_crop


def _get_frame(video: str | None):
    if video:
        import cv2
        cap = cv2.VideoCapture(video)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return _apply_source_crop(frame, cfg.SOURCE_CROP), "first video frame"
    # synthetic mid-grey frame at the original square phone resolution
    frame = np.full((1440, 1440, 3), 110, dtype=np.uint8)
    return _apply_source_crop(frame, cfg.SOURCE_CROP), "synthetic 1440x1440 frame with source crop"


def _time(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    return dt / n


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark the crossing model / pipeline.")
    ap.add_argument("--video", default=None, help="Optional video; uses its first frame. Else synthetic.")
    ap.add_argument("--frames", type=int, default=60, help="Iterations to time.")
    ap.add_argument("--threads", type=int, default=None, help="CPU inference threads (default: runtime default).")
    args = ap.parse_args()

    if args.threads:
        if torch is not None:
            torch.set_num_threads(args.threads)
        os.environ["CROSSING_ONNX_THREADS"] = str(args.threads)

    import cv2
    frame_bgr, desc = _get_frame(args.video)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    print("Loading model...")
    seg = RoadLineSegmenter()
    seg.crop_top_fraction = cfg.SOURCE_CROP_MODEL_TOP
    if torch is not None:
        thread_desc = f"torch_threads={torch.get_num_threads()}"
    else:
        thread_desc = f"threads={args.threads or 'runtime default'}"
    print(f"  device={seg.device}  input={seg.input_width}x{seg.input_height}  {thread_desc}")
    print(f"  frame={desc}")

    # warmup (first calls include lazy init and are not representative)
    for _ in range(3):
        seg.predict(frame_rgb)

    # model only
    model_s = _time(lambda: seg.predict(frame_rgb), args.frames)

    # full per-frame pipeline: model + opencv cleanup + tracking
    def full():
        pred = seg.predict(frame_rgb)
        post = postprocess_solid(pred["mask"])
        track_solid_line(post["banded_solid"])
    full_s = _time(full, args.frames)

    print("\n--- results (averaged over %d frames) ---" % args.frames)
    print(f"model only      : {model_s * 1000:7.1f} ms/frame  ->  {1.0 / model_s:5.2f} FPS")
    print(f"model + opencv   : {full_s * 1000:7.1f} ms/frame  ->  {1.0 / full_s:5.2f} FPS")
    print(f"opencv overhead  : {(full_s - model_s) * 1000:7.1f} ms/frame")
    print("\nNote: this excludes video reading and annotated-video writing. Real run_video.py")
    print("throughput will be a bit lower because of those, but the model dominates the cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
