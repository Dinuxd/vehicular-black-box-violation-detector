#!/usr/bin/env python3
"""Download the MediaPipe Face Landmarker task bundle."""

from __future__ import annotations

from pathlib import Path
import urllib.request


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path("models/face_landmarker.task")


def main() -> int:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"Model already exists: {MODEL_PATH}")
        return 0
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
