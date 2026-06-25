"""Convert HEVC/H.265 (or any) clips to H.264 MP4.

Phone "High Efficiency" recordings are HEVC, which many Windows players AND OpenCV
cannot decode. This re-encodes them to H.264 + yuv420p, which plays everywhere and
is readable by cv2.VideoCapture. Uses the ffmpeg binary bundled by imageio-ffmpeg
(no system install needed). Audio is dropped (-an); we only need the video.

    py crossing/convert_to_h264.py --input "C:\\path\\to\\clips_folder"
    py crossing/convert_to_h264.py --input "C:\\path\\to\\one_clip.mov"
    py crossing/convert_to_h264.py --input <path> --output-dir "C:\\path\\to\\out"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".hevc", ".h265", ".mkv", ".avi", ".3gp", ".mts"}


def convert_one(ffmpeg: str, src: Path, out_dir: Path, crf: int) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_h264.mp4"
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an", str(dst),
    ]
    print(f"\nConverting: {src.name} -> {dst.name}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"  FAILED:\n{result.stderr[-800:]}")
        return None
    print(f"  OK ({dst.stat().st_size // 1024} KB)")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert clips to H.264 MP4.")
    parser.add_argument("--input", required=True, help="A video file or a folder of videos.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input>/converted_h264 (or the file's folder).")
    parser.add_argument("--crf", type=int, default=20, help="Quality 18-28 (lower = better/larger).")
    args = parser.parse_args()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return 1

    if in_path.is_dir():
        sources = sorted(p for p in in_path.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        out_dir = Path(args.output_dir) if args.output_dir else in_path / "converted_h264"
    else:
        sources = [in_path]
        out_dir = Path(args.output_dir) if args.output_dir else in_path.parent / "converted_h264"

    if not sources:
        print(f"No video files found in {in_path}")
        return 1

    print(f"ffmpeg: {ffmpeg}")
    print(f"Found {len(sources)} file(s). Output -> {out_dir}")
    done = [convert_one(ffmpeg, s, out_dir, args.crf) for s in sources]
    ok = [d for d in done if d is not None]
    print(f"\nConverted {len(ok)}/{len(sources)} file(s) into: {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
