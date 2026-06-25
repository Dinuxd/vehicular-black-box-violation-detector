#!/usr/bin/env python3
"""Combined Audio + IMU AI + IMU threshold crash detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from detect_crash import CrashDetector
from imu_threshold_detector import detect_threshold_events


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_MODEL_DIR = BASE_DIR / "models" / "audio"
DEFAULT_IMU_AI_MODEL_DIR = BASE_DIR / "models" / "imu_ai"


def run_audio(audio_path: Path, model_dir: Path, threshold: float | None, threads: int) -> dict:
    detector = CrashDetector(model_dir=model_dir, threshold_override=threshold, threads=threads)
    detections, timeline = detector.score_file(audio_path)
    max_score = max((row["score"] for row in timeline), default=0.0)
    return {
        "available": True,
        "detected": bool(detections),
        "threshold": detector.threshold,
        "max_score": float(max_score),
        "detections": detections,
        "windows_scored": len(timeline),
    }


def run_imu_threshold(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    events = detect_threshold_events(df)
    return {
        "available": True,
        "detected": bool(events),
        "event_count": len(events),
        "events": events,
    }


def run_imu_ai(csv_path: Path, model_dir: Path, threshold: float | None) -> dict:
    try:
        from imu_ai_detector import predict_imu_csv

        result = predict_imu_csv(csv_path, model_dir=model_dir, threshold_override=threshold)
        return {
            "available": True,
            "detected": bool(result["detected"]),
            "threshold": result["threshold"],
            "max_probability": result["max_probability"],
            "crash_window_count": result["crash_window_count"],
            "crash_windows": result["crash_windows"],
            "windows_scored": result["windows_scored"],
        }
    except Exception as exc:
        return {
            "available": False,
            "detected": False,
            "error": str(exc),
        }


def fuse(audio: dict | None, imu_ai: dict | None, imu_threshold: dict | None) -> dict:
    audio_hit = bool(audio and audio.get("detected"))
    imu_ai_hit = bool(imu_ai and imu_ai.get("detected"))
    imu_rule_hit = bool(imu_threshold and imu_threshold.get("detected"))

    hit_count = sum([audio_hit, imu_ai_hit, imu_rule_hit])
    if (audio_hit and (imu_ai_hit or imu_rule_hit)) or (imu_ai_hit and imu_rule_hit):
        decision = "CRASH_CONFIRMED"
        confidence = "high"
    elif hit_count == 1:
        decision = "POSSIBLE_CRASH"
        confidence = "medium"
    else:
        decision = "NO_CRASH"
        confidence = "low"

    return {
        "decision": decision,
        "confidence": confidence,
        "audio_detected": audio_hit,
        "imu_ai_detected": imu_ai_hit,
        "imu_threshold_detected": imu_rule_hit,
        "detectors_triggered": hit_count,
    }


def print_summary(result: dict) -> None:
    fusion = result["fusion"]
    print(f"Combined decision: {fusion['decision']} ({fusion['confidence']} confidence)")
    print(
        "Signals: "
        f"audio={fusion['audio_detected']}, "
        f"imu_ai={fusion['imu_ai_detected']}, "
        f"imu_threshold={fusion['imu_threshold_detected']}"
    )

    audio = result.get("audio")
    if audio:
        print(f"Audio: detected={audio['detected']} max_score={audio['max_score']:.4f} windows={audio['windows_scored']}")

    imu_ai = result.get("imu_ai")
    if imu_ai:
        if imu_ai.get("available"):
            print(
                "IMU AI: "
                f"detected={imu_ai['detected']} "
                f"max_probability={imu_ai['max_probability']:.4f} "
                f"crash_windows={imu_ai['crash_window_count']}"
            )
        else:
            print(f"IMU AI: unavailable ({imu_ai.get('error')})")

    imu_threshold = result.get("imu_threshold")
    if imu_threshold:
        print(f"IMU threshold: detected={imu_threshold['detected']} events={imu_threshold['event_count']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run combined crash detection.")
    parser.add_argument("--audio", default=None, help="Optional audio file path.")
    parser.add_argument("--imu-csv", default=None, help="Optional IMU CSV path.")
    parser.add_argument("--audio-model-dir", default=str(DEFAULT_AUDIO_MODEL_DIR), help="Audio model directory.")
    parser.add_argument("--imu-ai-model-dir", default=str(DEFAULT_IMU_AI_MODEL_DIR), help="IMU AI model directory.")
    parser.add_argument("--audio-threshold", type=float, default=None, help="Override audio threshold.")
    parser.add_argument("--imu-ai-threshold", type=float, default=None, help="Override IMU AI threshold.")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for audio PyTorch model.")
    parser.add_argument("--skip-imu-ai", action="store_true", help="Skip TensorFlow/Keras IMU AI detector.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    if not args.audio and not args.imu_csv:
        parser.error("Provide --audio, --imu-csv, or both.")

    result: dict = {"inputs": {"audio": args.audio, "imu_csv": args.imu_csv}}

    audio_result = None
    imu_ai_result = None
    imu_threshold_result = None

    if args.audio:
        audio_result = run_audio(Path(args.audio), Path(args.audio_model_dir), args.audio_threshold, args.threads)
        result["audio"] = audio_result

    if args.imu_csv:
        imu_threshold_result = run_imu_threshold(Path(args.imu_csv))
        result["imu_threshold"] = imu_threshold_result
        if args.skip_imu_ai:
            imu_ai_result = {"available": False, "detected": False, "error": "Skipped by --skip-imu-ai"}
        else:
            imu_ai_result = run_imu_ai(Path(args.imu_csv), Path(args.imu_ai_model_dir), args.imu_ai_threshold)
        result["imu_ai"] = imu_ai_result

    result["fusion"] = fuse(audio_result, imu_ai_result, imu_threshold_result)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_summary(result)
        if args.out:
            print(f"Saved: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
