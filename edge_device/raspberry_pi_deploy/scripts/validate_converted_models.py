from __future__ import annotations

import argparse
import json
from pathlib import Path


DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (DEPLOY_ROOT / p).resolve()


def check_tflite(path: Path) -> str:
    try:
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception:
            from tensorflow.lite.python.interpreter import Interpreter
        interpreter = Interpreter(model_path=str(path), num_threads=1)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        return f"input={inp['shape'].tolist()} {inp['dtype']} output={out['shape'].tolist()} {out['dtype']}"
    except Exception as exc:
        return f"load failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate converted Pi model artifacts.")
    parser.add_argument("--config", default="config/pi_runtime.json")
    args = parser.parse_args()

    with resolve(args.config).open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    paths: list[str] = []
    for section in ("audio", "imu"):
        for model_cfg in cfg.get(section, {}).get("models", {}).values():
            if isinstance(model_cfg, dict) and model_cfg.get("enabled", True):
                if model_cfg.get("path"):
                    paths.append(model_cfg["path"])
    paths.append(cfg["driver_camera"]["model"])
    paths.extend([
        cfg["front_camera"]["detector"],
        cfg["front_camera"]["detector_fallback_onnx"],
        cfg["front_camera"]["classifier"],
    ])
    if cfg["front_camera"].get("classifier_external_data"):
        paths.append(cfg["front_camera"]["classifier_external_data"])

    missing = 0
    for rel in paths:
        path = resolve(rel)
        if not path.exists():
            print(f"MISSING {rel}")
            missing += 1
            continue
        suffix = path.suffix.lower()
        detail = ""
        if suffix == ".tflite":
            detail = " " + check_tflite(path)
        print(f"OK {rel}{detail}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
