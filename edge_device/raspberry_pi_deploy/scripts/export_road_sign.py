from __future__ import annotations

import json
import shutil
from pathlib import Path


DEPLOY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = DEPLOY_ROOT / "config" / "model_manifest.json"


def resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (DEPLOY_ROOT / p).resolve()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"SKIP missing source: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src} -> {dst}")


def main() -> int:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    detector = manifest["models"]["road_sign_detector"]
    source = resolve(detector["source"])
    ncnn_out = resolve(detector["deploy"])
    onnx_out = resolve(detector["fallback_deploy"])

    if not source.exists():
        print(f"missing detector source: {source}")
        return 1

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"ultralytics is required for road-sign export: {exc}")
        return 1

    model = YOLO(str(source))
    print("exporting road-sign detector to NCNN")
    ncnn_path = Path(model.export(format="ncnn", imgsz=640, half=True))
    if ncnn_out.exists() and ncnn_out.is_dir():
        shutil.rmtree(ncnn_out)
    ncnn_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ncnn_path, ncnn_out)
    print(f"exported NCNN -> {ncnn_out}")

    print("exporting road-sign detector ONNX fallback")
    onnx_path = Path(model.export(format="onnx", imgsz=640, simplify=True, opset=12))
    copy_file(onnx_path, onnx_out)

    classifier = manifest["models"]["road_sign_classifier"]
    copy_file(resolve(classifier["source"]), resolve(classifier["deploy"]))
    if classifier.get("data_source") and classifier.get("data_deploy"):
        copy_file(resolve(classifier["data_source"]), resolve(classifier["data_deploy"]))
    copy_file(resolve(classifier["summary"]), resolve("models/road_sign/classifier_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
