"""Single-frame inference wrapper for the frozen road-line model.

Wave 1 change: supports TWO backends with the SAME input -> output contract.
  * "onnx"  : runs best_model.onnx via onnxruntime  (fast on ARM; NO torch needed)
  * "torch" : runs best_model.pth via PyTorch        (original path, unchanged)

Selection (in order): explicit backend= arg, then env var CROSSING_BACKEND, then
"auto" (use ONNX if best_model.onnx exists and onnxruntime is importable, else torch).

Preprocessing and the returned mask/probabilities are identical for both backends,
so every downstream module (mask_postprocess, line_tracker, crossing_logic, ...) is
unaffected by the choice. Geometry (input size, crop, classes) is unchanged.

    py infer.py --image frame.jpg                 # auto backend
    CROSSING_BACKEND=torch py infer.py --image x   # force PyTorch
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import config_crossing as cfg


def _softmax_np(arr: np.ndarray, axis: int) -> np.ndarray:
    m = arr.max(axis=axis, keepdims=True)
    e = np.exp(arr - m)
    return e / e.sum(axis=axis, keepdims=True)


def _resolve_backend(requested: str | None, onnx_path: Path) -> str:
    backend = (requested or os.environ.get("CROSSING_BACKEND") or "auto").lower()
    if backend in ("onnx", "torch"):
        return backend
    if backend != "auto":
        raise ValueError("Backend must be one of: auto, onnx, torch")
    # auto: prefer ONNX if available
    if onnx_path.is_file():
        try:
            import onnxruntime  # noqa: F401
            return "onnx"
        except Exception as onnx_exc:
            try:
                import torch  # noqa: F401
                return "torch"
            except Exception:
                raise RuntimeError(
                    f"ONNX model found at {onnx_path}, but onnxruntime is not installed. "
                    "Install the Pi requirements with: python -m pip install -r requirements_pi.txt"
                ) from onnx_exc
    return "torch"


class RoadLineSegmenter:
    """frame (RGB) -> class-id mask + per-class probabilities, via ONNX or PyTorch."""

    def __init__(
        self,
        checkpoint=cfg.DEFAULT_CHECKPOINT,
        onnx_path=None,
        backend=None,
        device="auto",
        onnx_threads: int | None = None,
    ) -> None:
        checkpoint = Path(checkpoint)
        self.checkpoint_path = checkpoint
        self.onnx_path = Path(onnx_path) if onnx_path else checkpoint.with_suffix(".onnx")
        self.backend = _resolve_backend(backend, self.onnx_path)
        self.onnx_threads = onnx_threads

        # geometry defaults (match how the model was trained / exported)
        self.input_width = cfg.INPUT_WIDTH
        self.input_height = cfg.INPUT_HEIGHT
        self.crop_top_fraction = cfg.CROP_TOP_FRACTION
        self.num_classes = 3

        self._mean = np.asarray(cfg.IMAGENET_MEAN, dtype=np.float32)
        self._std = np.asarray(cfg.IMAGENET_STD, dtype=np.float32)

        if self.backend == "onnx":
            self._init_onnx()
        else:
            self._init_torch(device)

    # -- backends -----------------------------------------------------------
    def _init_onnx(self) -> None:
        import onnxruntime as ort

        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {self.onnx_path}. Run export_onnx.py first (on a machine with torch)."
            )
        session_options = ort.SessionOptions()
        thread_value = self.onnx_threads or os.environ.get("CROSSING_ONNX_THREADS")
        if thread_value:
            try:
                threads = int(thread_value)
                if threads < 1:
                    raise ValueError
                session_options.intra_op_num_threads = threads
                session_options.inter_op_num_threads = 1
            except ValueError as exc:
                raise ValueError("CROSSING_ONNX_THREADS must be a positive integer") from exc

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.device = "onnxruntime-cpu"
        if thread_value:
            self.device += f"/threads={threads}"

    def _init_torch(self, device: str) -> None:
        import torch  # lazy: only needed for the torch backend

        if str(cfg.TRAINING_DIR) not in sys.path:
            sys.path.insert(0, str(cfg.TRAINING_DIR))
        from model import build_model  # training/model.py

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        ckpt_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        self.input_width = int(ckpt_cfg.get("input_width", cfg.INPUT_WIDTH))
        self.input_height = int(ckpt_cfg.get("input_height", cfg.INPUT_HEIGHT))
        self.crop_top_fraction = float(ckpt_cfg.get("crop_top_fraction", cfg.CROP_TOP_FRACTION))
        self.num_classes = int(ckpt_cfg.get("num_classes", 3))

        self._torch = torch
        self.model = build_model(num_classes=self.num_classes, pretrained=False).to(self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # -- preprocessing (numpy; identical math for both backends) ------------
    def _to_pil_rgb(self, frame) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        array = np.asarray(frame)
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")

    def _crop_top(self, image: Image.Image) -> Image.Image:
        if self.crop_top_fraction <= 0:
            return image
        width, height = image.size
        top = int(round(height * self.crop_top_fraction))
        top = min(max(0, top), height - 1)
        return image.crop((0, top, width, height))

    def preprocess(self, frame) -> tuple[np.ndarray, Image.Image]:
        """Return (model input [1,3,H,W] float32, resized RGB image used as model input)."""
        image = self._to_pil_rgb(frame)
        image = self._crop_top(image)
        resized = image.resize((self.input_width, self.input_height), Image.Resampling.BILINEAR)

        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - self._mean) / self._std
        array = np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)
        return array, resized

    # -- inference ----------------------------------------------------------
    def predict(self, frame, solid_threshold: float | None = cfg.SOLID_CONF_THRESHOLD):
        """frame (PIL/np RGB) -> dict with class-id mask, probabilities, model-input image."""
        array, model_input = self.preprocess(frame)

        if self.backend == "onnx":
            logits = self.session.run([self.output_name], {self.input_name: array})[0]
        else:
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(array).to(self.device)
                logits = self.model(tensor).detach().cpu().numpy()

        probs = _softmax_np(logits[0], axis=0)  # (C, H, W)

        mask = probs.argmax(axis=0).astype(np.uint8)
        if solid_threshold is not None:
            solid_hit = probs[cfg.CLASS_SOLID] >= float(solid_threshold)
            non_solid = probs.copy()
            non_solid[cfg.CLASS_SOLID] = -1.0
            mask = np.where(solid_hit, cfg.CLASS_SOLID, non_solid.argmax(axis=0)).astype(np.uint8)

        return {
            "mask": mask,
            "probs": probs,
            "solid_prob": probs[cfg.CLASS_SOLID],
            "model_input": model_input,
        }


def overlay_mask(model_input: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(model_input.convert("RGB"), dtype=np.float32)
    out = base.copy()
    for class_id, color in cfg.OVERLAY_COLORS.items():
        sel = mask == class_id
        if sel.any():
            tint = np.asarray(color, dtype=np.float32)
            out[sel] = (1.0 - cfg.OVERLAY_ALPHA) * out[sel] + cfg.OVERLAY_ALPHA * tint
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _pick_default_image() -> Path:
    images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in cfg.TEST_IMAGE_DIR.glob(ext))
    if not images:
        raise FileNotFoundError(f"No test images found in {cfg.TEST_IMAGE_DIR}")
    return images[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Visual check: frame -> 3-class mask overlay.")
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--backend", default=None, help="onnx | torch | auto (default)")
    parser.add_argument("--solid-threshold", type=float, default=None)
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else _pick_default_image()
    segmenter = RoadLineSegmenter(backend=args.backend)
    print(f"Backend: {segmenter.backend}  ({segmenter.device})")
    print(f"Model input: {segmenter.input_width}x{segmenter.input_height}, crop_top={segmenter.crop_top_fraction}")

    result = segmenter.predict(Image.open(image_path), solid_threshold=args.solid_threshold)
    mask = result["mask"]
    total = mask.size
    for class_id, name in cfg.CLASS_NAMES.items():
        count = int((mask == class_id).sum())
        print(f"  class {class_id} {name:30s}: {count:8d} px ({100.0 * count / total:5.2f}%)")

    cfg.DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else cfg.DEBUG_OUTPUT_DIR / f"infer_overlay_{image_path.stem}.jpg"
    overlay_mask(result["model_input"], mask).save(output_path, quality=92)
    print(f"Saved overlay: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
