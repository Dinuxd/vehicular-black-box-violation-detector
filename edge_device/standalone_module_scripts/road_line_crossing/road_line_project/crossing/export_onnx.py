"""Export the frozen PyTorch model to ONNX (run ONCE, on a machine with torch).

Produces best_model.onnx next to best_model.pth, then verifies the ONNX output
matches PyTorch numerically so we KNOW the mask is identical (decision logic
unaffected). After this, the Raspberry Pi can run the flow via onnxruntime with NO
PyTorch installed.

    py export_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import config_crossing as cfg
from infer import RoadLineSegmenter

# The torch ONNX exporter prints unicode (emoji) status; make stdout UTF-8 so it
# does not crash on a Windows cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    import torch

    seg = RoadLineSegmenter(backend="torch")
    model = seg.model
    model.eval()
    # Export at the configured (Wave 2: reduced) input size, not the checkpoint's
    # training size. The model is fully convolutional, so it runs at any size.
    h, w = cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH
    onnx_path = Path(cfg.DEFAULT_CHECKPOINT).with_suffix(".onnx")

    dummy = torch.randn(1, 3, h, w)
    print(f"Exporting {cfg.DEFAULT_CHECKPOINT.name} -> {onnx_path.name}  (input 1x3x{h}x{w})")
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=18, dynamic_axes=None,
    )
    # Consolidate external weights into ONE self-contained .onnx file (easier to transfer).
    import onnx
    model_proto = onnx.load(str(onnx_path))  # pulls in the external best_model.onnx.data
    onnx.save_model(model_proto, str(onnx_path), save_as_external_data=False)
    data_file = Path(str(onnx_path) + ".data")
    if data_file.exists():
        data_file.unlink()
    print(f"Saved (single file): {onnx_path}  ({onnx_path.stat().st_size // (1024 * 1024)} MB)")

    # --- verify ONNX == PyTorch on a fresh random input ---
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        t_logits = model(x).numpy()
    o_logits = sess.run(["logits"], {"input": x.numpy()})[0]

    max_diff = float(np.abs(t_logits - o_logits).max())
    t_mask = t_logits[0].argmax(0)
    o_mask = o_logits[0].argmax(0)
    agree = float((t_mask == o_mask).mean()) * 100.0

    print("\n--- verification (PyTorch vs ONNX) ---")
    print(f"max |logit difference| : {max_diff:.6e}")
    print(f"argmax mask agreement  : {agree:.4f}%")
    ok = max_diff < 1e-3 and agree > 99.9
    print("RESULT:", "OK - outputs match, decision logic unaffected." if ok else "CHECK - larger diff than expected.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
