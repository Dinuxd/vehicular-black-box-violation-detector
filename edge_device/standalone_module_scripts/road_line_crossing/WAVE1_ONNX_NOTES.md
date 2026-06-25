# Wave 1 — ONNX Runtime (engine swap only)

This folder is a copy of `pi_deploy` with **one optimisation**: the model can run
through **ONNX Runtime** instead of PyTorch. Nothing else changed — same input size
(512x288), same crop, same 3-class mask, same decision logic, debug video still on.

## What changed vs pi_deploy
- `crossing/export_onnx.py` (new) — converts the model to ONNX once, on a laptop.
- `crossing/infer.py` — gained an ONNX backend (PyTorch path kept). Same output.
- `requirements_pi.txt` — uses `onnxruntime`; **torch is no longer needed on the Pi**.

The decision modules (`mask_postprocess.py`, `line_tracker.py`, `crossing_logic.py`,
`event_logger.py`) and `run_video.py` are unchanged.

## Step 1 — export the ONNX model  (ALREADY DONE - included in this folder)
`best_model.onnx` (single self-contained file, ~26 MB) is already built and verified,
sitting next to `best_model.pth`. It was checked against PyTorch: max logit diff
~1e-5 and 100% identical masks, and the full flow produced the exact same crossing
events on both backends. **So you can skip this step.**

Only re-run it if you ever change the model. On a laptop with PyTorch:
```
cd road_line_project/crossing
pip install torch torchvision onnxruntime onnxscript onnx
py export_onnx.py
```
(On Windows the exporter prints unicode status; the script already forces UTF-8 so it
won't crash.)

## Step 2 — copy this whole folder to the Pi
(Includes the freshly created `best_model.onnx`.) Same transfer options as before
(VNC file transfer, scp, USB).

## Step 3 — on the Pi: install deps (no torch needed)
```
cd ~/pi_deploy_wave1
python3 -m pip install -r requirements_pi.txt
```

## Step 4 — run / benchmark (it auto-uses ONNX)
`infer.py` picks the backend automatically: if `best_model.onnx` is present and
`onnxruntime` is installed, it uses ONNX; otherwise PyTorch.
```
cd ~/pi_deploy_wave1/road_line_project/crossing
python3 run_video.py --video /path/clip_h264.mp4     # produces annotated.mp4 (your demo)
python3 benchmark.py --video /path/clip_h264.mp4 --frames 100
```
Force a backend if you want to compare:
```
CROSSING_BACKEND=onnx  python3 benchmark.py --frames 100
CROSSING_BACKEND=torch python3 benchmark.py --frames 100   # only if torch is installed
```

## Expectations
- ONNX Runtime is typically ~2-4x faster than PyTorch on the Pi CPU, with the
  **same accuracy** (it is the same float math).
- This alone will not reach live real-time from a very low baseline, but it is a
  free, risk-free speedup and is enough to produce the annotated **demo video**
  faster. Further options (smaller input, INT8, frame-trigger) remain available
  later and were intentionally left out of Wave 1.
