# Raspberry Pi 4B Always-On Runtime

This folder is the deployable Raspberry Pi package for the vehicular black-box models. It keeps training notebooks and old experiments out of the live system and runs only lightweight artifacts:

- TFLite for Keras/TensorFlow models.
- ONNX or NCNN export for PyTorch/YOLO models.
- `.joblib` directly for small tabular IMU models.

## Folder Layout

- `config/pi_runtime.json` - live runtime configuration and thresholds.
- `config/model_manifest.json` - source training artifacts and deploy outputs.
- `models/` - converted/copy-ready Pi artifacts created by `scripts/convert_models.py`.
- `pi_runtime/` - always-on workers and event aggregator.
- `scripts/` - conversion, export, and validation utilities.

## 1. Convert And Prepare Models

Run this on the laptop/training machine first:

```bash
python raspberry_pi_deploy/scripts/convert_models.py --all
python raspberry_pi_deploy/scripts/export_road_sign.py
```

The scripts create the deploy artifacts under `raspberry_pi_deploy/models/`.

Conversion behavior:

- Horn, shouting, harsh braking, crash IMU, and drowsiness run as TFLite. Crash IMU is rebuilt with a fixed 16-sample unrolled GRU before export so it stays plain TFLite.
- Road-sign detection runs from the NCNN YOLO export, with ONNX kept as fallback. The road-sign classifier runs as ONNX through ONNX Runtime, with OpenCV DNN fallback.
- Hello word detection uses the selected Keras model as the preferred source, but the runtime points at `models/audio/hello_cnn.onnx` as a Pi-safe fallback because the Keras artifact contains Lambda deserialization that is fragile across Keras versions.
- Lane change and aggressive driving stay as `.joblib` models.

## 2. Copy To The Pi

Copy the whole `raspberry_pi_deploy` folder to the Raspberry Pi. Keep the same folder structure.

## 3. Install Pi Dependencies

On the Pi:

```bash
cd raspberry_pi_deploy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
```

For OpenCV camera support on Raspberry Pi OS, system packages are often more reliable than pip wheels:

```bash
sudo apt update
sudo apt install -y python3-opencv libatlas-base-dev libportaudio2 portaudio19-dev
```

## 4. Check Runtime Readiness

```bash
python -m pi_runtime.main --config config/pi_runtime.json --check-only
```

This verifies model paths and optional imports without starting the sensors.

If `serial` is missing on the laptop check, install `pyserial`; it is already listed in `requirements-pi.txt` for the Pi.

## 5. Run Always-On Detection

```bash
python -m pi_runtime.main --config config/pi_runtime.json
```

Events are written to `runtime_output/events.jsonl`. Periodic driver violation index summaries are written to `runtime_output/scores.json` when `Driver violation index/driving_index.py` is available.

## Notes

- The Pi runtime assumes two cameras: driver camera at index `0` and front camera at index `1`. Change these in `config/pi_runtime.json` if needed.
- The runtime is designed for stable 5-10 FPS behavior, not full 30 FPS vision.
- If a converted model is missing, that detector is disabled and the rest of the system continues.
