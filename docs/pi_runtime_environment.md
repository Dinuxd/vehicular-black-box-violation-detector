# Raspberry Pi Runtime Environment

## Minimal Integrated Runtime

Use:

```bash
pip install -r edge_device/raspberry_pi_deploy/requirements-pi.txt
```

This focuses on deploy-time libraries: NumPy, OpenCV, sounddevice, pyserial, joblib, scikit-learn, XGBoost, ONNX Runtime, TFLite Runtime, and Ultralytics.

## Full Tested Environment

Use this only when reproducing the full model-testing environment:

```bash
pip install -r edge_device/raspberry_pi_deploy/requirements-pi-full-tested.txt
```

TensorFlow and PyTorch were installed in the tested Pi environment, but they are heavy. The preferred live runtime path is still TFLite, ONNX, NCNN, and joblib.

## Check Installed Versions

```bash
python scripts/check_pi_environment.py
python scripts/check_pi_environment.py --include-heavy-imports
```

The first command uses package metadata and avoids slow imports. The second command tries heavy imports such as TensorFlow/PyTorch and may take time on Raspberry Pi.

## Expected Warning

ONNX Runtime can print GPU/device warnings on Raspberry Pi. That normally means it falls back to CPU execution, which is expected for this project.
