# Dependencies

Dependencies are separated into practical runtime installs and the larger Pi-tested environment.

## Minimal Integrated Runtime

`edge_device/raspberry_pi_deploy/requirements-pi.txt` is the practical install target for the integrated runtime:

| Package | Purpose |
| --- | --- |
| `numpy` | Numeric arrays and feature processing |
| `opencv-python-headless` | Camera/image processing without GUI |
| `sounddevice` | Live audio capture |
| `pyserial` | IMU/GPS/GSM serial communication |
| `joblib` | IMU model loading |
| `scikit-learn` | Classical ML model support |
| `xgboost` | Aggressive-driving model support |
| `onnxruntime` | ONNX inference |
| `tflite-runtime` | TFLite inference on Raspberry Pi |
| `ultralytics` | YOLO/NCNN/ONNX detector fallback support |

## Full Pi-Tested Environment

The following versions were reported from the Raspberry Pi environment used during final testing:

| Package | Version |
| --- | --- |
| `tensorflow` | `2.20.0` |
| `keras` | `3.13.2` |
| `torch` | `2.12.0` |
| `torchvision` | `0.27.0` |
| `torchaudio` | `2.11.0` |
| `onnxruntime` | `1.26.0` |
| `ncnn` | `1.0.20260526` |
| `ultralytics` | `8.4.66` |
| `opencv-python` | `4.13.0.92` |
| `numpy` | `2.4.6` |
| `scipy` | `1.17.1` |
| `pandas` | `3.0.3` |
| `scikit-learn` | `1.7.2` |
| `xgboost` | `3.2.0` |
| `joblib` | `1.5.3` |
| `librosa` | `0.11.0` |
| `soundfile` | `0.13.1` |
| `sounddevice` | `0.5.5` |
| `pillow` | `12.1.1` |
| `matplotlib` | `3.11.0` |
| `tflite-runtime` | `2.14.0` |

See `edge_device/raspberry_pi_deploy/requirements-pi-full-tested.txt` for a pinned list.

## Separate Drowsiness Environment

`edge_device/drowsiness_runtime/requirements.txt` is separate because the drowsiness runtime uses MediaPipe/JAX-related dependencies:

| Package | Version |
| --- | --- |
| `mediapipe` | `0.10.18` |
| `jax` | `0.4.38` |
| `jaxlib` | `0.4.38` |
| `opencv-contrib-python` | `4.11.0.86` |
| `numpy` | `1.26.4` |
| `gpiozero` | `2.0.1` |

## Model-To-Library Map

| Model group | Libraries | Artifacts |
| --- | --- | --- |
| Horn, shouting, crash IMU, harsh braking | TensorFlow/Keras for source/conversion, TFLite for deployment | `.keras`, `.h5`, `.tflite` |
| Hello/wake-word fallback | PyTorch source/export, ONNX Runtime deployment | `.pt`, `.onnx` |
| Crash audio | PyTorch source/export, ONNX Runtime deployment | `.pt`, `.onnx` |
| Road signs | Ultralytics YOLO export, NCNN detector, ONNX classifier | `.pt`, `.onnx`, `.ncnn` |
| Road-line crossing | ONNX Runtime | `.onnx` |
| Lane change and aggressive driving | scikit-learn, XGBoost, joblib | `.joblib` |
| Audio feature extraction | librosa, soundfile, sounddevice | live audio and mel/log-mel features |
| Raspberry Pi hardware | picamera2, gpiozero, RPi.GPIO, smbus2, spidev, pyserial, pigpio, lgpio, v4l2-python3 | camera, GPIO, SPI/I2C, serial devices |

## Runtime Notes

- TensorFlow and PyTorch are heavy on Raspberry Pi and are not the preferred live deployment path.
- ONNX Runtime may print GPU/device warnings on Raspberry Pi. CPU fallback is normal.
- Preferred live deployment uses TFLite, ONNX, NCNN, and joblib artifacts.
