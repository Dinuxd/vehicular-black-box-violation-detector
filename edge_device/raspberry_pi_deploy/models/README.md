# Converted Model Artifacts

This directory is populated by:

```bash
python raspberry_pi_deploy/scripts/convert_models.py --all
python raspberry_pi_deploy/scripts/export_road_sign.py
```

Do not put training notebooks here. Only copy or generate deployable files used by the Raspberry Pi runtime.

Expected live artifacts:

- `audio/horn_cnn_best_int8.tflite`
- `audio/shouting_int8.tflite`
- `audio/hello_cnn.onnx`
- `audio/crash_audio_cnn.onnx`
- `imu/harsh_braking_int8.tflite`
- `imu/crash_imu_int8.tflite`
- `imu/best_lane_change_detector.joblib`
- `imu/normal_vs_aggressive_imu3_best_model.joblib`
- `drowsiness/eye_model_int8.tflite`
- `road_sign/yolo26n_sign_640_ncnn_model/`
- `road_sign/best_classifier.onnx`
- `road_sign/best_classifier.onnx.data`
- `road_line/best_model.onnx`
