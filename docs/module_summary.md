# Module Summary

| Module | Folder | Notes |
| --- | --- | --- |
| Integrated Pi runtime | `edge_device/raspberry_pi_deploy` | Main deployable edge package |
| Audio modules | `edge_device/standalone_module_scripts/horn`, `hello`, `shouting` | Earlier standalone scripts plus integrated deploy artifacts |
| Crash detection | `edge_device/standalone_module_scripts/crash_detector` and integrated models | Audio ONNX plus IMU TFLite path |
| IMU driving events | integrated runtime and `standalone_module_scripts/imu` | Harsh braking, lane change, aggressive driving |
| Road signs | integrated road-sign models and `road_sign_twostage_deploy` scripts | NCNN/ONNX detector plus ONNX classifier |
| Road-line crossing | `standalone_module_scripts/road_line_crossing` and `models/road_line/best_model.onnx` | ONNX segmentation/crossing pipeline |
| Drowsiness | `edge_device/drowsiness_runtime` and integrated TFLite model | Separate MediaPipe/JAX runtime environment |
| GPS/GSM/tamper | `standalone_module_scripts/gps`, `gsm`, `tamper` | Hardware interface scripts |
| Backend | `backend/ingest-service`, `backend/media-service` | Go APIs and PostgreSQL integration |
| Dashboard | `frontend/vehicular-bbx-portal` | React device/event dashboard |
| Driver scoring | `scoring/driver_violation_index` | Python score model and sample events |
