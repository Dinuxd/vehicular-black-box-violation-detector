# Raspberry Pi Crash Detector

This folder is the copy-and-paste deployment package for the prototype crash detection system.

It includes:

- Audio CNN detector
- IMU Keras AI detector
- IMU threshold/rule detector
- Combined fusion runner

## 1. Copy To Raspberry Pi

Copy the whole folder:

```bash
raspberry_pi_crash_detector/
```

to your Raspberry Pi 4B.

Recommended OS: Raspberry Pi OS 64-bit.

## 2. Install Dependencies

Open a terminal inside this folder on the Pi:

```bash
cd raspberry_pi_crash_detector
sudo apt update
sudo apt install -y python3-venv libsndfile1 portaudio19-dev
bash install_pi.sh
source .venv/bin/activate
```

For the full system including the IMU AI model, install TensorFlow too:

```bash
bash install_pi.sh --with-tensorflow
source .venv/bin/activate
```

If `torch` or `tensorflow` fails to install, use Raspberry Pi OS 64-bit and Python 3.10 or 3.11.

## 3. Test Each Detector

Audio CNN:

```bash
python detect_crash.py file samples/sample_car_crash.wav
```

Expected output shape:

```text
Audio: samples/demo_synthetic_long.wav
Threshold: 0.6089
Windows scored: ...
Detections: ...
```

If a crash is detected, it prints:

```text
start_time,end_time,max_score
```

You can also test a non-crash road sample:

```bash
python detect_crash.py file samples/sample_road_traffic.wav
```

IMU threshold algorithm:

```bash
python imu_threshold_detector.py --csv samples/sample_imu_normal.csv
python imu_threshold_detector.py --csv samples/sample_imu_threshold_crash.csv
```

IMU AI model:

```bash
python imu_ai_detector.py --csv samples/sample_imu_crash.csv
```

If the IMU AI command says TensorFlow is unavailable, run `bash install_pi.sh --with-tensorflow`.

## 4. Run Combined Audio + IMU Detection

Confirmed crash test without TensorFlow:

```bash
python combined_detect.py --audio samples/sample_car_crash.wav --imu-csv samples/sample_imu_threshold_crash.csv --skip-imu-ai
```

Full system test with IMU AI:

```bash
python combined_detect.py --audio samples/sample_car_crash.wav --imu-csv samples/sample_imu_crash.csv
```

The fusion output can be:

```text
CRASH_CONFIRMED
POSSIBLE_CRASH
NO_CRASH
```

Save the full result:

```bash
python combined_detect.py --audio samples/sample_car_crash.wav --imu-csv samples/sample_imu_crash.csv --out result.json
```

## 5. Run On Your Own Audio File

```bash
python detect_crash.py file your_audio.wav
```

Save CSV outputs:

```bash
python detect_crash.py file your_audio.wav --save-prefix result
```

This creates:

```text
result_detections.csv
result_timeline.csv
```

## 6. Run Live From A Microphone

List microphone devices:

```bash
python detect_crash.py mic --list-devices
```

Listen for 60 seconds:

```bash
python detect_crash.py mic --seconds 60
```

Print every window score while listening:

```bash
python detect_crash.py mic --seconds 60 --print-all
```

Use a specific input device:

```bash
python detect_crash.py mic --device 1 --seconds 60
```

## 7. Expected IMU CSV Columns

For IMU AI:

```text
Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z, Speed_kmh
```

For IMU threshold:

```text
timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
```

The scripts accept common uppercase/lowercase variants such as `Acc_X` and `acc_x`.

## Notes

- The model expects mono audio at 44.1 kHz. The script converts normal audio files automatically.
- Live microphone mode records at 44.1 kHz.
- This is a prototype detector trained from the current dataset. Test it with real Raspberry Pi recordings before trusting it in the field.
- `combined_detect.py` confirms a crash when at least two detector signals agree. A single detector signal becomes `POSSIBLE_CRASH`.
