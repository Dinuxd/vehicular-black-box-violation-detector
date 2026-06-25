# Vehicular Black Box Driver Drowsiness Prototype

This project is a Raspberry Pi 4 camera-only runtime for detecting driver eye closure, drowsiness, head nodding, and attention-away events using a USB IR-cut camera and MediaPipe Face Landmarker.

## Hardware

- Raspberry Pi 4 running 64-bit Raspberry Pi OS Bookworm
- USB IR-cut camera facing the driver
- Active buzzer or buzzer module on GPIO 17 and GND

The camera must appear as `/dev/video0` or another `/dev/videoX` device. If it does not appear, reconnect the camera, check power, and confirm the user is in the `video` group.

## Setup

```bash
sudo apt update
sudo apt install -y python3-venv v4l-utils
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_model.py
```

On Raspberry Pi OS Bookworm, install Python packages inside a virtual environment.

## Camera Bring-Up

```bash
source .venv/bin/activate
python -m drowsiness_blackbox --camera-check
```

This prints detected `/dev/video*` devices, V4L2 formats, OpenCV capture health, and saves `camera_test_frame.jpg`.

## Run

```bash
source .venv/bin/activate
python -m drowsiness_blackbox --display
```

For headless operation:

```bash
python -m drowsiness_blackbox --no-buzzer
```

To print live detections in the terminal:

```bash
python -m drowsiness_blackbox --no-buzzer --print-detections
```

To send drowsiness violations to the backend after a condition stays active for 3 seconds:

```bash
export API_BASE_URL="https://<your-tunnel>.trycloudflare.com"
export DEVICE_ID="pi-001"
python -m drowsiness_blackbox --display --no-buzzer --print-detections
```

POSTs are sent to `$API_BASE_URL/events` as `DROWSINESS_DETECTED`. If `gpsd` has a real GPS fix, that fix is used. Otherwise the fallback GPS is `6.9158, 79.977733` with `5m` accuracy. Override fallback GPS with `FALLBACK_GPS_LATITUDE`, `FALLBACK_GPS_LONGITUDE`, and `FALLBACK_GPS_ACCURACY_M`.

The first 5 seconds are calibration. Keep your face visible, look forward, and keep both eyes open. After calibration, the terminal output prints the live left/right EAR and calibrated threshold.

If open eyes are still sometimes classified as closed, use a stricter threshold:

```bash
python -m drowsiness_blackbox --display --print-detections --max-eye-closed-ear 0.18 --eye-closed-ratio 0.45
```

To save one annotated face-detection image:

```bash
python -m drowsiness_blackbox --snapshot-detect detections/last_detection.jpg --no-buzzer
```

Useful options:

- `--camera-index 1` if the camera appears as `/dev/video1`
- `--width 320 --height 240` if FPS is too low
- `--log-dir blackbox_logs` for evidence output
- `--buzzer-gpio 17` for the GPIO warning pin

## Method

The application calibrates for 5 seconds while the driver looks forward with eyes open. It then computes:

- Eye Aspect Ratio from MediaPipe eye landmarks
- Sustained eye closure and PERCLOS for drowsiness
- Relative yaw and pitch from calibrated head pose
- Repeated pitch cycles for head nodding
- Missing face for occlusion or attention-away evidence

Events are debounced with cooldowns so one unsafe condition logs one violation instead of many repeated lines.

## Evidence Output

Events are stored by day under `blackbox_logs/YYYYMMDD/`:

- `logs/events_YYYYMMDD.jsonl`
- `logs/events_YYYYMMDD.csv`
- `snapshots/*.jpg`
- `clips/*.avi`

Each log record includes the event type, wall-clock time, duration, metrics, snapshot path, clip path, and FPS.

## Tests

```bash
python -m unittest discover -s tests
```

The unit tests cover the pure geometry and temporal event rules. Full validation still requires real camera tests in daylight, IR/night mode, normal blinking, sustained eye closure, head turns, looking down, repeated nodding, and no-face occlusion.

