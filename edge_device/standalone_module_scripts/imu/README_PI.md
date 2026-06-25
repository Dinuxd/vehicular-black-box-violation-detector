# Raspberry Pi IMU Deployment

This folder contains the three final model artifacts and one runner script.

Run on the Raspberry Pi:

```bash
source /home/pi/FYP\ demo/shouting/venv2/bin/activate
export API_BASE_URL="https://<your-tunnel>.trycloudflare.com"
export DEVICE_ID="pi-001"
python run_imu_models.py \
  --imu-interface bmi160-spi \
  --spi-bus 0 \
  --spi-device 0 \
  --gps-port /dev/serial0 \
  --gps-baud 9600 \
  --thresholds-file runtime_thresholds.json \
  --log-dir logs
```

The script posts detections to:

```text
POST ${API_BASE_URL}/events
Content-Type: application/json
```

Each event uses a fresh UUID `event_id`, `DEVICE_ID`, UTC timestamps, and the
backend event names `HARSH_BRAKING`, `LANE_CHANGE`, or `AGGRESSIVE_DRIVING`.
If no serial GPS is available during testing, set fallback coordinates:

```bash
export GPS_LATITUDE="6.915800"
export GPS_LONGITUDE="79.977733"
export GPS_ACCURACY_M="5.0"
```

Default thresholds are `0.90` for all three models. Thresholds can be changed
while the runner is still running by editing `runtime_thresholds.json` from
another terminal:

```bash
printf '{\n  "harsh_braking": 0.90,\n  "lane_change": 0.90,\n  "aggressive_driving": 0.90\n}\n' > runtime_thresholds.json
```

The next inference pass reloads the file automatically.

Every run writes CSV logs to `logs/`:

```text
logs/imu_samples_<run timestamp>.csv
logs/imu_events_<run timestamp>.csv
```

The sample CSV contains accelerometer XYZ, gyroscope XYZ, the latest three
model probabilities, thresholds, GPS, and triggered event names. The event CSV
contains each triggered event timestamp, event id, model probability, threshold,
GPS, and backend send status.

If TensorFlow reports a protobuf mismatch in a new environment, install or
repair the dependencies with `pip install -r requirements_pi.txt`.

The script defaults to a Bosch BMI160 over SPI, matching the project hardware.
It also has fallback readers for MPU6050 over I2C and MPU-6000/MPU-6500/MPU-9250
style SPI IMUs.

Keep the runner at `--sample-rate-hz 20`. The final models were trained and
exported for 20 Hz windows.

Useful options:

```bash
python run_imu_models.py --backend-url http://YOUR_BACKEND_HOST/api/events
python run_imu_models.py --thresholds-file runtime_thresholds.json
python run_imu_models.py --log-dir logs
python run_imu_models.py --imu-interface bmi160-spi --spi-bus 0 --spi-device 0
python run_imu_models.py --imu-interface mpu6050-i2c --i2c-bus 1 --i2c-address 0x68
python run_imu_models.py --imu-interface spi
python run_imu_models.py --spi-bus 0 --spi-device 0
python run_imu_models.py --gps-port /dev/serial0 --gps-baud 9600
python run_imu_models.py --no-gps
python run_imu_models.py --yaw-axis gz --forward-accel-axis ax --lateral-accel-axis ay
python run_imu_models.py --csv sample.csv --csv-realtime
```

Detection POSTs use this structure:

```json
{
  "event_id": "lane-change-550e8400-e29b-41d4-a716-446655440001",
  "device_id": "pi-001",
  "ts": "2026-06-11T12:51:00Z",
  "event_type": "LANE_CHANGE",
  "severity": "MEDIUM",
  "gps": {
    "latitude": 6.9158,
    "longitude": 79.977733,
    "captured_at": "2026-06-11T12:51:00Z",
    "accuracy_m": 5.0
  }
}
```

