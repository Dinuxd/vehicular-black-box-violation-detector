#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_pi.txt

if [ "${1:-}" = "--with-tensorflow" ]; then
  python -m pip install -r requirements_pi_tensorflow.txt
fi

echo
echo "Install complete."
echo "Activate with: source .venv/bin/activate"
echo "Audio test: python detect_crash.py file samples/sample_car_crash.wav"
echo "IMU threshold test: python imu_threshold_detector.py --csv samples/sample_imu_normal.csv"
echo "Combined test: python combined_detect.py --audio samples/sample_car_crash.wav --imu-csv samples/sample_imu_threshold_crash.csv --skip-imu-ai"
