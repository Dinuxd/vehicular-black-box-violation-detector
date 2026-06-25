#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/source_venv.sh"

python run_pi_ncnn_onnx.py --picamera --width 1280 --height 720 --frame-skip 2 --display "$@"
