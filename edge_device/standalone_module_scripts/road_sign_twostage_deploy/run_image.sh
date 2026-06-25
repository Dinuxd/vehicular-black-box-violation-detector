#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/source_venv.sh"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash run_image.sh /path/to/image_or_folder"
  exit 1
fi

python run_pi_ncnn_onnx.py --source "$1" "${@:2}"
