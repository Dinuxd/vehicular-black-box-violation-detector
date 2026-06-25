# Raspberry Pi Setup

## Integrated Runtime

```bash
cd edge_device/raspberry_pi_deploy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
python -m pi_runtime.main --config config/pi_runtime.json --check-only
```

Run live:

```bash
python -m pi_runtime.main --config config/pi_runtime.json
```

## System Packages

Raspberry Pi OS often works better when OpenCV/camera/audio support is installed through apt:

```bash
sudo apt update
sudo apt install -y python3-opencv libatlas-base-dev libportaudio2 portaudio19-dev
```

Depending on hardware modules, install or enable:

- camera interface
- I2C/SPI/serial
- GPIO permissions
- GSM/LTE modem network configuration

## Environment Variables

Use placeholders and do not commit real values:

```bash
export API_BASE_URL="https://<your-backend-host>"
export AUTH_TOKEN="<device-token>"
```

## Drowsiness Runtime

The drowsiness module has a separate environment:

```bash
cd edge_device/drowsiness_runtime
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m drowsiness_blackbox --help
```
