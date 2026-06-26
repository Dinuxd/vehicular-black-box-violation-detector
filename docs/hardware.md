# Hardware

## Main Hardware

- Raspberry Pi 4B
- Front-facing road camera
- Driver-facing camera
- Microphone/audio input
- BMI160 IMU
- GPS module
- GSM/LTE modem
- Tamper switch/sensor
- Custom PCB and power circuit
- Vehicle-mounted enclosure and wiring

## Hardware Folder

`hardware/pcb/` contains PCB screenshots and Gerber deliverables for the power and functional boards. Gerber zip files are intentionally committed because they are hardware artifacts, unlike runtime logs or dataset archives.

## Integration Notes

- Camera indices are configured in `edge_device/raspberry_pi_deploy/config/pi_runtime.json`.
- IMU serial input defaults to `/dev/ttyACM0` at 115200 baud.
- GSM/LTE upload expects backend URL and auth token to be supplied through environment variables or module config.
- Power stability matters; brownouts can corrupt local logs or interrupt upload.
