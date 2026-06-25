# Limitations

This repository is an academic FYP prototype. It should not be described as production-ready.

## Technical Limits

- Real-world road testing was limited.
- Some ML modules have honest but prototype-level metrics.
- Camera and audio performance depends strongly on Raspberry Pi load, lighting, microphone placement, vibration, and vehicle cabin conditions.
- GSM/LTE upload depends on network coverage and backend availability.
- GPS may be unavailable or inaccurate in tunnels, urban canyons, or indoor testing.
- Raspberry Pi 4B can run the selected deploy artifacts, but TensorFlow and PyTorch are heavy on the device.

## Safety Limits

- Not certified for emergency response.
- Not certified for insurance, legal evidence, or driver punishment decisions.
- Not validated against automotive safety standards.
- Not a replacement for vehicle OEM safety systems.

## ML Limits

- Dataset coverage is limited compared with commercial road-safety systems.
- Some classes may have low recall or false positives under unseen environments.
- Thresholds require calibration for each vehicle, microphone, IMU mount, camera angle, and road context.
- The project prioritizes transparent deployment and honest model reporting over inflated claims.
