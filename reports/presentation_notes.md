# Final Presentation Notes

Project title: **Vehicular Black Box: Vehicle Violation Detector**

The final presentation described a complete vehicle monitoring prototype with edge detection, event upload, backend storage, dashboard visualization, hardware integration, and driver scoring.

## Main System Idea

Sensor input is processed on the Raspberry Pi. Model/rule detections become violation events. Events are locally recorded, uploaded through GSM/LTE when available, stored in backend services, and displayed in a dashboard.

Presentation-level flow:

```text
Sensor Input -> Model/Rule Detection -> Violation Logic -> Event Record -> Local Storage -> GSM Upload -> Dashboard
```

## Highlighted Components

- Audio detection: horn, hello/phone-call cue, shouting, crash audio.
- IMU detection: harsh braking, lane change, aggressive driving, crash/impact.
- Camera detection: driver drowsiness, road signs, road-line crossing.
- GPS and speed/location rules.
- Tamper detection.
- GSM/LTE communication.
- Backend APIs, PostgreSQL, and dashboard.
- Driver violation index.
- Custom PCB/power/enclosure hardware.

## Backend/Frontend Stack From Presentation

- Go backend services.
- PostgreSQL database.
- React, TypeScript, Tailwind dashboard.
- JWT/login and bearer-style ingest concepts.
- Device/event views, GPS map, event history, and scoring output.

## Prototype Caveat

The final presentation and this repository both treat the system as a university FYP prototype. Real-world validation, safety certification, legal evidence handling, and production hardening would be separate future work.
