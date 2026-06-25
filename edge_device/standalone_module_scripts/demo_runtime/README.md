# Vehicular Black Box Demo

Run from the project root:

```bash
source /home/pi/FYP\ demo/shouting/venv2/bin/activate
python demo/demo.py --menu
python demo/demo.py --profile 7
python demo/demo.py --profile 1 --api-base-url https://<cloudflare>.trycloudflare.com
python demo/demo.py --profile 20 --api-base-url https://<cloudflare>.trycloudflare.com
```

Camera preview windows are shown automatically when `drowsiness`, `road_sign`, or `lane_crossing`
runs on an attached Raspberry Pi desktop/display:

```bash
python demo/demo.py --profile 4
python demo/demo.py --profile 21
python demo/demo.py --profile 1 --api-base-url https://<cloudflare>.trycloudflare.com
```

For headless SSH runs, disable camera windows with `--no-display`.

Drowsiness uses `--drowsiness-camera auto` by default and probes `/dev/video*`
until one returns frames. If you already know the driver-facing camera index,
pin it explicitly:

```bash
python demo/demo.py --profile 14 --drowsiness-camera 1 --display
```

If the camera clicks or powers on but no preview appears, check the kernel log:

```bash
dmesg | tail -n 120
v4l2-ctl --list-devices
```

Repeated USB `over-current` or disconnect/reconnect messages mean the camera is
detected but cannot stream reliably. Use a powered USB hub or reduce USB load
before rerunning the demo.

If GPS is on a known UART, set it explicitly. Otherwise the demo uses auto-detect
and tries `/dev/ttyAMA3`, `/dev/serial0`, other `ttyAMA*`, `ttyUSB0`, and `ttyACM0`.

```bash
python demo/demo.py --profile 1 --gps-port /dev/ttyAMA3
```

Useful single-model examples:

```bash
python demo/demo.py --profile 11  # hello
python demo/demo.py --profile 12  # horn
python demo/demo.py --profile 13  # shouting
python demo/demo.py --profile 21  # lane crossing violation
python demo/demo.py --models hello,horn,shouting
python demo/demo.py --models harsh,aggressive,tamper
```

Profile `21` runs the new road-line/lane-crossing violation module. It uses the
Wave 3 hybrid runner: ONNX segmentation confirms the restricted solid line, then
OpenCV optical flow tracks the line between model passes. Local evidence is saved
under `demo/proof/<run_id>/lane_crossing/events/` and backend events use:

```text
event_type = LANE_CROSSING
severity = HIGH
```

Road-sign and lane-crossing default to the same road camera source
(`/dev/video2`). If both are run together and the camera cannot be opened twice,
run them separately or set a different source:

```bash
python demo/demo.py --profile 21 --lane-crossing-camera /dev/video2 --gps-port /dev/ttyAMA3 \
  --api-base-url https://<your-tunnel>.trycloudflare.com
python demo/demo.py --profile 4 --road-sign-source /dev/video0 --lane-crossing-camera /dev/video2
```

Profile `20` runs startup/connectivity checks, then starts audio all, drowsiness,
tamper, heartbeat, and the GPS-only speeding monitor:

```bash
python demo/demo.py --profile 20 --audio-rate 44100 --shouting-gain 5 --gps-port /dev/ttyAMA3 \
  --api-base-url https://<your-tunnel>.trycloudflare.com
```

Use `44100` for profile `20` because horn is trained for 44.1 kHz. Hello and
shouting downsample from the shared stream; `--shouting-gain 5` keeps shouting
responsive at the current mic level.

Only drowsiness `EYE_CLOSED` and `HEAD_NOD` local events are sent to the backend,
and both use backend event type `DROWSINESS`; the exact reason is kept in
`_debug.drowsiness_reason`. Distracted, no-face, and generic drowsy local events
are kept as local proof only.

Road-sign detections are context, not backend violations by themselves. Speed
limit signs (`sls-15`, `sls-40`, `sls-50`, `sls-60`, `sls-70`, `sls-80`,
`sls-100`) are compared with GPS speed and send `SPEEDING` only when speed is
over the sign limit plus `--speeding-margin-kmh` (default `5`). Red light
(`tls-r`) sends `RED_LIGHT_VIOLATION` only when GPS speed is above
`--red-light-min-speed-kmh` (default `5`). `no honking` does not send an event by
itself; when road-sign and horn are running together, a horn inside the active
no-honking context sends `HORN` with severity `HIGH`.

GPS speed alone also sends `SPEEDING` with severity `HIGH` when speed is
`100 km/h` or above. Override with `--gps-speeding-threshold-kmh`, or disable
with `--no-gps-speeding`. To run only this GPS-speeding monitor:

```bash
python demo/demo.py --profile 22 --gps-port /dev/ttyAMA3 \
  --api-base-url https://<your-tunnel>.trycloudflare.com
```

Crash audio/IMU/fusion events are sent with event type `CRASH`; the
possible/confirmed result is kept in `_debug.crash_result`.

Horn detection defaults are intentionally conservative for the demo:
`--horn-th-on 0.75 --horn-th-off 0.45 --horn-hits-on 2`. If it still triggers
too easily, increase `--horn-th-on` to `0.80`.

```bash
python demo/demo.py --profile 12 --gps-port /dev/ttyAMA3 \
  --api-base-url https://<your-tunnel>.trycloudflare.com
```

For shouting-only testing, capture at 16 kHz to match the original shouting
runner. If the printed `raw_rms` stays very low while someone is shouting near
the mic, move closer to the mic or add `--shouting-gain 5`.
The demo trigger is tuned for your current mic level with
`--shouting-th-on 0.15 --shouting-th-off 0.05`.

```bash
python demo/demo.py --profile 13 --audio-rate 16000 --gps-port /dev/ttyAMA3 \
  --api-base-url https://<your-tunnel>.trycloudflare.com
```

Events are queued in `demo/runtime/outbox/events.sqlite3` and local proof is saved in
`demo/proof/<run_id>/`. If `API_BASE_URL` or `--api-base-url` is not provided, events stay queued.
Ctrl+C uses fast shutdown by default; unsent events stay in the outbox and retry
on the next run. If you want to wait for one final backend retry before exit,
add `--final-flush-on-exit`.

Backend sends use LTE by default. Without any network flag, the sender requires
`ppp0`; if LTE is missing or has no IPv4 address, events stay queued instead of
falling back to Wi-Fi.

For the SIMCom A7670G wired to Raspberry Pi GPIO14/GPIO15, the modem is normally
`/dev/ttyS0`. The demo now auto-starts PPP on LTE runs using Hutch APN
`hutch3g`, then waits for `ppp0`:

```bash
python demo/demo.py --profile 1 --gps-port /dev/ttyAMA3 \
  --api-base-url https://<your-tunnel>.trycloudflare.com
```

If your SIM uses a different APN, override it:

```bash
python demo/demo.py --profile 1 --lte-apn hutch3g --gps-port /dev/ttyAMA3 \
  --api-base-url https://<your-tunnel>.trycloudflare.com
```

To test only LTE PPP in a separate terminal:

```bash
python demo/lte_ppp.py --port /dev/ttyS0 --apn hutch3g
```

Use Wi-Fi only when you explicitly add `--wifi`:

```bash
python demo/demo.py --profile 1 --wifi --gps-port /dev/ttyAMA3 \
  --api-base-url https://<your-tunnel>.trycloudflare.com
```

Advanced override:

```bash
python demo/demo.py --profile 1 --backend-interface wlan0 --api-base-url https://<cloudflare>.trycloudflare.com
python demo/demo.py --profile 1 --backend-interface default --api-base-url https://<cloudflare>.trycloudflare.com
```

