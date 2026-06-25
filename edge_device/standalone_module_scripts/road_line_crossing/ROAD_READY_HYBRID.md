# Wave 3 Hybrid Road Runner

Use this for live road testing with the USB camera. It is faster than pure model
inference because the model confirms the solid line periodically and OpenCV tracks
that line between confirmations.

## USB Camera Command

```bash
cd ~/vehicular-black-box-violation-detector/edge_device/standalone_module_scripts/road_line_crossing/road_line_project/crossing
source "../../../shouting/venv2/bin/activate"

python run_hybrid_live.py \
  --camera /dev/video0 \
  --camera-backend v4l2 \
  --camera-fourcc MJPG \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --tracker-fps 30 \
  --model-interval 1.0 \
  --onnx-threads 4 \
  --display
```

For driving tests, remove `--display` to reduce CPU load:

```bash
python run_hybrid_live.py --camera /dev/video0 --camera-backend v4l2 --camera-fourcc MJPG --camera-width 1280 --camera-height 720 --camera-fps 30 --tracker-fps 30 --model-interval 1.0 --onnx-threads 4
```

Stop with `Ctrl+C`.

## Send Lane-Crossing Events To Backend

Use the same Cloudflare tunnel / GSM connection pattern as the integrated runtime:

```bash
python run_hybrid_live.py \
  --camera /dev/video0 \
  --camera-backend v4l2 \
  --camera-fourcc MJPG \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fps 30 \
  --tracker-fps 30 \
  --model-interval 1.0 \
  --onnx-threads 4 \
  --gps-port /dev/ttyAMA3 \
  --api-base-url https://<your-tunnel>.trycloudflare.com \
  --device-id pi-001
```

If GPS should auto-detect, use:

```bash
--gps-port auto
```

If GPS is not connected yet, add:

```bash
--no-gps
```

Backend events use:

```text
event_type = LANE_CROSSING
severity = HIGH
```

They are queued in the shared runtime outbox:

```text
../../../runtime_outputs/runtime/outbox/events.sqlite3
```

If the GSM/Cloudflare connection drops, events remain pending and flush later
when the tunnel is reachable.

## Output

The script prints:

```text
Captured frames: ...
Tracker frames: ... throughput=... FPS
Model confirmations: ... rate=... FPS
Events fired: ...
Backend outbox: pending=... sent=...
```

Event JSON, event frames, and short event clips are saved under:

```text
road_line_project/crossing/debug_outputs/hybrid_camera_*/
```

## Calibration

Default profile is `usb-road`:

```text
source_crop=none
model_crop_top=0.25
ego_center=0.50
hysteresis_left=0.42
hysteresis_right=0.58
```

If the camera is mounted off-center, tune:

```bash
--ego-center 0.50 --hysteresis-left 0.42 --hysteresis-right 0.58
```

The green line in display mode is the vehicle path. It should line up with the
center of the vehicle's forward path in the camera view.

This is a road-test runtime, not a certified safety system. Validate it with
recorded drives before relying on event logs.

