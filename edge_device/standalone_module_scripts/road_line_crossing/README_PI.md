# Raspberry Pi Deployment â€” Road-Line Crossing Flow (baseline test)

This folder is a **self-contained copy** of the runtime needed to run the crossing
flow on a Raspberry Pi 4B. Your original project was not modified. Goal: run the
**current (unoptimised) flow** on the Pi and measure its real performance.

```
pi_deploy/
  README_PI.md            <- this file
  requirements_pi.txt     <- Python deps
  road_line_project/
    crossing/             <- the runtime code + benchmark.py
    training/             <- model.py, config.py (network definition)
    training_outputs/.../models/best_model.pth   <- trained weights (26 MB)
```

Nothing here is optimised yet (no quantisation / no resolution drop). It is the
exact laptop flow, so you get a true baseline. Optimisation can be added later in
this same folder without touching the originals.

---

## A. Connect to the Pi over VNC (same Wi-Fi)

You said VNC is already installed. Steps:

1. **On the Pi (one time), make sure VNC + SSH are enabled.** On the Pi desktop:
   `Menu -> Preferences -> Raspberry Pi Configuration -> Interfaces` and turn ON
   both **VNC** and **SSH**. (CLI alternative: `sudo raspi-config` ->
   `Interface Options` -> enable VNC and SSH.)

2. **Find the Pi's address.** On the Pi, open a terminal and run:
   ```
   hostname -I        # shows the IP, e.g. 192.168.1.42
   hostname           # shows the name, usually raspberrypi
   ```
   On the same Wi-Fi you can usually reach it as `raspberrypi.local` without
   knowing the IP.

3. **On your Windows PC, open RealVNC Viewer.** In the address bar type either:
   - `raspberrypi.local`  (or your Pi's hostname), or
   - the IP from step 2, e.g. `192.168.1.42`
   Press Enter, then log in with the Pi's username and password (default user is
   often `pi`). You now see the Pi desktop.

   If `raspberrypi.local` does not resolve on Windows, install Apple Bonjour
   (bundled with iTunes) or just use the IP address.

---

## B. Copy this folder to the Pi

Pick ONE method:

- **RealVNC file transfer (easiest):** in the VNC Viewer window's toolbar there is
  a file-transfer button; send the whole `pi_deploy` folder to the Pi's home.

- **SCP from Windows PowerShell** (needs SSH enabled on the Pi):
  ```
  scp -r "~/vehicularbbx/pi_deploy" pi@raspberrypi.local:~/
  ```

- **WinSCP** (GUI SFTP): connect to `raspberrypi.local`, drag `pi_deploy` across.

- **USB stick:** copy `pi_deploy` to a stick, plug into the Pi, copy to home.

After copying, on the Pi you should have `~/pi_deploy/road_line_project/...`.

---

## C. Set up Python on the Pi (one time)

Open a terminal on the Pi (via VNC) and run:
```
sudo apt update
sudo apt install -y python3-pip ffmpeg
cd ~/pi_deploy
python3 -m pip install -r requirements_pi.txt
```
`torch`/`torchvision` take a while to install. Requires **64-bit Raspberry Pi OS**.

Quick check that torch + opencv import:
```
python3 -c "import torch, cv2, numpy, PIL; print('ok', torch.__version__, cv2.__version__)"
```

---

## D. Run the flow

Put a test clip on the Pi (an H.264 .mp4). Phone clips are HEVC; convert first on
your laptop (or on the Pi with `convert_to_h264.py`). Then:
```
cd ~/pi_deploy/road_line_project/crossing
python3 run_video.py --video /path/to/test_clip_h264.mp4
```
Outputs land in `crossing/debug_outputs/run_<clip>/` (annotated.mp4 + events/...).
It runs headless; no display needed.

---

## E. Measure performance

**1. Clean model / pipeline FPS (recommended) â€” use the benchmark tool:**
```
cd ~/pi_deploy/road_line_project/crossing
python3 benchmark.py                      # synthetic frame
python3 benchmark.py --video /path/clip_h264.mp4 --frames 100
python3 benchmark.py --threads 4          # use all 4 cores
```
It prints ms/frame and FPS for "model only" and "model + opencv". This is the
cleanest number (excludes video read/write).

**2. End-to-end run timing:**
```
time python3 run_video.py --video /path/clip_h264.mp4
```
Frames processed are roughly clip_seconds x target_fps (default 10). Divide that by
the real time `time` reports to get end-to-end FPS.

**3. While it runs, in a second VNC terminal, watch load + heat:**
```
htop                       # CPU and RAM per core
vcgencmd measure_temp      # temperature
vcgencmd get_throttled     # 0x0 = fine; nonzero = it throttled (add cooling)
```

**Tips for a fair test**
- Run the benchmark ALONE first to get the model's raw speed, then run it WITH your
  other models active to see the realistic contended speed.
- Ignore the slow first run / first import; judge steady-state numbers.
- The model dominates the cost; the OpenCV part is milliseconds.

---

## F. After you see the numbers

If it is too slow (likely, especially alongside other models), the optimisation
levers, in order of impact, are: convert the model to **ONNX Runtime or TFLite
INT8**, **lower the input resolution** (320x192 / 256x160), **run fewer frames**
(sample to ~5 FPS or trigger from the IMU), and **disable the annotated-video
writing** in production. Those can be added in this folder later without touching
the original project.

