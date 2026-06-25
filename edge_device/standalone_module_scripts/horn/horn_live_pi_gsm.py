import os, time, json, subprocess, uuid
from datetime import datetime, timezone
import numpy as np
import librosa
import tensorflow as tf
import urllib.request

# -------------------------
# PATHS
# -------------------------
BASE_DIR = "/home/pi/FYP demo/horn-new"
MODEL_PATH = os.path.join(BASE_DIR, "horn_cnn_best.keras")
NORM_PATH  = os.path.join(BASE_DIR, "norm_stats.npz")

# offline queue for horn events
QUEUE_PATH = os.path.join(BASE_DIR, "pending_horn_events.jsonl")
POST_TIMEOUT_SEC = 6

# -------------------------
# MATCH TRAINING
# -------------------------
TARGET_SR  = 44100
WIN_SEC    = 1.0
STEP_SEC   = 0.25

N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 512
POWER      = 2.0
CENTER     = True  # gives W=87 for 1s (matches your model: 128x87)

# -------------------------
# DETECTION STABILITY
# -------------------------
TH_ON       = 0.15
TH_OFF      = 0.10
EMA_ALPHA   = 0.80
HITS_ON     = 1
HITS_OFF    = 2
SILENCE_RMS = 0.001

# cooldown to avoid spamming
COOLDOWN_SEC = 3.0

# -------------------------
# I2S MIC
# -------------------------
ARECORD_DEVICE = "plughw:CARD=sndrpigooglevoi,DEV=0"
CAPTURE_SR     = 44100
ARECORD_FMT    = "S32_LE"
BYTES_PER_SAMP = 4

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============= HTTP helpers =============
def iso_now_utc():
    return datetime.now(timezone.utc).isoformat()

def http_post_json(url, payload, headers=None, timeout_sec=6):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            if v is not None and v != "":
                req.add_header(k, str(v))
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, body

def enqueue_event(payload):
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(QUEUE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

def send_create_then_finalize(events_url, finalize_base, payload, headers):
    # 1) create
    s1, b1 = http_post_json(events_url, payload, headers=headers, timeout_sec=POST_TIMEOUT_SEC)
    if s1 not in (200, 201, 202):
        return False, f"create_failed status={s1} body={b1[:120]}"

    # 2) finalize
    event_id = payload["event_id"]
    fin_url = finalize_base.rstrip("/") + f"/{event_id}/finalize"
    fin_payload = {"evidences": []}

    s2, b2 = http_post_json(fin_url, fin_payload, headers=headers, timeout_sec=POST_TIMEOUT_SEC)
    if s2 in (200, 201, 202):
        return True, f"finalize_ok status={s2}"

    return False, f"finalize_failed status={s2} body={b2[:120]}"

def flush_queue(events_url, finalize_base, headers):
    if not os.path.exists(QUEUE_PATH):
        return 0, 0

    try:
        with open(QUEUE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return 0, 1

    if not lines:
        return 0, 0

    kept = []
    sent = 0
    failed = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            failed += 1
            continue

        try:
            ok, _ = send_create_then_finalize(events_url, finalize_base, payload, headers)
            if ok:
                sent += 1
            else:
                kept.append(line)
                failed += 1
        except Exception:
            kept.append(line)
            failed += 1

    tmp = QUEUE_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
        os.replace(tmp, QUEUE_PATH)
    except Exception:
        pass

    return sent, failed


# ============= audio helpers =============
def read_exact(stream, nbytes: int):
    chunks = []
    got = 0
    while got < nbytes:
        chunk = stream.read(nbytes - got)
        if not chunk:
            return None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)

def audio_to_logmel(audio_441: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio_441,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=POWER,
        center=CENTER,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def start_arecord():
    cmd = [
        "arecord",
        "-D", ARECORD_DEVICE,
        "-c", "1",
        "-r", str(CAPTURE_SR),
        "-f", ARECORD_FMT,
        "-t", "raw",
        "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

def main():
    # ---- API URL (must be provided) ----
    api_base = os.environ.get("API_BASE_URL", "").strip()
    if not api_base:
        raise SystemExit("ERROR: API_BASE_URL not set. Example:\nexport API_BASE_URL='https://<tunnel>.trycloudflare.com'\n")

    events_url    = api_base.rstrip("/") + "/events"
    finalize_base = api_base.rstrip("/") + "/events"

    # optional auth header
    auth_token = os.environ.get("AUTH_TOKEN", "").strip()
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    # device id
    device_id = os.environ.get("DEVICE_ID", "pi-device-001")

    # Load norm
    stats = np.load(NORM_PATH)
    mean = float(np.array(stats["mean"]).reshape(-1)[0])
    std  = float(np.array(stats["std"]).reshape(-1)[0]) + 1e-9

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"? Loaded model: {model.input_shape} -> {model.output_shape}", flush=True)
    print(f"? Norm: mean={mean:.6f}, std={std:.6f}", flush=True)
    print(f"??? POST create -> {events_url}", flush=True)
    print(f"??? POST finalize -> {finalize_base}/<event_id>/finalize", flush=True)
    print(f"   device_id={device_id}", flush=True)

    # Start audio stream
    proc = start_arecord()
    time.sleep(0.2)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="ignore")
        print("? arecord exited immediately:\n", err, flush=True)
        return

    print(f"??? Live capture started: {ARECORD_DEVICE} @ {CAPTURE_SR}Hz ({ARECORD_FMT})", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    step_frames = int(CAPTURE_SR * STEP_SEC)
    step_bytes  = step_frames * BYTES_PER_SAMP

    win_frames  = int(CAPTURE_SR * WIN_SEC)
    ring = np.zeros(win_frames, dtype=np.float32)
    wpos = 0
    filled = 0

    smooth = 0.0
    on_hits = 0
    off_hits = 0
    triggered = False

    last_event_ts = 0.0

    # flush old queued at start
    sent_q, fail_q = flush_queue(events_url, finalize_base, headers)
    if sent_q or fail_q:
        print(f"? queue flush: sent={sent_q} failed={fail_q}", flush=True)

    try:
        while True:
            raw = read_exact(proc.stdout, step_bytes)
            if raw is None:
                err = proc.stderr.read().decode(errors="ignore")
                print("? Audio stream ended (arecord stopped).", flush=True)
                if err.strip():
                    print("arecord error:\n", err, flush=True)
                break

            x_i32 = np.frombuffer(raw, dtype=np.int32)
            x = x_i32.astype(np.float32) / 2147483648.0

            n = len(x)
            end = wpos + n
            if end <= win_frames:
                ring[wpos:end] = x
            else:
                k = win_frames - wpos
                ring[wpos:] = x[:k]
                ring[:end - win_frames] = x[k:]
            wpos = (wpos + n) % win_frames
            filled = min(win_frames, filled + n)

            if filled < win_frames:
                print(f"buffering {filled/CAPTURE_SR:.2f}s / {WIN_SEC:.2f}s", flush=True)
                continue

            win = ring.copy() if wpos == 0 else np.concatenate((ring[wpos:], ring[:wpos]))
            r = rms(win)
            
            if r < SILENCE_RMS:
                prob = 0.0
            else:
                mel = audio_to_logmel(win)          # (128,87)
                mel_norm = (mel - mean) / std
                inp = mel_norm[np.newaxis, ..., np.newaxis]  # (1,128,87,1)
                prob = float(model.predict(inp, verbose=0)[0][0])

            smooth = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * smooth

            if not triggered:
                on_hits = on_hits + 1 if smooth >= TH_ON else 0
                if on_hits >= HITS_ON:
                    triggered = True
                    off_hits = 0
                    print("?? HORN DETECTED!", flush=True)

                    now = time.time()
                    if now - last_event_ts >= COOLDOWN_SEC:
                        last_event_ts = now

                        payload = {
                            "event_id": str(uuid.uuid4()),
                            "device_id": device_id,
                            "ts": iso_now_utc(),
                            "event_type": "HORN",
                            "severity": "MEDIUM",
                            "media": [],
                            "_debug": {
                                "prob": round(prob, 4),
                                "smooth": round(smooth, 4),
                                "rms": round(r, 6)
                            }
                        }

                        try:
                            ok, msg = send_create_then_finalize(events_url, finalize_base, payload, headers)
                            if ok:
                                print(f"? CREATE+FINALIZE OK event_id={payload['event_id']}", flush=True)
                            else:
                                print(f"? CREATE/FINALIZE FAIL queued event_id={payload['event_id']} msg={msg}", flush=True)
                                enqueue_event(payload)
                        except Exception as e:
                            print(f"? EXCEPTION queued event_id={payload['event_id']} err={e}", flush=True)
                            enqueue_event(payload)

                    # flush after detection
                    sent_q, fail_q = flush_queue(events_url, finalize_base, headers)
                    if sent_q:
                        print(f"? queue flush: sent={sent_q} remaining_failed={fail_q}", flush=True)

            else:
                off_hits = off_hits + 1 if smooth <= TH_OFF else 0
                if off_hits >= HITS_OFF:
                    triggered = False
                    on_hits = 0
                    print("? Horn ended.", flush=True)

            ts = time.strftime("%H:%M:%S")
            state = "HORN" if triggered else "NOT "
            print(f"{ts} | {state} | rms={r:.4f} | prob={prob:.3f} | smooth={smooth:.3f}", flush=True)

            if int(time.time()) % 5 == 0:
                flush_queue(events_url, finalize_base, headers)

    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
