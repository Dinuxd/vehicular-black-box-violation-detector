import os
import json
import time
import subprocess
import sys
import uuid
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
import urllib.request


# =========================
# OPTIONAL: legacy Lambda fallback
# =========================
class ReduceSumLambda(tf.keras.layers.Layer):
    """Drop-in for legacy Lambda reduce_sum used in older checkpoints."""
    def __init__(self, **kwargs):
        for k in ["function", "output_shape", "arguments"]:
            kwargs.pop(k, None)
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        _ = mask
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


# =========================
# PATHS (YOUR PI PATH)
# =========================
BASE_DIR     = "/home/pi/FYP demo/hello-new"
MODEL_PATH   = os.path.join(BASE_DIR, "hello_cnn_tpool2.keras")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
HISTORY_PATH = os.path.join(BASE_DIR, "history.json")

# offline queue
QUEUE_PATH = os.path.join(BASE_DIR, "pending_hello_events.jsonl")
POST_TIMEOUT_SEC = 6

# =========================
# FEATURE PARAMS (MATCH TRAINING)
# =========================
TARGET_SR       = 16000
TARGET_SAMPLES  = TARGET_SR * 1  # 1 second

FRAME_LENGTH    = 640
FRAME_STEP      = 160
FFT_LENGTH      = 1024
N_MELS          = 40

# =========================
# LIVE STREAM PARAMS
# =========================
WIN_SEC   = 1.0
STEP_SEC  = 0.5

EMA_ALPHA   = 0.90
HITS_ON     = 1
HITS_OFF    = 1
SILENCE_RMS = 0.001

# avoid spamming events
COOLDOWN_SEC = 3.0

# =========================
# I2S MIC via ALSA
# =========================
ARECORD_DEVICE = "plughw:CARD=sndrpigooglevoi,DEV=0"
CAPTURE_SR     = 16000
ARECORD_FMT    = "S32_LE"
BYTES_PER_SAMP = 4

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# =========================
# HTTP helpers
# =========================
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


# =========================
# helpers
# =========================
def load_model_with_fallback():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    except Exception as first_err:
        try:
            custom_objects = {"<lambda>": lambda x: tf.reduce_sum(x, axis=1)}
            return tf.keras.models.load_model(
                MODEL_PATH, compile=False, safe_mode=False, custom_objects=custom_objects
            )
        except Exception as second_err:
            try:
                custom_objects = {
                    "Lambda": ReduceSumLambda,
                    "<lambda>": lambda x: tf.reduce_sum(x, axis=1),
                }
                return tf.keras.models.load_model(
                    MODEL_PATH, compile=False, safe_mode=False, custom_objects=custom_objects
                )
            except Exception as third_err:
                print("Failed to load model (direct):", first_err)
                print("Failed to load model (lambda fallback):", second_err)
                print("Failed to load model (Lambda override):", third_err)
                sys.exit(1)

def load_threshold():
    threshold = 0.3
    info = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                m = json.load(f)
            if isinstance(m, dict):
                info.update(m)
                if "threshold" in m:
                    threshold = float(m["threshold"])
        except Exception:
            pass

    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                h = json.load(f)
            if isinstance(h, dict):
                info["history"] = h
        except Exception:
            pass

    return threshold, info

def read_exact(stream, nbytes):
    chunks = []
    got = 0
    while got < nbytes:
        chunk = stream.read(nbytes - got)
        if not chunk:
            return None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)

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

def rms(x):
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def resample_if_needed(x):
    if CAPTURE_SR == TARGET_SR:
        return x

    try:
        from scipy.signal import resample_poly
        g = np.gcd(CAPTURE_SR, TARGET_SR)
        up = TARGET_SR // g
        down = CAPTURE_SR // g
        y = resample_poly(x, up, down).astype(np.float32)
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
        return y[:TARGET_SAMPLES]
    except Exception:
        xin = np.arange(len(x), dtype=np.float32)
        xout = np.linspace(0, len(x) - 1, TARGET_SAMPLES, dtype=np.float32)
        return np.interp(xout, xin, x).astype(np.float32)

def normalize_spec(spec):
    mean = tf.reduce_mean(spec)
    std = tf.math.reduce_std(spec)
    spec = (spec - mean) / (std + 1e-6)

    f_mean = tf.reduce_mean(spec, axis=0, keepdims=True)
    f_std = tf.math.reduce_std(spec, axis=0, keepdims=True)
    return (spec - f_mean) / (f_std + 1e-6)

def wav_to_features(wav_16k, model_input_shape):
    wav_tf = tf.convert_to_tensor(wav_16k, dtype=tf.float32)

    stft = tf.signal.stft(
        wav_tf,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    power = tf.square(tf.abs(stft))

    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=power.shape[-1],
        sample_rate=TARGET_SR,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )

    mel = tf.tensordot(power, mel_w, axes=1)
    mel.set_shape(power.shape[:-1].concatenate([N_MELS]))

    logmel = tf.math.log(tf.maximum(mel, 1e-6))
    spec = normalize_spec(logmel)       # (T, F)
    spec = tf.expand_dims(spec, -1)     # (T, F, 1)

    # If model expects (F, T, 1), transpose
    if model_input_shape and len(model_input_shape) == 4:
        H = model_input_shape[1]
        W = model_input_shape[2]
        if H == N_MELS and W != N_MELS:
            spec = tf.transpose(spec, perm=[1, 0, 2])  # (F, T, 1)

    feat = spec.numpy().astype(np.float32)

    # safety: ensure (T/F, 40, 1) has F=40
    if feat.ndim == 2:
        feat = feat[..., None]
    if feat.shape[1] != 40:
        feat = feat[:, :40, :]

    return feat

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found:", MODEL_PATH)
        sys.exit(1)

    # ---- API URL (must be provided) ----
    api_base = os.environ.get("API_BASE_URL", "").strip()
    if not api_base:
        raise SystemExit(
            "ERROR: API_BASE_URL not set.\n"
            "Example:\n  export API_BASE_URL='https://<tunnel>.trycloudflare.com'\n"
        )

    events_url    = api_base.rstrip("/") + "/events"
    finalize_base = api_base.rstrip("/") + "/events"

    # optional auth header
    auth_token = os.environ.get("AUTH_TOKEN", "").strip()
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    device_id = os.environ.get("DEVICE_ID", "pi-device-001")

    model = load_model_with_fallback()
    threshold, _info = load_threshold()

    TH_ON  = float(threshold)
    TH_OFF = max(0.0, TH_ON - 0.05)

    print(f"? Loaded model: {model.input_shape} -> {model.output_shape}", flush=True)
    print(f"? Threshold: {TH_ON:.3f} (TH_ON={TH_ON:.3f}, TH_OFF={TH_OFF:.3f})", flush=True)
    print(f"??? Device: {ARECORD_DEVICE} @ {CAPTURE_SR}Hz ({ARECORD_FMT})", flush=True)
    print(f"??? POST create   -> {events_url}", flush=True)
    print(f"??? POST finalize -> {finalize_base}/<event_id>/finalize", flush=True)
    print(f"   device_id={device_id}", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    proc = start_arecord()
    time.sleep(0.2)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="ignore")
        print("? arecord exited immediately:\n", err, flush=True)
        return

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

    # flush queued at start
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

            if ARECORD_FMT == "S32_LE":
                x_i = np.frombuffer(raw, dtype=np.int32)
                x = x_i.astype(np.float32) / 2147483648.0
            else:
                x_i = np.frombuffer(raw, dtype=np.int16)
                x = x_i.astype(np.float32) / 32768.0

            # ring buffer
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
                win_16k = resample_if_needed(win)
                feat = wav_to_features(win_16k, model.input_shape)
                inp = feat[np.newaxis, ...]
                out = model(inp, training=False).numpy()
                prob = float(out[0][0])

            smooth = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * smooth

            # hysteresis + hits
            if not triggered:
                on_hits = on_hits + 1 if smooth >= TH_ON else 0
                if on_hits >= HITS_ON:
                    triggered = True
                    off_hits = 0
                    print("?? HELLO DETECTED!", flush=True)

                    # send only on rising edge + cooldown
                    now = time.time()
                    if now - last_event_ts >= COOLDOWN_SEC:
                        last_event_ts = now

                        payload = {
                            "event_id": str(uuid.uuid4()),
                            "device_id": device_id,
                            "ts": iso_now_utc(),
                            "event_type": "HELLO_WAKEWORD",
                            "severity": "LOW",
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
                    print("? HELLO ended.", flush=True)

            ts = time.strftime("%H:%M:%S")
            state = "HELLO" if triggered else "NOT  "
            print(f"{ts} | {state} | rms={r:.4f} | prob={prob:.3f} | smooth={smooth:.3f}", flush=True)

            # periodic flush
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
