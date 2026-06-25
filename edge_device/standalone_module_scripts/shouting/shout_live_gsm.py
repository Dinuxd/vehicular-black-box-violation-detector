import os, time, json, subprocess, uuid
from datetime import datetime, timezone
import numpy as np
import librosa
import tensorflow as tf
import urllib.request
import urllib.error

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH  = os.path.join(BASE_DIR, "config.json")
META_PATH    = os.path.join(BASE_DIR, "metadata.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model.weights.h5")
MEAN_PATH    = os.path.join(BASE_DIR, "mean.npy")
STD_PATH     = os.path.join(BASE_DIR, "std.npy")

QUEUE_PATH = os.path.join(BASE_DIR, "pending_events.jsonl")
POST_TIMEOUT_SEC = 6

ARECORD_DEVICE = "plughw:CARD=sndrpigooglevoi,DEV=0"
ARECORD_FMT    = "S32_LE"
BYTES_PER_SAMP = 4

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
    """
    1) POST /events
    2) POST /events/{id}/finalize  with {"evidences":[]}
    """
    # create
    status1, body1 = http_post_json(events_url, payload, headers=headers, timeout_sec=POST_TIMEOUT_SEC)
    if status1 not in (200, 201, 202):
        return False, f"create_failed status={status1} body={body1[:120]}"

    # finalize
    event_id = payload["event_id"]
    finalize_url = finalize_base.rstrip("/") + f"/{event_id}/finalize"
    fin_payload = {"evidences": []}

    status2, body2 = http_post_json(finalize_url, fin_payload, headers=headers, timeout_sec=POST_TIMEOUT_SEC)

    # finalize can return 200 OK with {"status":"FINALIZED"} in your code
    if status2 in (200, 201, 202):
        return True, f"finalize_ok status={status2}"

    return False, f"finalize_failed status={status2} body={body2[:120]}"


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
            ok, _msg = send_create_then_finalize(events_url, finalize_base, payload, headers)
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


def rms(x):
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def start_arecord(sr):
    cmd = [
        "arecord",
        "-D", ARECORD_DEVICE,
        "-c", "1",
        "-r", str(sr),
        "-f", ARECORD_FMT,
        "-t", "raw",
        "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)


def load_json_if_exists(p):
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {}


def build_model_from_config(cfg):
    if isinstance(cfg, dict) and "class_name" in cfg and "config" in cfg:
        return tf.keras.models.model_from_json(json.dumps(cfg))
    if "model_json" in cfg:
        mj = cfg["model_json"]
        if isinstance(mj, dict) and "class_name" in mj and "config" in mj:
            return tf.keras.models.model_from_json(json.dumps(mj))
        if isinstance(mj, str):
            return tf.keras.models.model_from_json(mj)
    if "keras_model_json" in cfg and isinstance(cfg["keras_model_json"], str):
        return tf.keras.models.model_from_json(cfg["keras_model_json"])
    raise RuntimeError("config.json does not contain Keras model JSON (architecture).")


def audio_to_logmel(audio, sr, n_fft, hop, n_mels, fmin, fmax, center):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        center=center,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def pad_or_crop_2d(mel, Hexp, Wexp):
    H, W = mel.shape

    if H < Hexp:
        mel = np.pad(mel, ((0, Hexp - H), (0, 0)), mode="edge")
    elif H > Hexp:
        mel = mel[:Hexp, :]

    H, W = mel.shape
    if W < Wexp:
        mel = np.pad(mel, ((0, 0), (0, Wexp - W)), mode="edge")
    elif W > Wexp:
        start = (W - Wexp) // 2
        mel = mel[:, start:start + Wexp]

    return mel

def main():
    for p in [WEIGHTS_PATH, MEAN_PATH, STD_PATH, CONFIG_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    cfg  = load_json_if_exists(CONFIG_PATH)
    meta = load_json_if_exists(META_PATH)

    api_base = os.environ.get("API_BASE_URL", "").strip()
    if not api_base:
        api_base = str(cfg.get("api_base_url", "")).strip()
    if not api_base:
        raise SystemExit(
            "ERROR: API_BASE_URL not set.\n"
            "Run: export API_BASE_URL='https://<your-tunnel>.trycloudflare.com'\n"
        )

    events_url    = api_base.rstrip("/") + "/events"
    finalize_base = api_base.rstrip("/") + "/events"  # finalize is /events/{id}/finalize

    auth_token = os.environ.get("AUTH_TOKEN", cfg.get("auth_token", ""))
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    device_id = os.environ.get("DEVICE_ID", meta.get("device_id", cfg.get("device_id", "pi-device-001")))

    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)
    try:
        mean = mean.astype(np.float32)
        std  = (std.astype(np.float32) + 1e-9)
    except Exception:
        mean = np.float32(np.mean(mean))
        std  = np.float32(np.mean(std) + 1e-9)

    model = build_model_from_config(cfg)
    model.load_weights(WEIGHTS_PATH)

    Hexp = int(model.input_shape[1])
    Wexp = int(model.input_shape[2])

    SR       = int(cfg.get("sample_rate", cfg.get("sr", 16000)))
    WIN_SEC  = float(cfg.get("win_sec", cfg.get("window_seconds", 1.0)))
    STEP_SEC = float(cfg.get("step_sec", cfg.get("hop_seconds", 0.5)))

    N_FFT   = int(cfg.get("n_fft", 1024))
    CENTER  = bool(cfg.get("center", True))
    FMIN    = int(cfg.get("fmin", 0))
    FMAX    = int(cfg.get("fmax", SR // 2))

    hop = int(round((SR * WIN_SEC) / max(Wexp - 1, 1)))
    hop = max(1, hop)

    TH_ON       = float(cfg.get("th_on", 0.75))
    TH_OFF      = float(cfg.get("th_off", 0.15))
    EMA_ALPHA   = float(cfg.get("ema_alpha", 0.80))
    HITS_ON     = int(cfg.get("hits_on", 1))
    HITS_OFF    = int(cfg.get("hits_off", 3))
    SILENCE_RMS = float(cfg.get("silence_rms", 0.001))

    COOLDOWN_SEC = float(cfg.get("event_cooldown_sec", 3.0))
    last_event_ts = 0.0

    print("Loaded shouting model")
    print("   input :", model.input_shape)
    print("   output:", model.output_shape)
    print(f"POST create -> {events_url}")
    print(f"POST finalize -> {finalize_base}/<event_id>/finalize")
    print(f"   device_id={device_id}")
    print("\nPress Ctrl+C to stop.\n", flush=True)

    proc = start_arecord(SR)
    time.sleep(0.2)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="ignore")
        print("arecord exited immediately:\n", err, flush=True)
        return

    step_frames = int(SR * STEP_SEC)
    step_bytes  = step_frames * BYTES_PER_SAMP

    win_frames  = int(SR * WIN_SEC)
    ring = np.zeros(win_frames, dtype=np.float32)
    wpos = 0
    filled = 0

    smooth = 0.0
    on_hits = 0
    off_hits = 0
    triggered = False

    sent_q, fail_q = flush_queue(events_url, finalize_base, headers)
    if sent_q or fail_q:
        print(f"queue flush: sent={sent_q} failed={fail_q}", flush=True)

    try:
        while True:
            raw = read_exact(proc.stdout, step_bytes)
            if raw is None:
                err = proc.stderr.read().decode(errors="ignore")
                print("Audio stream ended.", flush=True)
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
                print(f"buffering {filled/SR:.2f}s / {WIN_SEC:.2f}s", flush=True)
                continue

            win = ring.copy() if wpos == 0 else np.concatenate((ring[wpos:], ring[:wpos]))
            r = rms(win)

            if r < SILENCE_RMS:
                prob = 0.0
            else:
                mel = audio_to_logmel(
                    win, sr=SR,
                    n_fft=N_FFT, hop=hop,
                    n_mels=Hexp,
                    fmin=FMIN, fmax=FMAX,
                    center=CENTER
                )
                mel = pad_or_crop_2d(mel, Hexp, Wexp)

                try:
                    mel_norm = (mel - mean) / std
                except Exception:
                    mean_s = float(np.mean(mean))
                    std_s  = float(np.mean(std))
                    mel_norm = (mel - mean_s) / (std_s + 1e-9)

                inp = mel_norm[np.newaxis, ..., np.newaxis].astype(np.float32)
                y = model.predict(inp, verbose=0)[0]
                prob = float(y[0])

            smooth = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * smooth

            if not triggered:
                on_hits = on_hits + 1 if smooth >= TH_ON else 0
                if on_hits >= HITS_ON:
                    triggered = True
                    off_hits = 0
                    print("SHOUTING DETECTED!", flush=True)

                    now = time.time()
                    if now - last_event_ts >= COOLDOWN_SEC:
                        last_event_ts = now

                        payload = {
                            "event_id": str(uuid.uuid4()),
                            "device_id": device_id,
                            "ts": iso_now_utc(),
                            "event_type": "SHOUTING",
                            "severity": "HIGH",
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
                                print(f"CREATE+FINALIZE OK event_id={payload['event_id']}", flush=True)
                            else:
                                print(f"CREATE/FINALIZE FAIL queued event_id={payload['event_id']} msg={msg}", flush=True)
                                enqueue_event(payload)
                        except Exception as e:
                            print(f"EXCEPTION queued event_id={payload['event_id']} err={e}", flush=True)
                            enqueue_event(payload)

                    sent_q, fail_q = flush_queue(events_url, finalize_base, headers)
                    if sent_q:
                        print(f"queue flush: sent={sent_q} remaining_failed={fail_q}", flush=True)

            else:
                off_hits = off_hits + 1 if smooth <= TH_OFF else 0
                if off_hits >= HITS_OFF:
                    triggered = False
                    on_hits = 0
                    print("Shouting ended.", flush=True)

            ts = time.strftime("%H:%M:%S")
            state = "SHOUT" if triggered else "NOT  "
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
