import os, time, json, subprocess
import numpy as np
import librosa
import tensorflow as tf

BASE_DIR     = "/home/pi/FYP demo/shouting"
CONFIG_PATH  = os.path.join(BASE_DIR, "config.json")
META_PATH    = os.path.join(BASE_DIR, "metadata.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model.weights.h5")
MEAN_PATH    = os.path.join(BASE_DIR, "mean.npy")
STD_PATH     = os.path.join(BASE_DIR, "std.npy")

ARECORD_DEVICE = "plughw:CARD=sndrpigooglevoi,DEV=0"
ARECORD_FMT    = "S32_LE"
BYTES_PER_SAMP = 4

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def start_arecord(sr: int):
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

def build_model_from_config(cfg: dict):
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

def audio_to_logmel(audio: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int, fmin: int, fmax: int, center: bool):
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

def pad_or_crop_2d(mel: np.ndarray, Hexp: int, Wexp: int) -> np.ndarray:
    # mel: (H, W)
    H, W = mel.shape

    # Fix height
    if H < Hexp:
        mel = np.pad(mel, ((0, Hexp - H), (0, 0)), mode="edge")
    elif H > Hexp:
        mel = mel[:Hexp, :]

    # Fix width
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

    # ---- LOAD NORM ----
    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)

    # if shapes are weird, fall back to scalar mean/std
    try:
        mean = mean.astype(np.float32)
        std  = (std.astype(np.float32) + 1e-9)
    except Exception:
        mean = np.float32(np.mean(mean))
        std  = np.float32(np.mean(std) + 1e-9)

    # ---- BUILD + LOAD WEIGHTS ----
    model = build_model_from_config(cfg)
    model.load_weights(WEIGHTS_PATH)

    # ---- EXPECTED INPUT SHAPE ----
    # (None, H, W, 1)
    Hexp = int(model.input_shape[1])
    Wexp = int(model.input_shape[2])

    # ---- SETTINGS (force match model) ----
    SR      = int(cfg.get("sample_rate", cfg.get("sr", 16000)))
    WIN_SEC = float(cfg.get("win_sec", cfg.get("window_seconds", 1.0)))
    STEP_SEC= float(cfg.get("step_sec", cfg.get("hop_seconds", 0.5)))

    N_FFT   = int(cfg.get("n_fft", 1024))
    CENTER  = bool(cfg.get("center", True))
    FMIN    = int(cfg.get("fmin", 0))
    FMAX    = int(cfg.get("fmax", SR // 2))

    # choose hop so 1 sec -> about Wexp frames
    # with center=True, frames approx = 1 + floor(len/hop)
    hop = int(round((SR * WIN_SEC) / max(Wexp - 1, 1)))
    hop = max(1, hop)

    # thresholds
    TH_ON       = float(cfg.get("th_on", 0.75))
    TH_OFF      = float(cfg.get("th_off", 0.15))
    EMA_ALPHA   = float(cfg.get("ema_alpha", 0.80))
    HITS_ON     = int(cfg.get("hits_on", 1))
    HITS_OFF    = int(cfg.get("hits_off", 3))
    SILENCE_RMS = float(cfg.get("silence_rms", 0.001))

    print("? Loaded shouting model")
    print("   input :", model.input_shape)
    print("   output:", model.output_shape)
    print(f"??? Live capture: {ARECORD_DEVICE} @ {SR}Hz ({ARECORD_FMT})")
    print(f"   EXPECT mel: {Hexp}x{Wexp} | using n_fft={N_FFT}, hop={hop}, center={CENTER}")
    print("\nPress Ctrl+C to stop.\n", flush=True)

    # ---- AUDIO STREAM ----
    proc = start_arecord(SR)
    time.sleep(0.2)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="ignore")
        print("? arecord exited immediately:\n", err, flush=True)
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

    try:
        while True:
            raw = read_exact(proc.stdout, step_bytes)
            if raw is None:
                err = proc.stderr.read().decode(errors="ignore")
                print("? Audio stream ended.", flush=True)
                if err.strip():
                    print("arecord error:\n", err, flush=True)
                break

            x_i32 = np.frombuffer(raw, dtype=np.int32)
            x = x_i32.astype(np.float32) / 2147483648.0

            # ring buffer write
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

                # force exact shape for model
                mel = pad_or_crop_2d(mel, Hexp, Wexp)

                # normalize (scalar or broadcastable arrays)
                try:
                    mel_norm = (mel - mean) / std
                except Exception:
                    mean_s = float(np.mean(mean))
                    std_s  = float(np.mean(std))
                    mel_norm = (mel - mean_s) / (std_s + 1e-9)

                inp = mel_norm[np.newaxis, ..., np.newaxis].astype(np.float32)  # (1,H,W,1)

                y = model.predict(inp, verbose=0)[0]
                prob = float(y[0])  # output is (None,1)

            smooth = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * smooth

            if not triggered:
                on_hits = on_hits + 1 if smooth >= TH_ON else 0
                if on_hits >= HITS_ON:
                    triggered = True
                    off_hits = 0
                    print("?? SHOUTING DETECTED!", flush=True)
            else:
                off_hits = off_hits + 1 if smooth <= TH_OFF else 0
                if off_hits >= HITS_OFF:
                    triggered = False
                    on_hits = 0
                    print("? Shouting ended.", flush=True)

            ts = time.strftime("%H:%M:%S")
            state = "SHOUT" if triggered else "NOT  "
            print(f"{ts} | {state} | rms={r:.4f} | prob={prob:.3f} | smooth={smooth:.3f}", flush=True)

    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            try: proc.kill()
            except Exception: pass

if __name__ == "__main__":
    main()
