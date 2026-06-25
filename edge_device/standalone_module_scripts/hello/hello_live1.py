import os
import json
import time
import subprocess
import sys
import numpy as np
import tensorflow as tf

# =========================
# OPTIONAL: legacy Lambda fallback
# =========================
class ReduceSumLambda(tf.keras.layers.Layer):
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
# PATHS
# =========================
BASE_DIR     = "/home/pi/hello-new"
MODEL_PATH   = os.path.join(BASE_DIR, "hello_cnn_tpool2.keras")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
HISTORY_PATH = os.path.join(BASE_DIR, "history.json")

# =========================
# TRAINING-LIKE AUDIO PARAMS
# (1s @ 16k, frame_length=640, frame_step=160 => ~97 frames)
# =========================
TARGET_SR      = 16000
TARGET_SAMPLES = TARGET_SR * 1

FRAME_LENGTH   = 640
FRAME_STEP     = 160
FFT_LENGTH     = 1024

# =========================
# LIVE STREAM PARAMS
# =========================
WIN_SEC   = 1.0
STEP_SEC  = 0.5

EMA_ALPHA   = 0.80
HITS_ON     = 1
HITS_OFF    = 1
SILENCE_RMS = 0.001

# =========================
# MIC (ALSA)
# =========================
ARECORD_DEVICE = "plughw:CARD=sndrpigooglevoi,DEV=0"
ARECORD_FMT    = "S32_LE"
BYTES_PER_SAMP = 4

CAPTURE_SR_PRIMARY  = 16000
CAPTURE_SR_FALLBACK = 44100

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------------
# Model + threshold loaders
# -------------------------
def load_model_with_fallback():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    except Exception as first_err:
        try:
            custom_objects = {"<lambda>": lambda x: tf.reduce_sum(x, axis=1)}
            return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False, custom_objects=custom_objects)
        except Exception as second_err:
            try:
                custom_objects = {"Lambda": ReduceSumLambda, "<lambda>": lambda x: tf.reduce_sum(x, axis=1)}
                return tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False, custom_objects=custom_objects)
            except Exception as third_err:
                print("Failed to load model (direct):", first_err)
                print("Failed to load model (lambda fallback):", second_err)
                print("Failed to load model (Lambda override):", third_err)
                sys.exit(1)

def load_threshold():
    threshold = 0.25
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                m = json.load(f)
            if isinstance(m, dict) and "threshold" in m:
                threshold = float(m["threshold"])
        except Exception:
            pass
    return threshold

# -------------------------
# Audio helpers
# -------------------------
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
    
def start_arecord(capture_sr: int):
    cmd = [
        "arecord",
        "-D", ARECORD_DEVICE,
        "-c", "1",
        "-r", str(capture_sr),
        "-f", ARECORD_FMT,
        "-t", "raw",
        "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def resample_to_target(x: np.ndarray, orig_sr: int, target_sr: int, target_len: int) -> np.ndarray:
    if orig_sr == target_sr:
        y = x.astype(np.float32, copy=False)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        return y[:target_len]

    # try scipy polyphase (best)
    try:
        from scipy.signal import resample_poly
        g = np.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        y = resample_poly(x, up, down).astype(np.float32)
    except Exception:
        # fallback: linear interpolation
        dur = len(x) / float(orig_sr)
        t_old = np.linspace(0.0, dur, num=len(x), endpoint=False)
        t_new = np.linspace(0.0, dur, num=int(round(dur * target_sr)), endpoint=False)
        if len(t_new) <= 1:
            y = np.zeros((target_len,), dtype=np.float32)
        else:
            y = np.interp(t_new, t_old, x).astype(np.float32)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

# -------------------------
# Feature helpers
# -------------------------
def normalize_spec(spec_tf: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(spec_tf)
    std  = tf.math.reduce_std(spec_tf)
    spec_tf = (spec_tf - mean) / (std + 1e-6)

    f_mean = tf.reduce_mean(spec_tf, axis=0, keepdims=True)
    f_std  = tf.math.reduce_std(spec_tf, axis=0, keepdims=True)
    return (spec_tf - f_mean) / (f_std + 1e-6)

def delta_np(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x, dtype=np.float32)
    d[1:-1] = 0.5 * (x[2:] - x[:-2])
    d[0]  = x[1] - x[0]
    d[-1] = x[-1] - x[-2]
    return d
    
def build_features(wav_16k: np.ndarray, model_input_shape):
    """
    Build features exactly matching model input (H,W,C).
    Handles either (T,F,C) or (F,T,C).
    """
    if not model_input_shape or len(model_input_shape) != 4:
        raise ValueError(f"Unexpected model input shape: {model_input_shape}")

    _, Hexp, Wexp, Cexp = model_input_shape
    if Hexp is None or Wexp is None or Cexp is None:
        raise ValueError(f"Model input dims must be fixed (got {model_input_shape})")

    Hexp, Wexp, Cexp = int(Hexp), int(Wexp), int(Cexp)

    # Detect layout:
    # If Hexp==97 => (T,F,C)
    # If Wexp==97 => (F,T,C)
    if Hexp == 97:
        time_first = True
        T_EXPECT, F_EXPECT = Hexp, Wexp
    elif Wexp == 97:
        time_first = False
        T_EXPECT, F_EXPECT = Wexp, Hexp
    else:
        # fallback assume (T,F,C)
        time_first = True
        T_EXPECT, F_EXPECT = Hexp, Wexp

    N_MELS = F_EXPECT  # THIS is the key fix (80 for your model)

    wav_tf = tf.convert_to_tensor(wav_16k, dtype=tf.float32)

    stft = tf.signal.stft(
        wav_tf,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    power = tf.square(tf.abs(stft))  # (T, fft_bins)

    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=power.shape[-1],
        sample_rate=TARGET_SR,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )

    mel = tf.tensordot(power, mel_w, axes=1)  # (T, F)
    mel.set_shape(power.shape[:-1].concatenate([N_MELS]))

    logmel = tf.math.log(tf.maximum(mel, 1e-6))
    spec_tf = normalize_spec(logmel)          # (T,F)
    spec = spec_tf.numpy().astype(np.float32)

    # Ensure time frames = 97
    if spec.shape[0] != T_EXPECT:
        if spec.shape[0] > T_EXPECT:
            spec = spec[:T_EXPECT, :]
        else:
            spec = np.pad(spec, ((0, T_EXPECT - spec.shape[0]), (0, 0)), mode="edge")

    # Build channels
    if Cexp == 1:
        feat = spec[..., None]  # (T,F,1)
    elif Cexp == 4:
        d1 = delta_np(spec)
        d2 = delta_np(d1)
        d3 = delta_np(d2)
        feat = np.stack([spec, d1, d2, d3], axis=-1)  # (T,F,4)
    else:
        base = spec[..., None]
        feat = np.repeat(base, Cexp, axis=-1)

    # If model expects (F,T,C), transpose
    if not time_first:
        feat = np.transpose(feat, (1, 0, 2))  # (F,T,C)

    # Final hard check
    if feat.shape != (Hexp, Wexp, Cexp):
        raise ValueError(f"Feature shape mismatch: got {feat.shape}, expected {(Hexp,Wexp,Cexp)}")

    return feat
    
def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found:", MODEL_PATH)
        sys.exit(1)

    model = load_model_with_fallback()
    threshold = load_threshold()

    TH_ON  = float(threshold)
    TH_OFF = max(0.0, TH_ON - 0.05)

    print(f"? Loaded model: {model.input_shape} -> {model.output_shape}", flush=True)
    print(f"? Using threshold: {TH_ON:.3f} (TH_ON={TH_ON:.3f}, TH_OFF={TH_OFF:.3f})", flush=True)

    # start arecord (try 16k first, fallback to 44.1k)
    cap_sr = CAPTURE_SR_PRIMARY
    proc = start_arecord(cap_sr)
    time.sleep(0.2)

    if proc.poll() is not None:
        err1 = proc.stderr.read().decode(errors="ignore")
        try: proc.kill()
        except Exception: pass

        cap_sr = CAPTURE_SR_FALLBACK
        proc = start_arecord(cap_sr)
        time.sleep(0.2)

        if proc.poll() is not None:
            err2 = proc.stderr.read().decode(errors="ignore")
            print("? arecord failed.", flush=True)
            print("---- 16000 error ----\n", err1, flush=True)
            print("---- 44100 error ----\n", err2, flush=True)
            return

        print("?? Using 44100 capture + resample to 16000", flush=True)

    print(f"??? Device: {ARECORD_DEVICE} @ {cap_sr}Hz ({ARECORD_FMT})", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    step_frames = int(cap_sr * STEP_SEC)
    step_bytes  = step_frames * BYTES_PER_SAMP

    win_frames  = int(cap_sr * WIN_SEC)
    ring = np.zeros(win_frames, dtype=np.float32)
    wpos = 0
    filled = 0

    smooth = 0.0
    on_hits = 0
    off_hits = 0
    triggered = False

    printed_debug = False
    
    
    try:
        while True:
            raw = read_exact(proc.stdout, step_bytes)
            if raw is None:
                err = proc.stderr.read().decode(errors="ignore")
                print("? Audio stream ended.", flush=True)
                if err.strip():
                    print("arecord error:\n", err, flush=True)
                break

            x_i = np.frombuffer(raw, dtype=np.int32)
            x = x_i.astype(np.float32) / 2147483648.0

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
                print(f"buffering {filled/cap_sr:.2f}s / {WIN_SEC:.2f}s", flush=True)
                continue

            if wpos == 0:
                win = ring.copy()
            else:
                win = np.concatenate((ring[wpos:], ring[:wpos]))

            r = rms(win)

            if r < SILENCE_RMS:
                prob = 0.0
            else:
                wav_16k = resample_to_target(win, cap_sr, TARGET_SR, TARGET_SAMPLES)
                feat = build_features(wav_16k, model.input_shape)
                inp = feat[np.newaxis, ...]  # (1,H,W,C)

                if not printed_debug:
                    print("DEBUG feat:", feat.shape, "inp:", inp.shape, "model:", model.input_shape, flush=True)
                    printed_debug = True

                out = model(inp, training=False).numpy()
                prob = float(out[0][0])

            smooth = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * smooth

            if not triggered:
                on_hits = on_hits + 1 if smooth >= TH_ON else 0
                if on_hits >= HITS_ON:
                    triggered = True
                    off_hits = 0
                    print("? HELLO DETECTED!", flush=True)
            else:
                off_hits = off_hits + 1 if smooth <= TH_OFF else 0
                if off_hits >= HITS_OFF:
                    triggered = False
                    on_hits = 0
                    print("? HELLO ended.", flush=True)

            ts = time.strftime("%H:%M:%S")
            state = "HELLO" if triggered else "NOT  "
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
     
