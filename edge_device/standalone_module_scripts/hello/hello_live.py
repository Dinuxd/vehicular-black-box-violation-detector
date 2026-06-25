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
# PATHS
# =========================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "hello_cnn_tpool2.keras")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
HISTORY_PATH = os.path.join(BASE_DIR, "history.json")


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
STEP_SEC  = 0.5  # slide every 0.5s (same idea as your Windows HOP_SEC)

# stability / false positive control
EMA_ALPHA   = 0.90
HITS_ON     = 1     # require 2 consecutive hits to turn ON
HITS_OFF    = 1      # require 3 consecutive lows to turn OFF
SILENCE_RMS = 0.001  # if too sensitive, increase (e.g., 0.004)


# =========================
# I2S MIC via ALSA (same style as horn code)
# Change this if your card name is different
# run: arecord -l
# =========================
ARECORD_DEVICE = "plughw:CARD=sndrpigooglevoi,DEV=0"

# Try TARGET_SR first. If your mic doesn't support 16000, set CAPTURE_SR=44100 or 48000
CAPTURE_SR     = 16000
ARECORD_FMT    = "S32_LE"   # your horn setup used S32_LE
BYTES_PER_SAMP = 4          # S32_LE => 4 bytes. If you use S16_LE then set 2.


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
    threshold = 0.1
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
    """If CAPTURE_SR != TARGET_SR, resample to 16k. (simple + works without extra libs)"""
    if CAPTURE_SR == TARGET_SR:
        return x

    # best if scipy exists
    try:
        from scipy.signal import resample_poly
        # polyphase: TARGET_SR/CAPTURE_SR
        # choose integer ratio via gcd
        g = np.gcd(CAPTURE_SR, TARGET_SR)
        up = TARGET_SR // g
        down = CAPTURE_SR // g
        y = resample_poly(x, up, down).astype(np.float32)
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
        return y[:TARGET_SAMPLES]
    except Exception:
        # fallback: linear interpolation
        xin = np.arange(len(x), dtype=np.float32)
        xout = np.linspace(0, len(x) - 1, TARGET_SAMPLES, dtype=np.float32)
        y = np.interp(xout, xin, x).astype(np.float32)
        return y

def normalize_spec(spec):
    # global z-score
    mean = tf.reduce_mean(spec)
    std = tf.math.reduce_std(spec)
    spec = (spec - mean) / (std + 1e-6)

    # per-frequency z-score
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
    spec = normalize_spec(logmel)           # (T, F)
    spec = tf.expand_dims(spec, -1)         # (T, F, 1)

    # If model was trained as (F, T, 1) instead of (T, F, 1), auto-fix
    # model_input_shape like: (None, H, W, C)
    if model_input_shape and len(model_input_shape) == 4:
        H = model_input_shape[1]
        W = model_input_shape[2]
        # if H==N_MELS, it's likely expecting (F, T)
        if H == N_MELS and W != N_MELS:
            spec = tf.transpose(spec, perm=[1, 0, 2])  # (F, T, 1)

    feat = spec.numpy().astype(np.float32)

 # make sure it's (T, F, 1)
    if feat.ndim == 2:
       feat = feat[..., None]

# safety: force F=40 if anything changes
    if feat.shape[1] != 40:
       feat = feat[:, :40, :]

    return feat



def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found:", MODEL_PATH)
        sys.exit(1)

    model = load_model_with_fallback()
    threshold, info = load_threshold()

    # hysteresis around threshold
    TH_ON  = threshold
    TH_OFF = max(0.0, threshold - 0.05)

    print(f"Loaded model: {model.input_shape} -> {model.output_shape}", flush=True)
    print(f"? Using threshold: {threshold:.3f} (TH_ON={TH_ON:.3f}, TH_OFF={TH_OFF:.3f})", flush=True)
    print(f"Device: {ARECORD_DEVICE} @ {CAPTURE_SR}Hz ({ARECORD_FMT})", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    proc = start_arecord()
    time.sleep(0.2)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="ignore")
        print("arecord exited immediately:\n", err, flush=True)
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

    try:
        while True:
            raw = read_exact(proc.stdout, step_bytes)
            if raw is None:
                err = proc.stderr.read().decode(errors="ignore")
                print("Audio stream ended (arecord stopped).", flush=True)
                if err.strip():
                    print("arecord error:\n", err, flush=True)
                break

            if ARECORD_FMT == "S32_LE":
                x_i = np.frombuffer(raw, dtype=np.int32)
                x = x_i.astype(np.float32) / 2147483648.0
            else:
                # if you change to S16_LE
                x_i = np.frombuffer(raw, dtype=np.int16)
                x = x_i.astype(np.float32) / 32768.0

            # write into ring buffer
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

            # reconstruct 1s window in correct time order
            if wpos == 0:
                win = ring.copy()
            else:
                win = np.concatenate((ring[wpos:], ring[:wpos]))

            r = rms(win)

            if r < SILENCE_RMS:
                prob = 0.0
            else:
                win_16k = resample_if_needed(win)  # ensures 16000 samples
                feat = wav_to_features(win_16k, model.input_shape)
                inp = feat[np.newaxis, ...]  # (1, T, F, 1) or (1, F, T, 1)

                # faster than predict()
                out = model(inp, training=False).numpy()
                prob = float(out[0][0])

            smooth = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * smooth

            # hysteresis + consecutive hits
            if not triggered:
                on_hits = on_hits + 1 if smooth >= TH_ON else 0
                if on_hits >= HITS_ON:
                    triggered = True
                    off_hits = 0
                    print("HELLO DETECTED!", flush=True)
            else:
                off_hits = off_hits + 1 if smooth <= TH_OFF else 0
                if off_hits >= HITS_OFF:
                    triggered = False
                    on_hits = 0
                    print("HELLO ended.", flush=True)

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
            try:
                proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
