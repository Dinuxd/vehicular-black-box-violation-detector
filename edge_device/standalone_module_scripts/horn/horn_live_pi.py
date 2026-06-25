import os, time, subprocess
import numpy as np
import librosa
import tensorflow as tf

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "horn_cnn_best.keras")
NORM_PATH  = os.path.join(BASE_DIR, "norm_stats.npz")

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
# DETECTION STABILITY (reduce false positives)
# -------------------------
TH_ON       = 0.05
TH_OFF      = 0.03
EMA_ALPHA   = 0.80
HITS_ON     = 1      # ~0.75s
HITS_OFF    = 2      # ~1.0s
SILENCE_RMS = 0.001  # increase if it triggers too easily

# -------------------------
# I2S MIC (INMP441 via googlevoicehat overlay)
# -------------------------
ARECORD_DEVICE = "plughw:CARD=sndrpigooglevoi,DEV=0"
CAPTURE_SR     = 44100
ARECORD_FMT    = "S32_LE"   # IMPORTANT: your card works with this
BYTES_PER_SAMP = 4          # S32_LE = 4 bytes

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def read_exact(stream, nbytes: int):
    """Read exactly nbytes from a pipe (handles partial reads). Returns None on EOF."""
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
    # Load norm
    stats = np.load(NORM_PATH)
    mean = float(np.array(stats["mean"]).reshape(-1)[0])
    std  = float(np.array(stats["std"]).reshape(-1)[0]) + 1e-9

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Loaded model: {model.input_shape} -> {model.output_shape}", flush=True)
    print(f"Norm: mean={mean:.6f}, std={std:.6f}", flush=True)

    # Start audio stream
    proc = start_arecord()
    time.sleep(0.2)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="ignore")
        print("arecord exited immediately:\n", err, flush=True)
        return

    print(f"Live capture started: {ARECORD_DEVICE} @ {CAPTURE_SR}Hz ({ARECORD_FMT})", flush=True)
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
    
    try:
        while True:
            raw = read_exact(proc.stdout, step_bytes)
            if raw is None:
                err = proc.stderr.read().decode(errors="ignore")
                print("Audio stream ended (arecord stopped).", flush=True)
                if err.strip():
                    print("arecord error:\n", err, flush=True)
                break

            # int32 -> float32 [-1,1]
            x_i32 = np.frombuffer(raw, dtype=np.int32)
            x = x_i32.astype(np.float32) / 2147483648.0

            # Write into ring buffer
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
                # show buffering progress
                print(f"buffering {filled/CAPTURE_SR:.2f}s / {WIN_SEC:.2f}s", flush=True)
                continue

            # Reconstruct 1s window in correct time order
            if wpos == 0:
                win = ring.copy()
            else:
                win = np.concatenate((ring[wpos:], ring[:wpos]))

            r = rms(win)

            if r < SILENCE_RMS:
                prob = 0.0
            else:
                mel = audio_to_logmel(win)          # (128,87)
                mel_norm = (mel - mean) / std
                inp = mel_norm[np.newaxis, ..., np.newaxis]  # (1,128,87,1)
                prob = float(model.predict(inp, verbose=0)[0][0])

            smooth = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * smooth

            # hysteresis + consecutive hits
            if not triggered:
                on_hits = on_hits + 1 if smooth >= TH_ON else 0
                if on_hits >= HITS_ON:
                    triggered = True
                    off_hits = 0
                    print("HORN DETECTED!", flush=True)
            else:
                off_hits = off_hits + 1 if smooth <= TH_OFF else 0
                if off_hits >= HITS_OFF:
                    triggered = False
                    on_hits = 0
                    print("Horn ended.", flush=True)

            ts = time.strftime("%H:%M:%S")
            state = "HORN" if triggered else "NOT "
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
