import os
import subprocess
import numpy as np
import librosa
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "horn_cnn_best.keras")
NORM_STATS_PATH = os.path.join(BASE_DIR, "norm_stats.npz")
REC_WAV_PATH = os.path.join(BASE_DIR, "recorded.wav")
PLOT_PATH = os.path.join(BASE_DIR, "horn_probability_curve.png")

# -------------------------
# AUDIO / FEATURE CONFIG (MATCH TRAINING)
# -------------------------
TARGET_SR = 44100
DURATION = 1.0
N_SAMPLES = int(TARGET_SR * DURATION)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
POWER = 2.0

# Sliding evaluation
RECORD_SECONDS = 20.0
STEP_SEC = 0.25
HORN_THRESHOLD = 0.70
EMA_ALPHA = 0.4
SILENCE_RMS = 0.005

# -------------------------
# I2S MIC CAPTURE SETTINGS (Pi)
# -------------------------
ARECORD_DEVICE = "plughw:CARD=sndrpigooglevoi,DEV=0"
CAPTURE_SR = 48000  # I2S voicehat often runs 48k reliably


def audio_to_logmel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=POWER,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)


def record_with_arecord(duration_sec: float, out_wav: str):
    print(f"Recording {duration_sec:.1f}s from I2S mic -> {out_wav}")
    cmd = [
        "arecord",
        "-D", ARECORD_DEVICE,
        "-c", "1",                 # mono
        "-r", str(CAPTURE_SR),      # capture rate
        "-f", "S16_LE",             # easier for loaders
        "-t", "wav",
        "-d", str(int(duration_sec)),
        out_wav
    ]
    subprocess.run(cmd, check=True)
    print("Recording complete.")


def load_audio_resample(wav_path: str) -> np.ndarray:
    # Load and resample to TARGET_SR so features match training
    audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)

def evaluate_clip(audio: np.ndarray, model, mean: float, std: float):
    window = N_SAMPLES
    step = int(STEP_SEC * TARGET_SR)

    times = []
    probs = []
    smooth_probs = []

    smooth = 0.0
    for start in range(0, len(audio) - window + 1, step):
        seg = audio[start:start + window]
        rms = float(np.sqrt(np.mean(seg ** 2)))

        if rms < SILENCE_RMS:
            prob = 0.0
        else:
            mel = audio_to_logmel(seg)
            mel_norm = (mel - mean) / (std + 1e-9)
            mel_norm = mel_norm[np.newaxis, ..., np.newaxis]  # (1,H,W,1)
            prob = float(model.predict(mel_norm, verbose=0)[0][0])

        smooth = EMA_ALPHA * prob + (1.0 - EMA_ALPHA) * smooth
        t_sec = start / TARGET_SR

        times.append(t_sec)
        probs.append(prob)
        smooth_probs.append(smooth)

    return np.array(times), np.array(probs), np.array(smooth_probs)


def plot_probs(times: np.ndarray, probs: np.ndarray, smooth_probs: np.ndarray, save_path: str):
    # If running headless (SSH), force non-GUI backend
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")

    plt.figure(figsize=(10, 4))
    plt.plot(times, probs, label="prob", alpha=0.5)
    plt.plot(times, smooth_probs, label="smooth", linewidth=2)
    plt.axhline(HORN_THRESHOLD, linestyle="--", label="threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title("Horn probability over recorded clip")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot -> {save_path}")

    # If GUI is available, also show
    if os.environ.get("DISPLAY", "") != "":
        plt.show()


def main():
    # Load norm
    stats = np.load(NORM_STATS_PATH)
    mean = float(np.array(stats["mean"]).reshape(-1)[0])
    std = float(np.array(stats["std"]).reshape(-1)[0])
    print(f"Loaded norm: mean={mean:.6f}, std={std:.6f}")

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded model:", model.input_shape, "->", model.output_shape)

    # Record audio using I2S mic
    record_with_arecord(RECORD_SECONDS, REC_WAV_PATH)

    # Load + resample to match training
    audio = load_audio_resample(REC_WAV_PATH)

    # Evaluate
    times, probs, smooth_probs = evaluate_clip(audio, model, mean, std)

    max_prob = float(np.max(probs)) if len(probs) else 0.0
    max_smooth = float(np.max(smooth_probs)) if len(smooth_probs) else 0.0
    horn_detected = max_smooth >= HORN_THRESHOLD

    print(f"Windows evaluated: {len(probs)}")
    print(f"Max prob: {max_prob:.3f}, Max smooth: {max_smooth:.3f}")
    print("Decision: HORN DETECTED" if horn_detected else "Decision: no horn")

    # Plot (saved to PNG)
    plot_probs(times, probs, smooth_probs, PLOT_PATH)


if __name__ == "__main__":
    main()
