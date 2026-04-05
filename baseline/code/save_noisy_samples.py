import io
import os
import numpy as np
import soundfile as sf

from datasets import load_dataset, Audio

NOISE_DIR = "noises"
OUTPUT_DIR = "audio_demo"
SNR_LEVELS = [20, 10, 0]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-8)

def mix(clean, noise, snr_db):
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]

    clean_rms = rms(clean)
    noise_rms = rms(noise)

    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / (noise_rms + 1e-8))

    mixed = clean + noise

    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed.astype("float32")

dataset = load_dataset("librispeech_asr", "clean", split="validation[:1]")
dataset = dataset.cast_column("audio", Audio(decode=False))

sample = dataset[0]
audio_info = sample["audio"]

if audio_info["bytes"] is not None:
    clean, sr = sf.read(io.BytesIO(audio_info["bytes"]))
else:
    clean, sr = sf.read(audio_info["path"])

if len(clean.shape) > 1:
    clean = clean.mean(axis=1)

sf.write(os.path.join(OUTPUT_DIR, "clean.wav"), clean, sr)

for noise_file in os.listdir(NOISE_DIR):
    if not noise_file.endswith(".wav"):
        continue

    noise_path = os.path.join(NOISE_DIR, noise_file)

    try:
        noise, sr_noise = sf.read(noise_path)
    except:
        continue

    if len(noise.shape) > 1:
        noise = noise.mean(axis=1)

    for snr in SNR_LEVELS:
        mixed = mix(clean, noise, snr)

        out_name = f"{noise_file.replace('.wav','')}_snr{snr}.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        sf.write(out_path, mixed, sr)

print("Done. Check audio_demo/ folder.")