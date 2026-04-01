import io
import numpy as np
import soundfile as sf
import torch

from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import cer

# 1. settings
MODEL_DIR = "./wav2vec2-multispeaker-10db"
NUM_SAMPLES = 100
SNR_DB = 10

NOISE_FILES = {
    "clean": None,
    "stationary": "noises/AirConditioner_2.wav",
    "nonstationary": "noises/AirportAnnouncements_1.wav",
    "multispeaker": "noises/Babble_1.wav",
}

# 2. load dataset
dataset = load_dataset("librispeech_asr", "clean", split=f"validation[:{NUM_SAMPLES}]")
dataset = dataset.cast_column("audio", Audio(decode=False))

# 3. load trained model
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
model.eval()

# 4. helper functions

def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-8)

def mix(clean, noise, snr_db):
    if len(noise) < len(clean):
        repeat_times = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeat_times)

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

def load_audio_from_sample(sample):
    audio_info = sample["audio"]

    if audio_info["bytes"] is not None:
        audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
    else:
        audio, sr = sf.read(audio_info["path"])

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000

    return audio.astype("float32"), sr

def load_noise(path):
    noise, sr = sf.read(path)

    if len(noise.shape) > 1:
        noise = noise.mean(axis=1)

    if sr != 16000:
        import librosa
        noise = librosa.resample(noise.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000

    return noise.astype("float32")

def run_asr(audio, sr=16000):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred = processor.batch_decode(pred_ids)[0]
    return pred

# 5. preload noises
loaded_noises = {}
for noise_type, path in NOISE_FILES.items():
    if path is not None:
        loaded_noises[noise_type] = load_noise(path)

# 6. evaluate
results = {k: [] for k in NOISE_FILES.keys()}

for i in range(NUM_SAMPLES):
    sample = dataset[i]
    clean_audio, sr = load_audio_from_sample(sample)
    gt = sample["text"]

    pred_clean = run_asr(clean_audio, sr)
    results["clean"].append(cer(gt, pred_clean))

    for noise_type in ["stationary", "nonstationary", "multispeaker"]:
        mixed = mix(clean_audio, loaded_noises[noise_type], snr_db=SNR_DB)
        pred = run_asr(mixed, sr)
        results[noise_type].append(cer(gt, pred))

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{NUM_SAMPLES}")

# 7. print summary
print(f"\n===== Multispeaker-Trained Model Average CER @ {SNR_DB} dB =====")
for k, v in results.items():
    print(f"{k:15s}: {np.mean(v):.4f}")