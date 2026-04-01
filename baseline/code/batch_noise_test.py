import io
import torch
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import cer

# 1. load dataset
num_samples = 100

dataset = load_dataset("librispeech_asr", "clean", split=f"validation[:{num_samples}]")
dataset = dataset.cast_column("audio", Audio(decode=False))

# 2. load model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# 3. helper functions
def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-8)

def mix(clean, noise, snr_db):
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]

    clean_r = rms(clean)
    noise_r = rms(noise)

    target_noise_r = clean_r / (10 ** (snr_db / 20))
    noise = noise * (target_noise_r / (noise_r + 1e-8))

    mixed = clean + noise
    max_val = np.max(np.abs(mixed))
    if max_val > 1:
        mixed = mixed / max_val

    return mixed

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

    return audio, sr

def load_noise(path):
    noise, sr = sf.read(path)

    if len(noise.shape) > 1:
        noise = noise.mean(axis=1)

    if sr != 16000:
        import librosa
        noise = librosa.resample(noise.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000

    return noise

def run_asr(audio, sr=16000):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred = processor.batch_decode(pred_ids)[0]
    return pred

# 4. load one noise file per category
noise_files = {
    "clean": None,
    "stationary": "noises/AirConditioner_2.wav",
    "nonstationary": "noises/AirportAnnouncements_1.wav",
    "multispeaker": "noises/Babble_1.wav"
}

loaded_noises = {}
for name, path in noise_files.items():
    if path is not None:
        loaded_noises[name] = load_noise(path)

# 5. evaluate
results = {k: [] for k in noise_files.keys()}
snr_db = 10

for i in range(num_samples):
    sample = dataset[i]
    clean_audio, sr = load_audio_from_sample(sample)
    gt = sample["text"]

    # clean
    pred_clean = run_asr(clean_audio, sr)
    results["clean"].append(cer(gt, pred_clean))

    # noisy
    for noise_type in ["stationary", "nonstationary", "multispeaker"]:
        mixed = mix(clean_audio, loaded_noises[noise_type], snr_db=snr_db)
        pred = run_asr(mixed, sr)
        results[noise_type].append(cer(gt, pred))

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{num_samples}")

# 6. print summary
print("\n===== Average CER Results =====")
for k, v in results.items():
    print(f"{k:15s}: {np.mean(v):.4f}")

    import matplotlib.pyplot as plt

labels = list(results.keys())
values = [np.mean(results[k]) for k in labels]

plt.figure()
plt.bar(labels, values)
plt.xlabel("Noise Type")
plt.ylabel("Average CER")
plt.title("Impact of Noise Type on ASR Performance")
plt.tight_layout()

plt.savefig("cer_comparison.png")
plt.show()
