import io
import json
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import cer


# 1. settings
num_samples = 100
snr_levels = [20, 10, 0]

noise_files = {
    "stationary": "noises/AirConditioner_2.wav",
    "nonstationary": "noises/AirportAnnouncements_1.wav",
    "multispeaker": "noises/Babble_1.wav",
}


# 2. load dataset
dataset = load_dataset("librispeech_asr", "clean", split=f"validation[:{num_samples}]")
dataset = dataset.cast_column("audio", Audio(decode=False))


# 3. load model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
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
for noise_type, path in noise_files.items():
    loaded_noises[noise_type] = load_noise(path)


# 6. evaluate
results = {noise_type: {snr: [] for snr in snr_levels} for noise_type in noise_files}

for i in range(num_samples):
    sample = dataset[i]
    clean_audio, sr = load_audio_from_sample(sample)
    gt = sample["text"]

    for noise_type in noise_files:
        noise = loaded_noises[noise_type]

        for snr in snr_levels:
            mixed = mix(clean_audio, noise, snr_db=snr)
            pred = run_asr(mixed, sr)
            sample_cer = cer(gt, pred)
            results[noise_type][snr].append(sample_cer)

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{num_samples}")

# 7. summarize
summary = {}
print("\n===== Average CER by Noise Type and SNR =====")

for noise_type in noise_files:
    summary[noise_type] = {}
    print(f"\n{noise_type.upper()}")
    for snr in snr_levels:
        avg_cer = float(np.mean(results[noise_type][snr]))
        summary[noise_type][snr] = avg_cer
        print(f"  SNR {snr:>2} dB : {avg_cer:.4f}")

# 8. save JSON
with open("snr_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved snr_results.json")


# 9. plot
plt.figure()

for noise_type in noise_files:
    y = [summary[noise_type][snr] for snr in snr_levels]
    plt.plot(snr_levels, y, marker="o", label=noise_type)

plt.xlabel("SNR (dB)")
plt.ylabel("Average CER")
plt.title("ASR Performance under Different Noise Types and SNR Levels")
plt.gca().invert_xaxis()  # 20 -> 10 -> 0 visually goes worse left-to-right if you prefer reversed; remove if unwanted
plt.legend()
plt.tight_layout()

plt.savefig("snr_analysis.png", dpi=300)
plt.show()

print("Saved snr_analysis.png")