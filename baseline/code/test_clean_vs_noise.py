import io
import torch
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import cer

# ========= 1. load clean sample =========
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

text = sample["text"]

# ========= 2. load model =========
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# ========= 3. mix function =========
def rms(x):
    return np.sqrt(np.mean(x**2) + 1e-8)

def mix(clean, noise, snr_db):
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean)/len(noise))))
    noise = noise[:len(clean)]

    clean_r = rms(clean)
    noise_r = rms(noise)

    target_noise_r = clean_r / (10**(snr_db/20))
    noise = noise * (target_noise_r / (noise_r + 1e-8))

    mixed = clean + noise
    mixed = mixed / np.max(np.abs(mixed))

    return mixed

# ========= 4. ASR function =========
def run_asr(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]

# ========= 5. test clean =========
pred_clean = run_asr(clean)
cer_clean = cer(text, pred_clean)

print("\n=== CLEAN ===")
print("GT :", text)
print("PR :", pred_clean)
print("CER:", cer_clean)

# ========= 6. test noise =========
noise_files = {
    "stationary": "noises/AirConditioner_2.wav",
    "nonstationary": "noises/AirportAnnouncements_1.wav",
    "multispeaker": "noises/Babble_1.wav"
}

for name, path in noise_files.items():
    noise, sr2 = sf.read(path)

    if len(noise.shape) > 1:
        noise = noise.mean(axis=1)

    mixed = mix(clean, noise, snr_db=10)

    pred = run_asr(mixed)
    c = cer(text, pred)

    print(f"\n=== {name.upper()} (10dB) ===")
    print("PR :", pred)
    print("CER:", c)