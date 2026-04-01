import io
import torch
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import cer

# 1. load dataset
dataset = load_dataset("librispeech_asr", "clean", split="validation[:20]")
dataset = dataset.cast_column("audio", Audio(decode=False))

# 2. load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

total_cer = 0.0
num_samples = 5

for i in range(num_samples):
    sample = dataset[i]
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
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    prediction = processor.batch_decode(pred_ids)[0]

    ground_truth = sample["text"]

    sample_cer = cer(ground_truth, prediction)
    total_cer += sample_cer

    print(f"\nSample {i+1}")
    print("Ground truth:", ground_truth)
    print("Prediction  :", prediction)
    print("CER         :", sample_cer)

# 6. average CER
avg_cer = total_cer / num_samples
print("\n==============================")
print("Average CER:", avg_cer)
print("==============================")