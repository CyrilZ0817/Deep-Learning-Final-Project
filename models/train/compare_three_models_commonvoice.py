import os
import csv
import numpy as np
import soundfile as sf
import torch

from jiwer import cer, wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


COMMONVOICE_DIR = r"C:\Users\72399\Desktop\final project\cv-corpus-25.0-2026-03-09\en"
TSV_PATH = os.path.join(COMMONVOICE_DIR, "test.tsv")
CLIPS_DIR = os.path.join(COMMONVOICE_DIR, "clips")

# 你训练出来的 8h model 路径
MODEL_8H_DIR = r"C:\Users\72399\Desktop\final project\wav2vec2-base-from-librispeech"

# 预训练 960h model
MODEL_960H_NAME = "facebook/wav2vec2-base-960h"

# 完全 clean 的 base
MODEL_BASE_NAME = "facebook/wav2vec2-base"

# tokenizer / processor 借用 960h 版本
PROCESSOR_NAME = "facebook/wav2vec2-base-960h"

# 可调：先测前多少条，跑通后再加大
MAX_SAMPLES = 100


def load_commonvoice_samples(tsv_path, clips_dir, max_samples=None):
    samples = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = row.get("sentence", "").strip()
            path = row.get("path", "").strip()

            if not sentence or not path:
                continue

            audio_path = os.path.join(clips_dir, path)
            if not os.path.exists(audio_path):
                continue

            samples.append({
                "audio_path": audio_path,
                "text": sentence.upper().strip()
            })

            if max_samples is not None and len(samples) >= max_samples:
                break

    return samples


def load_audio(audio_path):
    audio, sr = sf.read(audio_path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype("float32")

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    return audio, sr


def normalize_text(s):
    return str(s).upper().strip()


def run_asr(processor, model, audio, sr=16000):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    pred = processor.batch_decode(pred_ids)[0]
    return normalize_text(pred)


def evaluate_model(model_name, processor, model, samples):
    cer_scores = []
    wer_scores = []

    for i, sample in enumerate(samples, start=1):
        audio, sr = load_audio(sample["audio_path"])
        ref = normalize_text(sample["text"])

        try:
            hyp = run_asr(processor, model, audio, sr)
        except Exception as e:
            print(f"[{model_name}] sample {i} failed: {e}")
            hyp = ""

        cer_scores.append(cer(ref, hyp))
        wer_scores.append(wer(ref, hyp))

        if i % 10 == 0:
            print(f"{model_name}: processed {i}/{len(samples)}")

    return {
        "cer": float(np.mean(cer_scores)),
        "wer": float(np.mean(wer_scores)),
    }


samples = load_commonvoice_samples(TSV_PATH, CLIPS_DIR, MAX_SAMPLES)
print(f"Loaded {len(samples)} Common Voice test samples.")


processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)


print("\n===== Loading clean base model =====")
base_model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_BASE_NAME,
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_loss_reduction="mean",
    ignore_mismatched_sizes=True,
)
base_model.eval()


print("\n===== Loading 8h model =====")
model_8h = Wav2Vec2ForCTC.from_pretrained(MODEL_8H_DIR)
model_8h.eval()


print("\n===== Loading 960h model =====")
model_960h = Wav2Vec2ForCTC.from_pretrained(MODEL_960H_NAME)
model_960h.eval()


results = {}

print("\n===== Evaluating clean base =====")
results["clean_base"] = evaluate_model("clean_base", processor, base_model, samples)

print("\n===== Evaluating 8h model =====")
results["8h_model"] = evaluate_model("8h_model", processor, model_8h, samples)

print("\n===== Evaluating 960h model =====")
results["960h_model"] = evaluate_model("960h_model", processor, model_960h, samples)


print("\n===== Common Voice Real-World Evaluation =====")
for name, metrics in results.items():
    print(f"\n{name}")
    print(f"CER: {metrics['cer']:.4f}")
    print(f"WER: {metrics['wer']:.4f}")