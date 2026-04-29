import os
import io
import glob
import re
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from jiwer import cer, wer
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


PARQUET_DIR = r"C:\Users\72399\Desktop\final project\NOISE"
TRANSCRIPT_CSV = r"C:\Users\72399\Desktop\final project\clean_speech_transcripts_fixed.csv"

PROCESSOR_SOURCE = "facebook/wav2vec2-base"

MODEL_DIRS = {
    "babble-25000":   r"C:\Users\72399\Desktop\final project\babble-checkpoint-25000-20260421T042236Z-3-001\babble-checkpoint-25000",
    "babble-28000":   r"C:\Users\72399\Desktop\final project\babble-checkpoint-28000-20260421T042237Z-3-001\babble-checkpoint-28000",
    "baseline-23500": r"C:\Users\72399\Desktop\final project\baseline-checkpoint-23500-20260421T042238Z-3-001\baseline-checkpoint-23500",
    "baseline-28000": r"C:\Users\72399\Desktop\final project\baseline-checkpoint-28000-20260421T042239Z-3-001\baseline-checkpoint-28000",
    "complex-26000":  r"C:\Users\72399\Desktop\final project\complex-checkpoint-26000-20260421T042239Z-3-001\complex-checkpoint-26000",
    "complex-28000":  r"C:\Users\72399\Desktop\final project\complex-checkpoint-28000-20260421T042240Z-3-001\complex-checkpoint-28000",
    "static-25000":   r"C:\Users\72399\Desktop\final project\static-checkpoint-25000-20260421T042241Z-3-001\static-checkpoint-25000",
    "static-28000":   r"C:\Users\72399\Desktop\final project\static-checkpoint-28000-20260421T042241Z-3-001\static-checkpoint-28000",
}

# clean / noisy 对应区间
CLEAN_START = 0
CLEAN_END = 1216

NOISY_START = 1217
NOISY_END = 2433

DEBUG_PRINT_SAMPLES = 3
OUTPUT_CSV = r"C:\Users\72399\Desktop\final project\noisy_eval_results_8models_newpaths.csv"


def normalize_text(text: str) -> str:
    text = str(text).upper().strip()
    text = re.sub(r"[\r\n\t]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
if len(parquet_files) == 0:
    raise FileNotFoundError(f"No parquet files found in: {PARQUET_DIR}")

print(f"Found {len(parquet_files)} parquet files.")

dataset = load_dataset(
    "parquet",
    data_files=parquet_files,
    split="train"
)

print("Columns:", dataset.column_names)
print("Dataset size:", len(dataset))

for col in ["audio", "filename", "label"]:
    if col not in dataset.column_names:
        raise ValueError(f"Expected column '{col}' not found.")

try:
    dataset = dataset.cast_column("audio", Audio(decode=False))
except Exception:
    pass

if NOISY_END >= len(dataset):
    raise ValueError(f"NOISY_END={NOISY_END} exceeds dataset size {len(dataset)}")



if not os.path.exists(TRANSCRIPT_CSV):
    raise FileNotFoundError(f"Transcript CSV not found: {TRANSCRIPT_CSV}")

transcript_df = pd.read_csv(TRANSCRIPT_CSV)

required_cols = {"clean_index", "predicted_transcript"}
missing = required_cols - set(transcript_df.columns)
if missing:
    raise ValueError(f"TRANSCRIPT_CSV missing required columns: {missing}")

transcript_df["predicted_transcript"] = (
    transcript_df["predicted_transcript"]
    .fillna("")
    .map(normalize_text)
)

transcript_map = dict(zip(transcript_df["clean_index"], transcript_df["predicted_transcript"]))

print("Loaded transcript rows:", len(transcript_df))


def load_audio_from_record(record_value):
    if isinstance(record_value, dict):
        if "bytes" in record_value and record_value["bytes"] is not None:
            audio, sr = sf.read(io.BytesIO(record_value["bytes"]))
        elif "path" in record_value and record_value["path"] is not None:
            audio, sr = sf.read(record_value["path"])
        elif "array" in record_value and record_value["array"] is not None:
            audio = np.asarray(record_value["array"], dtype=np.float32)
            sr = record_value.get("sampling_rate", 16000)
        else:
            raise ValueError(f"Unsupported dict audio format: {record_value.keys()}")
    elif isinstance(record_value, str):
        audio, sr = sf.read(record_value)
    else:
        audio = np.asarray(record_value, dtype=np.float32)
        sr = 16000

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype("float32")

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    return audio, sr


samples = []

for noisy_idx in range(NOISY_START, NOISY_END + 1):
    clean_idx = noisy_idx - NOISY_START + CLEAN_START

    if clean_idx not in transcript_map:
        print(f"Warning: transcript missing for clean index {clean_idx}")
        continue

    ref = transcript_map[clean_idx]
    if ref == "":
        print(f"Warning: empty transcript for clean index {clean_idx}")
        continue

    row = dataset[noisy_idx]

    samples.append({
        "noisy_index": noisy_idx,
        "clean_index": clean_idx,
        "filename": row["filename"],
        "label": row["label"],
        "audio_record": row["audio"],
        "reference_text": ref,
    })

print("Usable matched noisy samples:", len(samples))

if len(samples) == 0:
    raise RuntimeError("No usable matched noisy samples found.")


print("\nLoading shared processor...")
processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_SOURCE)


def run_asr(processor, model, audio, sr):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    pred = processor.batch_decode(pred_ids)[0]
    return normalize_text(pred)



def evaluate_model(model_name, model, samples):
    cer_scores = []
    wer_scores = []

    for i, sample in enumerate(samples, start=1):
        ref = sample["reference_text"]

        try:
            audio, sr = load_audio_from_record(sample["audio_record"])
            hyp = run_asr(processor, model, audio, sr)
        except Exception as e:
            print(f"[{model_name}] failed on {sample['filename']}: {e}")
            hyp = ""

        cer_scores.append(cer(ref, hyp))
        wer_scores.append(wer(ref, hyp))

        if i <= DEBUG_PRINT_SAMPLES:
            print(f"\n[{model_name}] Sample {i}")
            print("Noisy index :", sample["noisy_index"])
            print("Clean index :", sample["clean_index"])
            print("Filename    :", sample["filename"])
            print("Label       :", sample["label"])
            print("REF         :", ref)
            print("HYP         :", hyp)

        if i % 10 == 0 or i == len(samples):
            print(f"{model_name}: processed {i}/{len(samples)}")

    return {
        "cer": float(np.mean(cer_scores)),
        "wer": float(np.mean(wer_scores)),
    }



results = []

for model_name, model_dir in MODEL_DIRS.items():
    print(f"\n===== {model_name} =====")

    if not os.path.exists(model_dir):
        print(f"Skip: model not found -> {model_dir}")
        continue

    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_dir)
        model.eval()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        continue

    metrics = evaluate_model(model_name, model, samples)

    results.append({
        "model": model_name,
        "cer": metrics["cer"],
        "wer": metrics["wer"],
    })



results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\n===== FINAL RESULTS =====")
for _, row in results_df.iterrows():
    print(row["model"])
    print(f"CER: {row['cer']:.4f}")
    print(f"WER: {row['wer']:.4f}\n")

print("Saved to:", OUTPUT_CSV)