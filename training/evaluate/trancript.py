
import os
import io
import glob
import re
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


PARQUET_DIR = r"C:\Users\72399\Desktop\final project\NOISE"
MODEL_NAME = "facebook/wav2vec2-base-960h"
OUTPUT_CSV = r"C:\Users\72399\Desktop\final project\clean_speech_transcripts_fixed.csv"

CLEAN_START = 0
CLEAN_END = 1216   # inclusive

DEBUG_PRINT_SAMPLES = 3

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

if "audio" not in dataset.column_names:
    raise ValueError("Expected column 'audio' not found.")

try:
    dataset = dataset.cast_column("audio", Audio(decode=False))
except Exception:
    pass

if CLEAN_END >= len(dataset):
    raise ValueError(f"CLEAN_END={CLEAN_END} exceeds dataset size {len(dataset)}")


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
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000
        )
        sr = 16000

    return audio, sr


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)
model.eval()



def transcribe(audio, sr):
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    pred = processor.batch_decode(pred_ids)[0]
    return normalize_text(pred)



rows = []

for idx in range(CLEAN_START, CLEAN_END + 1):
    sample = dataset[idx]

    try:
        audio, sr = load_audio_from_record(sample["audio"])
        hyp = transcribe(audio, sr)
    except Exception as e:
        print(f"Sample {idx} failed: {e}")
        hyp = ""

    row = {
        "clean_index": idx,
        "filename": sample["filename"] if "filename" in sample else f"sample_{idx}",
        "label": sample["label"] if "label" in sample else "",
        "predicted_transcript": hyp,
    }
    rows.append(row)

    if idx - CLEAN_START < DEBUG_PRINT_SAMPLES:
        print(f"\nSample {idx}")
        print("Filename:", row["filename"])
        print("Label   :", row["label"])
        print("HYP     :", hyp)

    done = idx - CLEAN_START + 1
    total = CLEAN_END - CLEAN_START + 1
    if done % 10 == 0 or idx == CLEAN_END:
        print(f"Processed {done}/{total}")



df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\nSaved transcripts to:")
print(OUTPUT_CSV)
