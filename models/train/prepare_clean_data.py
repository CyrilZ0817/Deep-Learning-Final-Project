import io
import os
import numpy as np
import soundfile as sf
import yaml
import librosa
from datasets import load_from_disk, Audio

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IN = os.path.join(SCRIPT_DIR, "data/librispeech_train_100")
VALID_IN = os.path.join(SCRIPT_DIR, "data/librispeech_val")
OUT_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")

TARGET_SR = 16000
NUM_PROC = 4

def process_clean(batch):
    audio_info = batch["audio"]
    # 1. Decode
    try:
        if audio_info.get("bytes"):
            audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
        else:
            audio, sr = sf.read(audio_info["path"])
    except Exception:
        audio, sr = np.zeros(TARGET_SR), TARGET_SR

    # 2. Mono & Resample
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
    
    # We store the float array and the normalized text
    batch["clean_audio"] = audio.astype("float32")
    batch["clean_text"] = batch["text"].upper()
    return batch

if __name__ == "__main__":
    print("Starting Clean Pre-processing (16kHz)...")
    train_ds = load_from_disk(TRAIN_IN).cast_column("audio", Audio(decode=False))
    valid_ds = load_from_disk(VALID_IN).cast_column("audio", Audio(decode=False))

    # We remove the original 'audio' and 'text' to save space
    cols_to_remove = ["audio", "text"]

    train_ds = train_ds.map(process_clean, remove_columns=cols_to_remove, num_proc=NUM_PROC)
    valid_ds = valid_ds.map(process_clean, remove_columns=cols_to_remove, num_proc=NUM_PROC)

    print(f"Saving to {OUT_PATH}...")
    train_ds.save_to_disk(os.path.join(OUT_PATH, "train"))
    valid_ds.save_to_disk(os.path.join(OUT_PATH, "valid"))
    print("Done! Clean data is ready.")