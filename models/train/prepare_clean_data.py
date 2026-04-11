import io
import os
import numpy as np
import soundfile as sf
import librosa
import random
from datasets import load_from_disk, Audio

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IN = os.path.join(SCRIPT_DIR, "data/librispeech_train_100")
VALID_IN = os.path.join(SCRIPT_DIR, "data/librispeech_val")
OUT_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")

TARGET_SR = 16000
NUM_PROC = 4
MIN_DUR = 3.0
MAX_DUR = 16.0

def process_clean(batch):
    audio_info = batch["audio"]
    
    # 1. Decode
    try:
        if audio_info.get("bytes"):
            audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
        else:
            audio, sr = sf.read(audio_info["path"])
    except Exception:
        return {"keep": False}

    # 2. Basic Signal Processing
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
    
    audio = audio.astype("float32")
    duration = len(audio) / TARGET_SR

    # 3. Filtering Logic
    # Skip if absolute silence (max amplitude is 0)
    if np.max(np.abs(audio)) < 1e-6:
        return {"keep": False}
    
    # Skip based on duration
    if not (MIN_DUR <= duration <= MAX_DUR):
        return {"keep": False}

    return {
        "clean_audio": audio,
        "clean_text": batch["text"].upper(),
        "keep": True
    }

if __name__ == "__main__":
    print("Starting Filtered Pre-processing...")
    
    # Load and map
    train_ds = load_from_disk(TRAIN_IN).cast_column("audio", Audio(decode=False))
    
    # Process and filter out the 'keep=False' rows
    train_ds = train_ds.map(process_clean, num_proc=NUM_PROC)
    train_ds = train_ds.filter(lambda x: x["keep"], num_proc=NUM_PROC)
    
    # Clean up columns
    final_cols = ["clean_audio", "clean_text"]
    train_ds = train_ds.select_columns(final_cols)

    # Save processed data
    print(f"Saving to {OUT_PATH}...")
    train_ds.save_to_disk(os.path.join(OUT_PATH, "train"))

    # --- OUTPUT 5 RANDOM SAMPLES ---
    print("Generating 5 random samples...")
    indices = random.sample(range(len(train_ds)), 5)
    
    with open(os.path.join(SCRIPT_DIR, "samples_transcript.txt"), "w") as f:
        for i, idx in enumerate(indices):
            sample = train_ds[idx]
            audio_name = f"sample_{i+1}.wav"
            audio_path = os.path.join(SCRIPT_DIR, audio_name)
            
            # Save Audio
            sf.write(audio_path, sample["clean_audio"], TARGET_SR)
            
            # Save Text to file
            f.write(f"{audio_name}: {sample['clean_text']}\n")
            
    print(f"Done! Samples and transcript saved to {SCRIPT_DIR}")