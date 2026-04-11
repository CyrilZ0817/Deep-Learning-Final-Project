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
OUT_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")

TARGET_SR = 16000
NUM_PROC = 4
MIN_DUR = 3.0
MAX_DUR = 16.0

def process_clean(batch):
    result = {
        "clean_audio": None,
        "clean_text": "",
        "keep": False
    }

    try:
        # 1. Validate Keys
        if "audio" not in batch or "text" not in batch:
            return result

        audio_info = batch["audio"]
        
        # 2. Decode Audio
        if audio_info.get("bytes"):
            audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
        elif audio_info.get("path") and os.path.exists(audio_info["path"]):
            audio, sr = sf.read(audio_info["path"])
        else:
            return result

        # 3. Mono & Resample
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
        
        audio = audio.astype("float32")
        duration = len(audio) / TARGET_SR

        # 4. Filtering Logic
        if np.max(np.abs(audio)) < 1e-6:
            return result
        
        if not (MIN_DUR <= duration <= MAX_DUR):
            return result

        # Success
        result["clean_audio"] = audio
        result["clean_text"] = batch["text"].upper()
        result["keep"] = True

    except Exception:
        return result

    return result

if __name__ == "__main__":
    print("Starting Clean Pre-processing...")
    
    # Load dataset
    ds = load_from_disk(TRAIN_IN).cast_column("audio", Audio(decode=False))
    initial_count = len(ds)
    print(f"Initial dataset size: {initial_count}")
    
    # Process
    ds = ds.map(process_clean, num_proc=NUM_PROC)
    
    # Filter
    print("Filtering rows...")
    final_ds = ds.filter(lambda x: x["keep"], num_proc=NUM_PROC)
    
    # Calculate stats
    final_count = len(final_ds)
    removed_count = initial_count - final_count
    pass_percentage = (final_count / initial_count) * 100

    # Cleanup columns
    final_ds = final_ds.select_columns(["clean_audio", "clean_text"])

    # Summary Output
    print("-" * 30)
    print("PRE-PROCESSING SUMMARY")
    print(f"Total processed: {initial_count}")
    print(f"Samples passed:  {final_count}")
    print(f"Samples removed: {removed_count}")
    print(f"Success rate:    {pass_percentage:.2f}%")
    print("-" *