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
    # Initialize the return dictionary with placeholders to keep schema consistent
    result = {
        "clean_audio": None,
        "clean_text": "",
        "keep": False
    }

    try:
        # 1. Validate Keys
        if "audio" not in batch or "text" not in batch:
            # Find which key is missing for the print statement
            missing = [k for k in ["audio", "text"] if k not in batch]
            print(f"[SKIP] Missing keys {missing} in row.")
            return result

        audio_info = batch["audio"]
        
        # 2. Decode Audio
        if audio_info.get("bytes"):
            audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
        elif audio_info.get("path") and os.path.exists(audio_info["path"]):
            audio, sr = sf.read(audio_info["path"])
        else:
            print(f"[SKIP] Audio path invalid or bytes empty: {audio_info.get('path')}")
            return result

        # 3. Mono & Resample
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
        
        audio = audio.astype("float32")
        duration = len(audio) / TARGET_SR

        # 4. Filtering Logic (Silence & Duration)
        if np.max(np.abs(audio)) < 1e-6:
            print(f"[SKIP] Absolute silence detected.")
            return result
        
        if not (MIN_DUR <= duration <= MAX_DUR):
            # Optional: print(f"[SKIP] Duration {duration:.2f}s out of bounds.")
            return result

        # If we made it here, the row is "good"
        result["clean_audio"] = audio
        result["clean_text"] = batch["text"].upper()
        result["keep"] = True

    except Exception as e:
        print(f"[ERROR] Unexpected error processing row: {e}")
        return result

    return result

if __name__ == "__main__":
    print("Starting Clean Pre-processing...")
    
    # Load and cast
    ds = load_from_disk(TRAIN_IN).cast_column("audio", Audio(decode=False))
    
    # Map: Processes every row and marks 'keep' as True/False
    # Note: printing inside map with num_proc can be messy, but it works for debugging
    ds = ds.map(process_clean, num_proc=NUM_PROC)
    
    # Filter: Physically removes the rows where keep is False
    print("Filtering rows...")
    ds = ds.filter(lambda x: x["keep"], num_proc=NUM_PROC)
    
    # Cleanup: Remove helper columns and original data
    final_ds = ds.select_columns(["clean_audio", "clean_text"])

    print(f"Saving to {OUT_PATH}...")
    final_ds.save_to_disk(os.path.join(OUT_PATH, "train"))

    # --- OUTPUT 5 RANDOM SAMPLES ---
    if len(final_ds) > 5:
        print("Saving 5 random samples for verification...")
        indices = random.sample(range(len(final_ds)), 5)
        
        with open(os.path.join(SCRIPT_DIR, "samples_transcript.txt"), "w") as f:
            for i, idx in enumerate(indices):
                sample = final_ds[idx]
                audio_name = f"sample_{i+1}.wav"
                sf.write(os.path.join(SCRIPT_DIR, audio_name), sample["clean_audio"], TARGET_SR)
                f.write(f"{audio_name}: {sample['clean_text']}\n")
        print(f"Samples saved to {SCRIPT_DIR}")
    else:
        print("Not enough samples remaining after filtering to save 5 verification files.")