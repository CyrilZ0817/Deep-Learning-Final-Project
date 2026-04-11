import os
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from datasets import load_from_disk

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")
TARGET_SR  = 16000
OUTPUT_FILE = "audio_distribution_debug.png"
LOG_INTERVAL = 100
def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

def plot_distribution():
    splits = ["train", "valid"]
    fig, axes = plt.subplots(len(splits), 1, figsize=(12, 10), sharex=True)
    bins = np.arange(0, 24, 2) 

    for i, split in enumerate(splits):
        ds_path = os.path.join(DATA_PATH, split)
        if not os.path.exists(ds_path):
            log(f"SKIPPING: {ds_path} not found.")
            continue
            
        log(f"LOADING: '{split}' split metadata...")
        ds = load_from_disk(ds_path)
        total_samples = len(ds)
        log(f"SUCCESS: Loaded {total_samples} samples metadata.")

        durations = []
        start_time = time.time()

        log(f"PROCESSING: Calculating durations for {total_samples} samples...")
        
        # Using a loop instead of list comprehension for progress tracking
        for idx, example in enumerate(ds):
            # The actual disk read happens HERE when we access ["clean_audio"]
            dur = len(example["clean_audio"]) / TARGET_SR
            durations.append(dur)

            if (idx + 1) % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                percent = ((idx + 1) / total_samples) * 100
                speed = (idx + 1) / elapsed
                log(f"  > Progress: {idx+1}/{total_samples} ({percent:.1f}%) | Speed: {speed:.1f} samples/sec")

        total_elapsed = time.time() - start_time
        log(f"FINISHED: '{split}' processing took {total_elapsed:.2f} seconds.")

        # Plotting logic
        axes[i].hist(durations, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(f"{split.capitalize()} Distribution ({total_samples} samples)")
        axes[i].set_ylabel("Count")
        
        counts, _ = np.histogram(durations, bins=bins)
        for j, count in enumerate(counts):
            if count > 0:
                axes[i].text(bins[j] + 1, count, str(count), ha='center', va='bottom')

    plt.xlabel("Duration (seconds)")
    plt.tight_layout()
    
    save_path = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
    plt.savefig(save_path)
    log(f"DONE: Plot saved to {save_path}")

if __name__ == "__main__":
    log("Starting Audio Distribution Script with Debugging...")
    plot_distribution()