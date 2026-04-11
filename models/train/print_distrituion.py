import os
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")
TARGET_SR  = 16000
OUTPUT_FILE = "audio_distribution.png"

def plot_distribution():
    splits = ["train", "valid"]
    fig, axes = plt.subplots(len(splits), 1, figsize=(12, 10), sharex=True)
    
    # Define bins: 0, 2, 4, 6, ... up to 22+
    # We create bins up to 24 to capture the "22+" samples nicely
    bins = np.arange(0, 24, 2) 

    for i, split in enumerate(splits):
        ds_path = os.path.join(DATA_PATH, split)
        if not os.path.exists(ds_path):
            print(f"Warning: {ds_path} not found. Skipping...")
            continue
            
        print(f"Loading {split} split for plotting...")
        ds = load_from_disk(ds_path)
        
        # Calculate durations in seconds
        durations = [len(x["clean_audio"]) / TARGET_SR for x in ds]
        
        # Plotting
        axes[i].hist(durations, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(f"Audio Duration Distribution: {split.capitalize()} ({len(ds)} samples)")
        axes[i].set_ylabel("Count")
        axes[i].set_xticks(bins)
        
        # Add labels on top of bars
        counts, _ = np.histogram(durations, bins=bins)
        for j, count in enumerate(counts):
            if count > 0:
                axes[i].text(bins[j] + 1, count, str(count), ha='center', va='bottom', fontsize=9)

    plt.xlabel("Duration (seconds)")
    plt.tight_layout()
    
    save_path = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
    plt.savefig(save_path)
    print(f"Distribution plot saved to: {save_path}")

if __name__ == "__main__":
    plot_distribution()