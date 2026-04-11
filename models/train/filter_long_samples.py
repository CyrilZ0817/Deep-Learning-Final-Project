import os
from datasets import load_from_disk

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")

MAX_WORDS       = 25    # drop samples with more than this many words
MAX_AUDIO_SEC   = 10  # drop samples longer than this many seconds
TARGET_SR       = 16000
NUM_PROC        = 4

def is_short_enough(example):
    word_count  = len(example["clean_text"].split())
    audio_sec   = len(example["clean_audio"]) / TARGET_SR
    return word_count <= MAX_WORDS and audio_sec <= MAX_AUDIO_SEC

if __name__ == "__main__":
    for split in ("train", "valid"):
        in_path  = os.path.join(DATA_PATH, split)
        out_path = os.path.join(DATA_PATH, f"{split}_filtered")

        print(f"\n[{split}] Loading from {in_path} ...")
        ds = load_from_disk(in_path)
        before = len(ds)

        print(f"[{split}] Filtering (max_words={MAX_WORDS}, max_sec={MAX_AUDIO_SEC}) ...")
        ds_filtered = ds.filter(is_short_enough, num_proc=NUM_PROC)
        after = len(ds_filtered)

        print(f"[{split}] {before} → {after} samples  ({before - after} removed, {100*(before-after)/before:.1f}%)")

        print(f"[{split}] Saving to {out_path} ...")
        ds_filtered.save_to_disk(out_path)
        print(f"[{split}] Done.")

    print("\nAll splits filtered and saved.")