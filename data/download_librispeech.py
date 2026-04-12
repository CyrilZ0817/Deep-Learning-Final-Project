import os
import shutil

# --- 1. DIRECTORY SETUP ---
# This is your large 100GB storage area
PROJECT_DIR = "/common/users/sk2779/Deep-Learning-Final-Project"
CACHE_DIR = os.path.join(PROJECT_DIR, "hf_cache")
TMP_DIR = os.path.join(PROJECT_DIR, "tmp")

# Create these directories immediately
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# --- 2. ENVIRONMENT REDIRECTION (The "Triple Lock") ---
# We set these BEFORE importing datasets to ensure no "leaks" to your home dir
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR
os.environ["TMPDIR"] = TMP_DIR

# --- 3. IMPORTS ---
from datasets import load_dataset

def main():
    print(f"Starting download. All files will be redirected to: {PROJECT_DIR}")
    # --- DOWNLOAD TRAIN-CLEAN-100 ---
    print("\nStep 1: Downloading Librispeech train.100...")
    train_ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100",
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )

    train_output_path = os.path.join(PROJECT_DIR, "librispeech_train_100")
    print(f"Saving Train set to disk at: {train_output_path}")
    train_ds.save_to_disk(train_output_path)
    print("Train set saved successfully.")

    # --- DOWNLOAD VALIDATION ---
    print("\nStep 2: Downloading Librispeech validation...")
    val_ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    val_output_path = os.path.join(PROJECT_DIR, "librispeech_val")
    print(f"Saving Validation set to disk at: {val_output_path}")
    val_ds.save_to_disk(val_output_path)
    print("Validation set saved successfully.")

    print("\n--- ALL DONE! ---")
    print(f"Your datasets are ready in: {PROJECT_DIR}")
    print(f"Train: {train_output_path}")
    print(f"Valid: {val_output_path}")

if __name__ == "__main__":
    main()