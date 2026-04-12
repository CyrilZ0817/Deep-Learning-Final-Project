import io
import os
import numpy as np
import soundfile as sf
import librosa
import random
from datasets import load_from_disk, Audio
from transformers import Wav2Vec2Processor

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IN = os.path.join(SCRIPT_DIR, "data/librispeech_train_100")
OUT_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")

TARGET_SR = 16000
NUM_PROC = 4
MIN_DUR = 3.0
MAX_DUR = 16.0

# Fix 1: raised silence threshold (was 1e-6, too permissive)
MIN_AMPLITUDE = 0.01

# Fix 2: CTC feasibility margin — output frames must be >= labels * this factor
CTC_SAFETY_MARGIN = 2

# Fix 3: cheap pre-filter before tokenizing, ~10 chars/sec upper bound for wav2vec2-base
MAX_CHARS_PER_SEC = 10

MODEL_NAME = "facebook/wav2vec2-base-960h"

# Load processor once at module level (used in CTC feasibility check)
print(f"Loading processor from {MODEL_NAME}...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)


def ctc_output_length(input_len: int) -> int:
    """Mirror of wav2vec2-base CNN feature extractor output length."""
    output_len = input_len
    for kernel, stride in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
        output_len = (output_len - kernel) // stride + 1
    return output_len


def process_clean(batch):
    result = {
        "clean_audio": None,
        "clean_text": "",
        "keep": False,
        "reject_reason": "",
    }

    try:
        # 1. Validate keys
        if "audio" not in batch or "text" not in batch:
            result["reject_reason"] = "missing_keys"
            return result

        text = str(batch["text"]).strip()
        if not text:
            result["reject_reason"] = "empty_text"
            return result

        audio_info = batch["audio"]

        # 2. Decode audio
        if audio_info.get("bytes"):
            audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
        elif audio_info.get("path") and os.path.exists(audio_info["path"]):
            audio, sr = sf.read(audio_info["path"])
        else:
            result["reject_reason"] = "no_audio_data"
            return result

        # 3. Mono & resample
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)

        audio = audio.astype("float32")
        duration = len(audio) / TARGET_SR

        # 4. Silence check (Fix 1: raised from 1e-6 to MIN_AMPLITUDE)
        if np.max(np.abs(audio)) < MIN_AMPLITUDE:
            result["reject_reason"] = "too_silent"
            return result

        # 5. NaN / Inf guard on raw audio
        if np.isnan(audio).any() or np.isinf(audio).any():
            result["reject_reason"] = "nan_inf_audio"
            return result

        # 6. Duration filter
        if not (MIN_DUR <= duration <= MAX_DUR):
            result["reject_reason"] = "bad_duration"
            return result

        # 7. Cheap chars-per-second pre-filter (Fix 3: avoids tokenizing hopeless samples)
        text_upper = text.upper()
        chars = len(text_upper.replace(" ", ""))
        if chars / duration > MAX_CHARS_PER_SEC:
            result["reject_reason"] = "too_many_chars_per_sec"
            return result

        # 8. CTC feasibility check (Fix 2: exact token-level check with safety margin)
        token_ids = processor.tokenizer(text_upper).input_ids
        token_len = len(token_ids)
        if token_len == 0:
            result["reject_reason"] = "empty_tokens"
            return result

        audio_frames = ctc_output_length(len(audio))
        if audio_frames < token_len * CTC_SAFETY_MARGIN:
            result["reject_reason"] = (
                f"ctc_infeasible(frames={audio_frames},tokens={token_len},"
                f"ratio={audio_frames/token_len:.2f})"
            )
            return result

        # All checks passed
        result["clean_audio"] = audio
        result["clean_text"] = text_upper
        result["keep"] = True

    except Exception as e:
        result["reject_reason"] = f"exception:{e}"
        return result

    return result


if __name__ == "__main__":
    print("Starting clean pre-processing...")
    print(f"  Silence threshold : {MIN_AMPLITUDE}")
    print(f"  Duration range    : [{MIN_DUR}s, {MAX_DUR}s]")
    print(f"  Max chars/sec     : {MAX_CHARS_PER_SEC}")
    print(f"  CTC safety margin : {CTC_SAFETY_MARGIN}x")

    # Load dataset
    ds = load_from_disk(TRAIN_IN).cast_column("audio", Audio(decode=False))
    initial_count = len(ds)
    print(f"\nInitial dataset size: {initial_count}")

    # Process — num_proc=1 because processor is not picklable across workers;
    # increase to NUM_PROC if you move the processor inside process_clean instead.
    print("Processing samples (this may take a while)...")
    ds = ds.map(process_clean, num_proc=1)

    # Rejection breakdown
    if "reject_reason" in ds.column_names:
        reasons = {}
        for r in ds["reject_reason"]:
            if r:
                key = r.split("(")[0]  # group ctc_infeasible variants together
                reasons[key] = reasons.get(key, 0) + 1
        if reasons:
            print("\nRejection breakdown:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  {reason:35s} {count:>6}")

    # Filter
    print("\nFiltering rows...")
    final_ds = ds.filter(lambda x: x["keep"], num_proc=1)

    # Stats
    final_count = len(final_ds)
    removed_count = initial_count - final_count
    pass_pct = (final_count / initial_count) * 100 if initial_count > 0 else 0.0

    # Cleanup columns
    final_ds = final_ds.select_columns(["clean_audio", "clean_text"])

    print("-" * 40)
    print("PRE-PROCESSING SUMMARY")
    print(f"  Total processed : {initial_count}")
    print(f"  Samples passed  : {final_count}")
    print(f"  Samples removed : {removed_count}")
    print(f"  Success rate    : {pass_pct:.2f}%")
    print("-" * 40)

    if final_count == 0:
        print("ERROR: No samples passed. Check thresholds or input data.")
    else:
        save_path = os.path.join(OUT_PATH, "train")
        print(f"Saving to {save_path} ...")
        final_ds.save_to_disk(save_path)

        # Save a few random samples for manual inspection
        num_samples = min(5, final_count)
        print(f"Writing {num_samples} random audio samples for inspection...")
        indices = random.sample(range(final_count), num_samples)

        with open(os.path.join(SCRIPT_DIR, "samples_transcript.txt"), "w") as f:
            for i, idx in enumerate(indices):
                sample = final_ds[idx]
                audio_name = f"sample_{i + 1}.wav"
                sf.write(
                    os.path.join(SCRIPT_DIR, audio_name),
                    sample["clean_audio"],
                    TARGET_SR,
                )
                f.write(f"{audio_name}: {sample['clean_text']}\n")
                print(f"  {audio_name}: {sample['clean_text'][:80]}")

    print("\nDone!")