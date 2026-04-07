import io
import os
import random
import numpy as np
import soundfile as sf
import torch

from dataclasses import dataclass
from typing import Dict, List, Union
import yaml

from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from jiwer import cer


# --- LOAD CONFIG ---
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- LOAD DATASET ---
train_dataset = load_dataset(
    config["dataset"]["name"], 
    config["dataset"]["subset"], 
    split=config["dataset"]["train_split"]
)
valid_dataset = load_dataset(
    config["dataset"]["name"], 
    config["dataset"]["subset"], 
    split=config["dataset"]["valid_split"]
)

SEED = config["training"]["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 2. load dataset
train_dataset = load_dataset(
    config["dataset"]["name"], 
    config["dataset"]["subset"], 
    split=config["dataset"]["train_split"]
)
valid_dataset = load_dataset(
    config["dataset"]["name"], 
    config["dataset"]["subset"], 
    split=config["dataset"]["valid_split"]
)

train_dataset = train_dataset.cast_column("audio", Audio(decode=False))
valid_dataset = valid_dataset.cast_column("audio", Audio(decode=False))


# 3. load processor
processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])
target_sr = config["dataset"]["target_sampling_rate"]



# 4. noise utils
def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-8)


def load_noise(path):
    noise, sr = sf.read(path)

    if len(noise.shape) > 1:
        noise = noise.mean(axis=1)

    if sr != 16000:
        import librosa
        noise = librosa.resample(noise.astype("float32"), orig_sr=sr, target_sr=16000)

    return noise.astype("float32")


def mix(clean, noise, snr_db):
    if len(noise) < len(clean):
        repeat_times = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeat_times)

    noise = noise[:len(clean)]

    clean_rms = rms(clean)
    noise_rms = rms(noise)

    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / (noise_rms + 1e-8))

    mixed = clean + noise

    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed.astype("float32")


# 5. preload noises
active_type = "babble"
profile = config["noise"]["types"][active_type]
subfolder = profile["subfolder"]
snr_min = profile["snr_range"]["min"]
snr_max = profile["snr_range"]["max"]
noise_dir = os.path.join(config["noise"]["base_dir"], subfolder)
loaded_noises = {}
# Check if the directory exists
if not os.path.exists(noise_dir):
    raise RuntimeError(f"Noise directory not found: {noise_dir}")
# Iterate through all files in that specific folder
for filename in os.listdir(noise_dir):
    # Only process audio files (add other extensions if needed)
    if not filename.lower().endswith(('.wav')):
        continue
        
    noise_path = os.path.join(noise_dir, filename)
    # Use the filename (without extension) as the dictionary key
    noise_name = os.path.splitext(filename)[0]

    # Skip suspiciously tiny / broken files
    if os.path.getsize(noise_path) < 1000:
        print(f"Warning: invalid or tiny noise file skipped: {noise_path}")
        continue

    try:
        # Assuming load_noise is defined elsewhere in your script
        loaded_noises[noise_name] = load_noise(noise_path)
        print(f"Loaded {active_type} noise: {noise_name} from {filename}")
    except Exception as e:
        print(f"Warning: failed to load noise file {noise_path}: {e}")

# Final validation
if len(loaded_noises) == 0:
    raise RuntimeError(f"No valid noise files were loaded from {noise_dir}. Check your folder!")

available_noise_names = list(loaded_noises.keys())
print(f"--- Successfully loaded {len(available_noise_names)} {active_type} noises ---")


def load_audio_from_record(batch):
    audio_info = batch["audio"]

    # 1. Load Clean Audio
    if audio_info["bytes"] is not None:
        audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
    else:
        audio, sr = sf.read(audio_info["path"])

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Standardize Sample Rate (from config)
    target_sr = config["dataset"]["target_sampling_rate"]
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio = audio.astype("float32")

    # 2. Pick Random Noise and SNR
    # These are derived from the pre-loaded dictionary we built in the previous step
    chosen_noise_name = random.choice(available_noise_names)
    noise_signal = loaded_noises[chosen_noise_name]
    
    # SNR levels from config (e.g., [5, 10, 15])
    chosen_snr = random.randint(snr_min, snr_max)

    # 3. Handle Length Alignment
    audio_len = len(audio)
    noise_len = len(noise_signal)

    if audio_len > noise_len:
        # a. Audio is longer -> Crop audio to match noise length
        audio = audio[:noise_len]
        final_noise = noise_signal
    else:
        # b. Noise is longer -> Pick a random segment of noise to fit the audio
        # Randomly choose a start index between 0 and (noise_len - audio_len)
        max_start = noise_len - audio_len
        start_idx = random.randint(0, max_start)
        final_noise = noise_signal[start_idx : start_idx + audio_len]

    # 4. Mix and return
    # Assuming your mix function handles the SNR math
    noisy_audio = mix(audio, final_noise, chosen_snr)
    batch["speech"] = noisy_audio
    batch["sampling_rate"] = sr
    batch["target_text"] = batch["text"]
    batch["chosen_noise"] = chosen_noise_name
    batch["chosen_snr"] = chosen_snr
    return batch


train_dataset = train_dataset.map(load_audio_from_record)
valid_dataset = valid_dataset.map(load_audio_from_record)

print("Example augmentation:")
print("train noise =", train_dataset[0]["chosen_noise"])
print("train snr   =", train_dataset[0]["chosen_snr"])


# 7. prepare features
def prepare_dataset(batch):
    batch["input_values"] = processor(
        batch["speech"],
        sampling_rate=batch["sampling_rate"]
    ).input_values[0]

    batch["labels"] = processor.tokenizer(batch["target_text"]).input_ids
    return batch


train_dataset = train_dataset.map(
    prepare_dataset,
    remove_columns=train_dataset.column_names
)

valid_dataset = valid_dataset.map(
    prepare_dataset,
    remove_columns=valid_dataset.column_names
)


# 8. data collator
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)


# 9. metrics
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer_scores = [cer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
    avg_cer = float(np.mean(cer_scores))
    return {"cer": avg_cer}


# --- MODEL & TRAINING ---
model = Wav2Vec2ForCTC.from_pretrained(
    config["model"]["name"],
    ctc_loss_reduction=config["model"]["ctc_loss_reduction"],
    pad_token_id=processor.tokenizer.pad_token_id,
)
model.freeze_feature_encoder()


# 11. training args

training_args = TrainingArguments(
    output_dir=config["training"]["output_dir"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"],
    learning_rate=config["training"]["learning_rate"],
    warmup_steps=config["training"]["warmup_steps"],
    logging_steps=config["training"]["logging_steps"],
    save_total_limit=config["training"]["save_total_limit"],
    metric_for_best_model=config["training"]["metric_for_best_model"],
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    load_best_model_at_end=True,
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- EXECUTION ---
trainer.train()
metrics = trainer.evaluate()
print(f"\nFinal evaluation: {metrics}")

trainer.save_model(config["training"]["output_dir"])
processor.save_pretrained(config["training"]["output_dir"])