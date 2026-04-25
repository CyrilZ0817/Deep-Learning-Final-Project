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
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
ACTIVE_TYPE = "mixed" 

SEED = config["training"]["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# load dataset
train_dataset = load_dataset(
    config["dataset"]["name"], 
    config["dataset"]["subset"], 
    split=config["dataset"]["train_split"],
    streaming=True 
)
valid_dataset = load_dataset(
    config["dataset"]["name"], 
    config["dataset"]["subset"], 
    split=config["dataset"]["valid_split"],
    streaming=True 
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
# Initialize aggregated storage
loaded_noises = {}
all_types = config["noise"]["types"]

print(f"--- Starting bulk load for ALL noise types ---")

# Iterate through every noise type defined in the config (env, social, speech, etc.)
for noise_type, p in all_types.items():
    subfolder = p["subfolder"]
    noise_dir = os.path.join(script_dir, subfolder)
    
    # Check if the directory exists for this specific type
    if not os.path.exists(noise_dir):
        print(f"Warning: Directory for {noise_type} not found at {noise_dir}. Skipping.")
        continue

    print(f"Loading files for category: {noise_type}...")

    # Iterate through files in this specific subfolder
    for filename in os.listdir(noise_dir):
        if not filename.lower().endswith('.wav'):
            continue
            
        noise_path = os.path.join(noise_dir, filename)
        
        # We include the noise_type in the key to prevent name collisions 
        # (e.g., if two folders both have 'noise_1.wav')
        noise_key = f"{noise_type}_{os.path.splitext(filename)[0]}"

        # Skip tiny files
        if os.path.getsize(noise_path) < 1000:
            print(f"  [Skip] Tiny file: {filename}")
            continue

        try:
            # Load the noise
            loaded_noises[noise_key] = load_noise(noise_path)
            print(f"  [OK] Loaded: {noise_key}")
        except Exception as e:
            print(f"  [Error] Failed to load {filename}: {e}")

# Final validation across all categories
if len(loaded_noises) == 0:
    raise RuntimeError("No valid noise files were loaded from ANY folder. Check your config and paths!")

available_noise_names = list(loaded_noises.keys())
print(f"\n--- SUCCESS: Total noises loaded across all types: {len(available_noise_names)} ---")

def load_audio_from_record(batch):
    audio_info = batch["audio"]

    # --- 1. Load Clean Audio ---
    if audio_info["bytes"] is not None:
        audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
    else:
        audio, sr = sf.read(audio_info["path"])

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    target_sr = config["dataset"]["target_sampling_rate"]
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio = audio.astype("float32")

    # --- 2. Step 1: Pick a Random TYPE (Scenario) ---
    # Get the list of all available types from your config keys
    all_types = list(config["noise"]["types"].keys())
    chosen_type = random.choice(all_types)
    profile = config["noise"]["types"][chosen_type]
    
    # --- Step 2: Pick a Random NOISE from that specific Type ---
    # We filter our loaded_noises keys to only those belonging to this type
    # (Assuming we used the "noise_type_filename" naming convention from the previous step)
    type_specific_noises = [k for k in loaded_noises.keys() if k.startswith(f"{chosen_type}_")]
    
    if not type_specific_noises:
        # Fallback if a folder was empty
        chosen_noise_name = random.choice(list(loaded_noises.keys()))
    else:
        chosen_noise_name = random.choice(type_specific_noises)
    
    noise_signal = loaded_noises[chosen_noise_name]
    
    # --- Step 3: Get the SNR range specific to this Type ---
    # This is where you ensure 'speech' is quieter than 'env'
    snr_min = profile["snr_range"]["min"]
    snr_max = profile["snr_range"]["max"]
    chosen_snr = random.randint(snr_min, snr_max)

    # --- 3. Handle Length Alignment ---
    audio_len = len(audio)
    noise_len = len(noise_signal)

    if audio_len > noise_len:
        # Crop audio to match noise (or you could loop the noise)
        audio = audio[:noise_len]
        final_noise = noise_signal
    else:
        # Pick random segment of noise
        max_start = noise_len - audio_len
        start_idx = random.randint(0, max_start)
        final_noise = noise_signal[start_idx : start_idx + audio_len]

    # --- 4. Mix and return ---
    noisy_audio = mix(audio, final_noise, chosen_snr)
    
    batch["speech"] = noisy_audio
    batch["sampling_rate"] = sr
    batch["target_text"] = batch["text"]
    batch["chosen_noise_type"] = chosen_type # Useful for debugging/metrics
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
    output_dir= os.path.join(script_dir, profile["output_dir"],),
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    max_steps=config["training"]["max_steps"],                
    learning_rate=config["training"]["learning_rate"],
    logging_steps=config["training"]["logging_steps"],            
    
    fp16=torch.cuda.is_available(),
    report_to="none",
    load_best_model_at_end=True,
    greater_is_better=False,
    
    eval_strategy="steps",       
    save_strategy="steps",       
    eval_steps=config["training"]["eval_steps"],                
    save_steps=config["training"]["save_steps"],
    metric_for_best_model=config["training"]["metric_for_best_model"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset.take(100),
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- EXECUTION ---
trainer.train()
metrics = trainer.evaluate(eval_dataset=valid_dataset.take(100))
print(f"\nFinal evaluation: {metrics}")

trainer.save_model(config["training"]["types"][ACTIVE_TYPE]["output_dir"])
processor.save_pretrained(config["training"]["types"][ACTIVE_TYPE]["output_dir"])