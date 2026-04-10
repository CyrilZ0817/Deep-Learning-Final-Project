import io
import os
import random
import numpy as np
import soundfile as sf
import torch
import yaml
from dataclasses import dataclass
from typing import Dict, List, Union

from datasets import load_dataset, Audio, IterableDataset
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
    
ACTIVE_TYPE = "static"  

SEED = config["training"]["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Load dataset with streaming
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

# 3. Load processor
processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])

# 4. Noise utils
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
    # Noise is already aligned in the load_audio_from_record function
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / (noise_rms + 1e-8))
    mixed = clean + noise
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    return mixed.astype("float32")

# 5. Preload noises
profile = config["training"]["types"][ACTIVE_TYPE]
noise_dir = os.path.join(script_dir, profile["subfolder"])
snr_min = profile["snr_range"]["min"]
snr_max = profile["snr_range"]["max"]
loaded_noises = {}

if not os.path.exists(noise_dir):
    raise RuntimeError(f"Noise directory not found: {noise_dir}")

for filename in os.listdir(noise_dir):
    if not filename.lower().endswith('.wav'):
        continue
    noise_path = os.path.join(noise_dir, filename)
    noise_name = os.path.splitext(filename)[0]
    try:
        loaded_noises[noise_name] = load_noise(noise_path)
    except Exception as e:
        print(f"Warning: failed to load {noise_path}: {e}")

available_noise_names = list(loaded_noises.keys())
print(f"--- Successfully loaded {len(available_noise_names)} {ACTIVE_TYPE} noises ---")

# 6. Augmented Audio Loader
def load_audio_from_record(batch):
    audio_info = batch["audio"]

    # MANUALLY DECODE using soundfile (bypasses torchcodec)
    try:
        if audio_info["bytes"] is not None:
            # Handle streamed bytes
            audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
        else:
            # Handle local file paths
            audio, sr = sf.read(audio_info["path"])
    except Exception as e:
        print(f"Error decoding audio: {e}")
        # Return a dummy signal so the trainer doesn't crash, 
        # but you should investigate if this happens often.
        audio, sr = np.zeros(16000), 16000

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if necessary
    target_sr = 16000
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    audio = audio.astype("float32")

    # Pick random noise and align (Looping logic)
    chosen_noise_name = random.choice(available_noise_names)
    noise_signal = loaded_noises[chosen_noise_name]
    chosen_snr = random.randint(snr_min, snr_max)

    audio_len = len(audio)
    noise_len = len(noise_signal)

    if audio_len > noise_len:
        repeats = (audio_len // noise_len) + 1
        final_noise = np.tile(noise_signal, repeats)[:audio_len]
    else:
        start_idx = random.randint(0, noise_len - audio_len)
        final_noise = noise_signal[start_idx : start_idx + audio_len]

    batch["speech"] = mix(audio, final_noise, chosen_snr)
    batch["sampling_rate"] = sr
    batch["target_text"] = batch["text"]
    batch["chosen_noise"] = chosen_noise_name
    batch["chosen_snr"] = chosen_snr
    return batch

train_dataset = train_dataset.map(load_audio_from_record)
valid_dataset = valid_dataset.map(load_audio_from_record)

# DEBUG PEAK: Proper way to look at streaming data
debug_sample = next(iter(train_dataset))
print("Example augmentation:")
print(f"train noise = {debug_sample['chosen_noise']}")
print(f"train snr   = {debug_sample['chosen_snr']}")
print(f"audio shape = {debug_sample['speech'].shape}")

# 7. Prepare features
def prepare_dataset(batch):
    # 1. Clean and Uppercase the text (MANDATORY for this model)
    # Librispeech is usually uppercase, but streaming can sometimes vary
    text = batch["target_text"].upper()
    
    # 2. Tokenize
    labels = processor.tokenizer(text).input_ids
    
    # 3. Diagnostic Print (Only shows up once in logs usually)
    if len(labels) == 0:
        print(f"!!! CRITICAL WARNING: Text '{text}' resulted in empty labels!")
    
    batch["input_values"] = processor(
        batch["speech"],
        sampling_rate=batch["sampling_rate"]
    ).input_values[0]

    batch["labels"] = labels
    return batch

# We remove columns manually because iterable datasets can be picky about column_names
columns_to_remove = ["audio", "text", "speech", "sampling_rate", "target_text", "chosen_noise", "chosen_snr"]
train_dataset = train_dataset.map(prepare_dataset, remove_columns=columns_to_remove)
valid_dataset = valid_dataset.map(prepare_dataset, remove_columns=columns_to_remove)

# 8. Data collator (Remains the same)
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# 9. Metrics
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    cer_scores = [cer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
    return {"cer": float(np.mean(cer_scores))}

# --- MODEL & TRAINING ---
model = Wav2Vec2ForCTC.from_pretrained(
    config["model"]["name"],
    ctc_loss_reduction=config["model"]["ctc_loss_reduction"],
    pad_token_id=processor.tokenizer.pad_token_id,
)
model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir=os.path.join(script_dir, profile["output_dir"]),
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    max_steps=config["training"]["max_steps"],                
    learning_rate=float(config["training"]["learning_rate"]),
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
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- EXECUTION ---
trainer.train()
trainer.save_model(os.path.join(script_dir, profile["output_dir"]))
processor.save_pretrained(os.path.join(script_dir, profile["output_dir"]))