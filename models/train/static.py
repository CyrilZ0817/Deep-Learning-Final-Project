import os
import random
import numpy as np
import torch
import yaml
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import load_from_disk
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC, 
    TrainingArguments, 
    Trainer
)
from jiwer import cer

# --- 1. SETUP & CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

ACTIVE_TYPE = "static"  # Change this based on your experiment
profile = config["training"]["types"][ACTIVE_TYPE]

# Load Clean Data
DATA_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")
train_raw = load_from_disk(os.path.join(DATA_PATH, "train"))

# --- 2. DATA VERIFICATION ---
print(f"--- DEBUG: Verifying first 5 rows of Clean Data ---")
for i in range(5):
    sample_text = train_raw[i]["clean_text"]
    sample_audio_len = len(train_raw[i]["clean_audio"])
    print(f"Row {i} | Length: {sample_audio_len:7} | Text: {sample_text[:80]}...")

if not train_raw[0]["clean_text"]:
    raise ValueError("CRITICAL ERROR: No text found in the 'clean_text' column!")

# Convert to Iterable for on-the-fly mixing
train_dataset = train_raw.to_iterable_dataset()
valid_dataset = load_from_disk(os.path.join(DATA_PATH, "valid")).to_iterable_dataset()

processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])

# --- 3. NOISE LOADING ---
def load_noises():
    loaded = {}
    noise_dir = os.path.join(SCRIPT_DIR, profile["subfolder"])
    if not os.path.exists(noise_dir):
        raise FileNotFoundError(f"Noise directory {noise_dir} not found!")
    
    files = [f for f in os.listdir(noise_dir) if f.lower().endswith(".wav")]
    print(f"--- DEBUG: Loading {len(files)} noise files ---")
    for f in files:
        audio, _ = sf.read(os.path.join(noise_dir, f))
        loaded[f] = audio.astype("float32")
    return loaded

LOADED_NOISES = load_noises()
NOISE_NAMES = list(LOADED_NOISES.keys())

# --- 4. ON-THE-FLY MIXING & TOKENIZATION ---
def mix_on_the_fly(batch):
    clean = np.array(batch["clean_audio"])
    text = str(batch["clean_text"]).upper().strip()

    # RMS Noise Mixing
    noise = LOADED_NOISES[random.choice(NOISE_NAMES)]
    snr = random.randint(profile["snr_range"]["min"], profile["snr_range"]["max"])
    
    c_rms = np.sqrt(np.mean(clean**2) + 1e-8)
    n_rms = np.sqrt(np.mean(noise**2) + 1e-8)
    target_n_rms = c_rms / (10 ** (snr / 20))
    
    # Align noise length
    if len(clean) > len(noise):
        noise_aligned = np.tile(noise, (len(clean) // len(noise)) + 1)[:len(clean)]
    else:
        start = random.randint(0, len(noise) - len(clean))
        noise_aligned = noise[start : start + len(clean)]
    
    mixed = clean + noise_aligned * (target_n_rms / (n_rms + 1e-8))
    
    # Normalizing audio
    if np.max(np.abs(mixed)) > 1.0:
        mixed = mixed / np.max(np.abs(mixed))

    # Features and Labels
    # We call tokenizer directly to avoid as_target_processor AttributeError
    batch["input_values"] = processor(mixed, sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

train_dataset = train_dataset.map(mix_on_the_fly)
valid_dataset = valid_dataset.map(mix_on_the_fly)

# --- 5. DATA COLLATOR ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        
        # Use tokenizer.pad to handle text sequence padding
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        
        # Replace padding with -100 to ignore it in CTC Loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# --- 6. METRICS ---
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    cer_scores = [cer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
    return {"cer": float(np.mean(cer_scores))}

# --- 7. TRAINING ---
model = Wav2Vec2ForCTC.from_pretrained(
    config["model"]["name"], 
    ctc_loss_reduction=config["model"]["ctc_loss_reduction"],
    pad_token_id=processor.tokenizer.pad_token_id
)
model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir=os.path.join(SCRIPT_DIR, profile["output_dir"]),
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    max_steps=config["training"]["max_steps"],
    learning_rate=float(config["training"]["learning_rate"]),
    fp16=torch.cuda.is_available(),
    logging_steps=config["training"]["logging_steps"],
    eval_strategy="steps",
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    load_best_model_at_end=False,  # Essential for Iterable datasets
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset.take(100),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("--- Starting Training ---")
trainer.train()

# Final Save
trainer.save_model(os.path.join(SCRIPT_DIR, profile["output_dir"]))
processor.save_pretrained(os.path.join(SCRIPT_DIR, profile["output_dir"]))