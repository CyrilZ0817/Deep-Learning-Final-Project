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

ACTIVE_TYPE = "static" 
profile = config["training"]["types"][ACTIVE_TYPE]

# Load Pre-processed Clean Data
DATA_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")
train_raw = load_from_disk(os.path.join(DATA_PATH, "train"))

# Debug: Verify data existence
print(f"--- DEBUG: Dataset size on disk: {len(train_raw)} samples ---")
for i in range(3):
    print(f"Row {i} Text: {train_raw[i]['clean_text'][:50]}...")

train_dataset = train_raw.to_iterable_dataset().shuffle(buffer_size=500, seed=42)
valid_dataset = load_from_disk(os.path.join(DATA_PATH, "valid")).to_iterable_dataset()

processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])

# --- 2. NOISE LOADING ---
def load_noises():
    loaded = {}
    noise_dir = os.path.join(SCRIPT_DIR, profile["subfolder"])
    files = [f for f in os.listdir(noise_dir) if f.lower().endswith(".wav")]
    for f in files:
        audio, _ = sf.read(os.path.join(noise_dir, f))
        loaded[f] = audio.astype("float32")
    return loaded

LOADED_NOISES = load_noises()
NOISE_NAMES = list(LOADED_NOISES.keys())

# --- 3. ON-THE-FLY NOISE MIXING ---
def mix_on_the_fly(batch):
    clean = np.array(batch["clean_audio"])
    text = str(batch["clean_text"]).upper().strip()

    # Noise mixing
    noise = LOADED_NOISES[random.choice(NOISE_NAMES)]
    snr = random.randint(profile["snr_range"]["min"], profile["snr_range"]["max"])
    
    c_rms = np.sqrt(np.mean(clean**2) + 1e-8)
    n_rms = np.sqrt(np.mean(noise**2) + 1e-8)
    target_n_rms = c_rms / (10 ** (snr / 20))
    
    if len(clean) > len(noise):
        noise_aligned = np.tile(noise, (len(clean) // len(noise)) + 1)[:len(clean)]
    else:
        start = random.randint(0, len(noise) - len(clean))
        noise_aligned = noise[start : start + len(clean)]
    
    mixed = clean + noise_aligned * (target_n_rms / (n_rms + 1e-8))

    # To ensure no clipping
    if np.max(np.abs(mixed)) > 1.0:
        mixed = mixed / np.max(np.abs(mixed))

    # Feature Extraction
    batch["input_values"] = processor(mixed, sampling_rate=16000).input_values[0]
    
    # FIX: Use tokenizer directly to avoid AttributeError
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

train_dataset = train_dataset.map(mix_on_the_fly)
valid_dataset = valid_dataset.map(mix_on_the_fly)

# --- 4. ROBUST DATA COLLATOR ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        
        # FIX: Pad labels and mask with -100 to avoid Loss 0
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# --- 5. METRICS ---
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    cer_scores = [cer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
    return {"cer": float(np.mean(cer_scores))}

# --- 6. MODEL & TRAINING ---
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
    logging_steps=1,  # Keep logging_steps=1 to monitor the Loss 0 issue
    eval_strategy="steps",
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    load_best_model_at_end=False,
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