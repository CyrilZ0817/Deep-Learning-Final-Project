import os
import random
import numpy as np
import torch
import yaml
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import load_from_disk
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from jiwer import cer

# --- 1. SETUP & CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

ACTIVE_TYPE = "babble" 
profile = config["training"]["types"][ACTIVE_TYPE]

DATA_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")
train_raw = load_from_disk(os.path.join(DATA_PATH, "train"))
train_dataset = train_raw.to_iterable_dataset().shuffle(buffer_size=500, seed=42)
valid_dataset = load_from_disk(os.path.join(DATA_PATH, "valid")).to_iterable_dataset()
print(f"the keys of the dataset are {train_dataset.features.keys()}")


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

# --- 3. MIXING WITH NUMERICAL SANITIZATION ---
# noise utils
def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-8)

def mix_on_the_fly(batch):
    clean = np.array(batch["clean_audio"], dtype=np.float32)
    text = str(batch["clean_text"]).upper().strip()

    noise = LOADED_NOISES[random.choice(NOISE_NAMES)]
    snr = random.randint(profile["snr_range"]["min"], profile["snr_range"]["max"])
    
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    target_n_rms = clean_rms / (10 ** (snr / 20))
    
    if len(clean) > len(noise):
        noise_aligned = np.tile(noise, (len(clean) // len(noise)) + 1)[:len(clean)]
    else:
        start = random.randint(0, len(noise) - len(clean))
        noise_aligned = noise[start : start + len(clean)]
    
    mixed = clean + noise_aligned * (target_n_rms / (noise_rms + 1e-8))
    
    # --- FIX: Extreme Sanitization ---
    # Ensure no NaNs and clip to valid audio range
    mixed = np.nan_to_num(mixed)
    if np.max(np.abs(mixed)) > 1.0:
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-8)

    batch["input_values"] = processor(mixed, sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(text).input_ids
    return batch

train_dataset = train_dataset.map(mix_on_the_fly)
valid_dataset = valid_dataset.map(mix_on_the_fly)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 (ignored tokens) with pad token id so tokenizer can decode
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # group_tokens=False is important for labels to maintain original structure
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer_score = cer(label_str, pred_str)

    return {"cer": cer_score}

# --- 4. FAIL-SAFE DATA COLLATOR ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        
        # Standardize labels
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        
        # Use a more conservative mask
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# --- 5. MODEL WITH CTC STABILITY ---
model = Wav2Vec2ForCTC.from_pretrained(
    config["model"]["name"], 
    ctc_loss_reduction=config["model"]["ctc_loss_reduction"],
    pad_token_id=processor.tokenizer.pad_token_id,
    # --- FIX: ZERO INFINITY ---
    # This prevents 'nan' loss if the alignment is mathematically impossible
    ctc_zero_infinity=True 
)
model.freeze_feature_encoder()

# --- 6. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=os.path.join(SCRIPT_DIR, profile["output_dir"]),
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    max_steps=config["training"]["max_steps"],
    learning_rate=float(config["training"]["learning_rate"]),
    
    logging_steps=200,
    eval_strategy="steps",
    logging_strategy="steps",
    warmup_steps=config["training"]["warmup_steps"],
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset.take(100),
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# --- 7. FINAL BATCH INSPECTION ---
print("--- DEBUG: Inspecting the first real batch for the model ---")
dataloader = trainer.get_train_dataloader()
first_batch = next(iter(dataloader))
print(f"Batch Inputs Shape: {first_batch['input_values'].shape}")
print(f"Batch Labels Shape: {first_batch['labels'].shape}")
print(f"Non-masked labels in first sample: {(first_batch['labels'][0] != -100).sum()}")

if (first_batch['labels'] != -100).sum() == 0:
    print("!!! CRITICAL: THE ENTIRE BATCH IS MASKED. TRAINING WILL FAIL.")

print("--- Starting Trainer ---")
trainer.train()