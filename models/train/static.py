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

ACTIVE_TYPE = "static" 
profile = config["training"]["types"][ACTIVE_TYPE]

print(f"--- DEBUG: Loading Clean Data from Disk ---")
DATA_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")

# Check if directory exists
if not os.path.exists(DATA_PATH):
    print(f"CRITICAL ERROR: Data path {DATA_PATH} does not exist!")

train_raw = load_from_disk(os.path.join(DATA_PATH, "train"))
print(f"--- DEBUG: Dataset size on disk: {len(train_raw)} samples ---")

if len(train_raw) == 0:
    print("CRITICAL ERROR: The processed dataset is EMPTY. Check your pre-processing script.")

# Convert to iterable
train_dataset = train_raw.to_iterable_dataset()
valid_dataset = load_from_disk(os.path.join(DATA_PATH, "valid")).to_iterable_dataset()

processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])

# --- 2. NOISE UTILS WITH DEBUG ---
def load_noises():
    loaded = {}
    noise_dir = os.path.join(SCRIPT_DIR, profile["subfolder"])
    print(f"--- DEBUG: Looking for noises in {noise_dir} ---")
    
    if not os.path.exists(noise_dir):
        print(f"CRITICAL ERROR: Noise directory {noise_dir} not found!")
        return loaded

    files = [f for f in os.listdir(noise_dir) if f.lower().endswith(".wav")]
    print(f"--- DEBUG: Found {len(files)} noise files ---")

    for f in files:
        audio, _ = sf.read(os.path.join(noise_dir, f))
        loaded[f] = audio.astype("float32")
    return loaded

LOADED_NOISES = load_noises()
NOISE_NAMES = list(LOADED_NOISES.keys())

def mix_on_the_fly(batch):
    # This print will run for EVERY sample. 
    # If you see this scrolling fast, the map is working.
    clean = np.array(batch["clean_audio"])
    
    if len(clean) == 0:
        print("WARNING: Encountered empty clean_audio array!")
        return batch

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
    
    # Check for NaNs (Common cause of speed-training)
    if np.isnan(mixed).any():
        print("CRITICAL: Mixed audio contains NaNs!")

    batch["input_values"] = processor(mixed, sampling_rate=16000).input_values[0]
    batch["labels"] = processor.tokenizer(batch["clean_text"]).input_ids
    return batch

train_dataset = train_dataset.map(mix_on_the_fly)

# --- 3. TEST A SAMPLE BEFORE TRAINING ---
print("--- DEBUG: Testing one sample from the pipeline ---")
try:
    test_sample = next(iter(train_dataset))
    print(f"Test Sample Label IDs: {test_sample['labels']}")
    print(f"Test Sample Input Shape: {len(test_sample['input_values'])}")
    if len(test_sample['input_values']) < 1000:
        print("WARNING: Input values are suspiciously short!")
except Exception as e:
    print(f"CRITICAL ERROR during pipeline test: {e}")

# --- 4. DATA COLLATOR ---
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
        
        # DEBUG: Check batch shape
        # print(f"DEBUG Batch: Inputs {batch['input_values'].shape}, Labels {batch['labels'].shape}")
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# --- 5. TRAINING ---
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
    logging_steps=1, # Log every single step to see loss
    eval_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("--- Starting Trainer ---")
trainer.train()