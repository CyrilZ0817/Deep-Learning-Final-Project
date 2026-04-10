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

DATA_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")
train_raw = load_from_disk(os.path.join(DATA_PATH, "train"))

# --- DATA VERIFICATION ---
print(f"--- DEBUG: Verifying first 5 rows of Clean Data ---")
for i in range(5):
    sample_text = train_raw[i]["clean_text"]
    sample_audio_len = len(train_raw[i]["clean_audio"])
    print(f"Row {i} | Length: {sample_audio_len:7} | Text: {sample_text}")

if not train_raw[0]["clean_text"]:
    print("CRITICAL ERROR: No text found in the 'clean_text' column!")
# -------------------------

train_dataset = train_raw.to_iterable_dataset()

# Load processor
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

# --- 3. ACTION PLAN: FIXING THE MAPPING ---
def mix_on_the_fly(batch):
    clean = np.array(batch["clean_audio"])
    text = str(batch["clean_text"]).upper().strip() # ACTION: Force uppercase & clean

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

    # ACTION: Feature Extraction
    batch["input_values"] = processor(mixed, sampling_rate=16000).input_values[0]
    
    # ACTION: Tokenize with target processor (Essential for CTC labels)
    with processor.as_target_processor():
        batch["labels"] = processor.tokenizer(text).input_ids

    # ACTION: Debugging empty labels
    if len(batch["labels"]) == 0:
        print(f"!!! WARNING: Empty labels for text: '{text}'")
        
    return batch

train_dataset = train_dataset.map(mix_on_the_fly)

# --- 4. ACTION PLAN: REFINED COLLATOR ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Input padding
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        # Label padding
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")

        # ACTION: Replace pad with -100 so it's ignored by the loss function
        # If labels remain 0, model will achieve 0 loss.
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# --- 5. MODEL & TRAINING ---
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
    logging_steps=1, 
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("--- Starting Corrected Trainer ---")
trainer.train()