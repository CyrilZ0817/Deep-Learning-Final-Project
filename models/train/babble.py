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
    mixed = np.nan_to_num(mixed, nan=0.0, posinf=1.0, neginf=-1.0)
    mixed = mixed / (np.abs(mixed).max() + 1e-8)   # normalize to [-1, 1]
    assert not np.isnan(mixed).any(), "NaN survived sanitization!"
    

    batch["input_values"] = np.array(
        processor(mixed, sampling_rate=16000, return_tensors="np").input_values[0],
        dtype=np.float32
    )
    batch["labels"] = processor.tokenizer(text).input_ids
    
    # DEBUG PRINTS
    audio_len = len(batch["input_values"])
    label_len = len(batch["labels"])
    frames = audio_len // 320
    
    if frames <= label_len:
        print(f"!!! CRITICAL: Audio too short! Frames: {frames}, Labels: {label_len}")
        print(f"Text: {batch['clean_text']}")
        
    if np.abs(mixed).max() < 1e-5:
        print("!!! CRITICAL: Audio is silent!")
        
    return batch

train_dataset = train_dataset.map(mix_on_the_fly)
valid_dataset = valid_dataset.map(mix_on_the_fly)

def check_batch(batch):
    audio_len = len(batch["input_values"])
    label_len = len(batch["labels"])
    output_frames = (audio_len - 400) // 320
    batch["is_valid"] = int(output_frames >= label_len + 2)
    return batch

# Check on a sample
sample = train_dataset.take(200)
sample = sample.map(check_batch)
valid_count = sum(s["is_valid"] for s in sample)
print(f"Valid samples: {valid_count}/200")

train_dataset = train_dataset.map(mix_on_the_fly).filter(
    lambda x: x["input_values"] is not None
)

# --- 4. FAIL-SAFE DATA COLLATOR ---
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
    
    logging_steps=50,
    eval_strategy="steps",
    logging_strategy="steps",
    warmup_steps=config["training"]["warmup_steps"],
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,
    fp16=False,
    max_grad_norm=1.0,
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

# 1. Confirm attention_mask is present
first_batch = next(iter(trainer.get_train_dataloader()))
print("Batch keys:", first_batch.keys())
print("Attention mask shape:", first_batch.get("attention_mask"))

# 2. Check for NaN in raw audio
for name, audio in LOADED_NOISES.items():
    if np.isnan(audio).any():
        print(f"NaN in noise file: {name}")

# 3. Check a single mixed sample
sample = next(iter(train_dataset))
print("NaN in input_values:", np.isnan(sample["input_values"]).any())
print("Input range:", np.min(sample["input_values"]), np.max(sample["input_values"]))


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