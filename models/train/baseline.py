import os
import numpy as np
import torch
import yaml
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import load_from_disk
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from jiwer import wer

# --- 1. SETUP & CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Using a generic baseline profile or default from config
ACTIVE_TYPE = "baseline" 
SEED = config["training"]["seed"]

DATA_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k")
train_raw = load_from_disk(os.path.join(DATA_PATH, "train"))
train_dataset = train_raw.to_iterable_dataset().shuffle(buffer_size=500, seed=SEED)
valid_dataset = load_from_disk(os.path.join(DATA_PATH, "valid")).to_iterable_dataset()

processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])

# --- 2. BASELINE DATA PREPARATION (NO AUGMENTATION) ---
def prepare_baseline_batch(batch):
    # Use the clean audio directly
    audio = batch["clean_audio"]
    text = str(batch["clean_text"]).upper().strip()

    # 1. Process audio (input_values)
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    
    # 2. Process labels (input_ids)
    # We call the processor directly on the text. 
    # It automatically routes to the tokenizer.
    batch["labels"] = processor(text=text).input_ids
        
    return batch

# Apply mapping without noise injection
train_dataset = train_dataset.map(prepare_baseline_batch)
valid_dataset = valid_dataset.map(prepare_baseline_batch)

# --- 3. DATA COLLATOR ---
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

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

# --- 4. METRICS ---
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_score = wer(label_str, pred_str)
    return {"wer": wer_score}

# --- 5. MODEL ---
model = Wav2Vec2ForCTC.from_pretrained(
    config["model"]["name"], 
    ctc_loss_reduction=config["model"]["ctc_loss_reduction"],
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_zero_infinity=True
)
model.freeze_feature_encoder()

# --- 6. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=os.path.join(SCRIPT_DIR, "output/baseline"),
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    max_steps=config["training"]["max_steps"],
    learning_rate=float(config["training"]["learning_rate"]),
    logging_steps=50,
    eval_strategy="steps",
    warmup_steps=config["training"]["warmup_steps"],
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(), # Use FP16 if GPU is available for speed
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset.take(100),
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("--- Starting Baseline Training ---")
trainer.train()