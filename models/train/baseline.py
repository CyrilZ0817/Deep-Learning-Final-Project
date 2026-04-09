import io
import yaml
import numpy as np
import soundfile as sf
import torch
import librosa

from dataclasses import dataclass
from typing import Any, Dict, List, Union

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

ACTIVE_TYPE = "baseline"  

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

# --- PROCESSOR & AUDIO HELPER ---
processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])
target_sr = config["dataset"]["target_sampling_rate"]

def load_audio_from_record(batch):
    audio_info = batch["audio"]
    
    if audio_info["bytes"] is not None:
        audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
    else:
        audio, sr = sf.read(audio_info["path"])

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    batch["speech"] = audio.astype("float32")
    batch["sampling_rate"] = sr
    batch["target_text"] = batch["text"]
    return batch

train_dataset = train_dataset.map(load_audio_from_record)
valid_dataset = valid_dataset.map(load_audio_from_record)

def prepare_dataset(batch):
    batch["input_values"] = processor(
        batch["speech"],
        sampling_rate=batch["sampling_rate"]
    ).input_values[0]
    batch["labels"] = processor.tokenizer(batch["target_text"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
valid_dataset = valid_dataset.map(prepare_dataset, remove_columns=valid_dataset.column_names)

# --- COLLATOR & METRICS ---
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

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

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
    output_dir= config["training"]["types"][ACTIVE_TYPE]["output_dir"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"],
    learning_rate=config["training"]["learning_rate"],
    max_steps=config["training"]["max_steps"],
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

trainer.save_model(config["training"]["types"][ACTIVE_TYPE]["output_dir"])
processor.save_pretrained(config["training"]["types"][ACTIVE_TYPE]["output_dir"])