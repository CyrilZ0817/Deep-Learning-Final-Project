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
from torch.utils.data import Dataset as TorchDataset
from jiwer import wer
import glob
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import DataLoader

# Set up connection to config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Get type
ACTIVE_TYPE = "complex" 
profile = config["training"]["types"][ACTIVE_TYPE]

SEED = config["training"]["seed"]

# Get datasets
FLAC_ROOT = os.path.join(SCRIPT_DIR, "data/LibriSpeech/train-clean-100") 

def build_flac_index(root):
    """Return list of (flac_path, transcript_text) pairs."""
    samples = []
    for trans_file in glob.glob(os.path.join(root, "**/*.trans.txt"), recursive=True):
        folder = os.path.dirname(trans_file)
        with open(trans_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    flac_path = os.path.join(folder, utt_id + ".flac")
                    if os.path.exists(flac_path):
                        samples.append({"flac_path": flac_path, "clean_text": text})
    return samples

class LibriSpeechFLACDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio, _ = sf.read(item["flac_path"])          # sf already handles FLAC
        return {
            "clean_audio": audio.astype("float32"),
            "clean_text":  item["clean_text"],
            "input_length": len(audio),
        }

all_samples = build_flac_index(FLAC_ROOT)
random.seed(SEED)
random.shuffle(all_samples)

split = int(0.95 * len(all_samples))
train_dataset = LibriSpeechFLACDataset(all_samples[:split])
valid_dataset = LibriSpeechFLACDataset(all_samples[split:])
print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

# Use the same processor as Wav2Vec2 
# Since we are using the same vocabulary and the audio files are all 16 kHz, this should be compatible.
processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])

# Load associated noises
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

# Add noise at same sampling rate
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
    return batch

class NoisyDataset(TorchDataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.lengths = [
            sf.info(s["flac_path"]).frames for s in base_dataset.samples
        ]
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        return mix_on_the_fly(self.base[idx])

train_dataset = NoisyDataset(train_dataset)
valid_dataset = NoisyDataset(valid_dataset)

batch_size = config["training"]["per_device_train_batch_size"]



# Data collator with padding
# Pud input to the longest sample
# Pud text to the longest label
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

# Metrics
# Use WER as the main metric for evaluation
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_scores = [wer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
    avg_wer = float(np.mean(wer_scores))
    return {"wer": avg_wer}

# Use base model 
model = Wav2Vec2ForCTC.from_pretrained(
    config["model"]["name"], 
    ctc_loss_reduction=config["model"]["ctc_loss_reduction"],
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_zero_infinity=True
)
# no need to further fine tune
model.freeze_feature_encoder()

# Start training
training_args = TrainingArguments(
    output_dir=os.path.join(SCRIPT_DIR, profile["output_dir"]),
    group_by_length=True,
    length_column_name="input_length",
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    max_steps=config["training"]["max_steps"],
    learning_rate=float(config["training"]["learning_rate"]),
    
    logging_steps=50,
    eval_strategy="steps",
    logging_strategy="steps",
    warmup_steps=config["training"]["warmup_steps"],
    eval_steps=config["training"]["eval_steps"],
    save_steps=config["training"]["save_steps"],
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,
    fp16=True,
    weight_decay=config["training"]["weight_decay"],
    save_total_limit= config["training"]["save_total_limit"],
)

class LengthGroupedTrainer(Trainer):
    def get_train_dataloader(self):
        train_sampler = LengthGroupedSampler(
            batch_size=batch_size,
            lengths=self.train_dataset.lengths,
            dataset=self.train_dataset,
        )
        return DataLoader(
            self.train_dataset,
            collate_fn=data_collator,
            sampler=train_sampler,
            batch_size=batch_size,
        )

trainer = LengthGroupedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


print("--- Starting Trainer ---")
trainer.train()