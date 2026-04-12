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
    Trainer,
    EarlyStoppingCallback,
)
from jiwer import cer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ACTIVE_TYPE = "complex"
profile = config["training"]["types"][ACTIVE_TYPE]

SEED = config["training"]["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k/train")
full_ds = load_from_disk(DATA_PATH)
print(f"Loaded combined dataset: {len(full_ds)} samples.")

# 2. Create the split on the fly
# Adjust test_size to match your original validation ratio (e.g., 0.1 for 10%)
split_ds = full_ds.train_test_split(test_size=0.1, seed=SEED)

train_raw = split_ds["train"]
valid_raw = split_ds["test"]

print(f"Split complete! Train: {len(train_raw)}, Valid: {len(valid_raw)}")

train_dataset = train_raw.to_iterable_dataset().shuffle(buffer_size=1000, seed=SEED)
valid_dataset = valid_raw.to_iterable_dataset()

print(f"train keys: {train_dataset.features.keys()}")
print(f"valid keys: {valid_dataset.features.keys()}")

processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])
processor.feature_extractor.do_normalize = True


def load_noises():
    loaded = {}
    noise_dir = os.path.join(SCRIPT_DIR, profile["subfolder"])
    files = [f for f in os.listdir(noise_dir) if f.lower().endswith(".wav")]

    for fname in files:
        path = os.path.join(noise_dir, fname)
        audio, sr = sf.read(path)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            import librosa
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=16000)

        audio = np.asarray(audio, dtype=np.float32)

        if np.isnan(audio).any() or np.isinf(audio).any():
            print(f"Skipping invalid noise file: {fname}")
            continue

        loaded[fname] = audio

    return loaded


LOADED_NOISES = load_noises()
NOISE_NAMES = list(LOADED_NOISES.keys())

if len(NOISE_NAMES) == 0:
    raise RuntimeError("No valid babble noise files found.")

print(f"Loaded {len(NOISE_NAMES)} noise files.")



def rms(x):
    x = np.asarray(x, dtype=np.float32)
    return np.sqrt(np.mean(x ** 2) + 1e-8)


def align_noise(noise, target_len):
    if len(noise) < target_len:
        repeat_times = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeat_times)
        return noise[:target_len]

    start = random.randint(0, len(noise) - target_len)
    return noise[start:start + target_len]


def mix_with_snr(clean, noise, snr_db):
    clean = np.asarray(clean, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)

    noise_aligned = align_noise(noise, len(clean))

    # 关键：对齐之后再算 noise_rms
    clean_rms = rms(clean)
    noise_rms = rms(noise_aligned)

    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + 1e-8)

    mixed = clean + noise_aligned * scale
    mixed = np.nan_to_num(mixed, nan=0.0, posinf=1.0, neginf=-1.0)

    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def mix_on_the_fly(batch):
    clean = np.asarray(batch["clean_audio"], dtype=np.float32)
    text = str(batch["clean_text"]).upper().strip()

    noise_name = random.choice(NOISE_NAMES)
    noise = LOADED_NOISES[noise_name]
    snr = random.randint(profile["snr_range"]["min"], profile["snr_range"]["max"])

    mixed = mix_with_snr(clean, noise, snr)

    input_values = processor(
        mixed,
        sampling_rate=16000,
        return_tensors="np"
    ).input_values[0].astype(np.float32)

    labels = processor.tokenizer(text).input_ids

    batch["input_values"] = input_values
    batch["labels"] = labels
    batch["chosen_noise"] = noise_name
    batch["chosen_snr"] = snr
    batch["clean_text"] = text
    return batch


def ctc_output_length(input_len):
    output_len = input_len
    for kernel, stride in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
        output_len = (output_len - kernel) // stride + 1
    return output_len


def is_ctc_valid(example):
    audio_len = len(example["input_values"])
    label_len = len(example["labels"])
    out_len = ctc_output_length(audio_len)

    # make it really strict
    return out_len >= label_len* 6 


train_dataset = train_dataset.map(mix_on_the_fly).filter(is_ctc_valid)
valid_dataset = valid_dataset.map(mix_on_the_fly).filter(is_ctc_valid)



@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True,
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
        batch["attention_mask"] = batch["attention_mask"].long()
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)



def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer_scores = [cer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
    return {"cer": float(np.mean(cer_scores))}


model = Wav2Vec2ForCTC.from_pretrained(
    config["model"]["name"],
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_zero_infinity=False,   # 先不要把错误样本静默归零
)

# 关键：不要冻结
# model.freeze_feature_encoder()

training_args = TrainingArguments(
    output_dir=os.path.join(SCRIPT_DIR, profile["output_dir"] + "_fixed"),
    per_device_train_batch_size=min(2, int(config["training"]["per_device_train_batch_size"])),
    per_device_eval_batch_size=2,
    max_steps=int(config["training"]["max_steps"]),
    learning_rate=1e-5,
    warmup_steps=config["training"]["warmup_steps"],
    logging_steps=50,
    eval_strategy="steps",
    logging_strategy="steps",
    eval_steps=int(config["training"]["eval_steps"]),
    save_steps=int(config["training"]["save_steps"]),
    metric_for_best_model="cer",
    greater_is_better=False,
    fp16=False,
    max_grad_norm=1.0,
    report_to="none",
    remove_unused_columns=False,
    load_best_model_at_end=True,
    save_total_limit=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset.take(100),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)


print("--- DEBUG: Inspecting first batch ---")
first_batch = next(iter(trainer.get_train_dataloader()))
print("Batch keys:", first_batch.keys())
print("input_values shape:", first_batch["input_values"].shape)
print("labels shape:", first_batch["labels"].shape)
print("attention_mask dtype:", first_batch["attention_mask"].dtype)

model.train()
probe_batch = {k: v.to(model.device) for k, v in first_batch.items()}
probe_out = model(**probe_batch)
print("Probe loss:", probe_out.loss.item())

if np.isnan(probe_out.loss.item()) or probe_out.loss.item() == 0.0:
    print("WARNING: probe loss is suspicious. Check filtering / text lengths / tokenizer.")


print("--- Starting Trainer ---")
trainer.train()

metrics = trainer.evaluate()
print("\nFinal evaluation:")
print(metrics)

save_dir = os.path.join(SCRIPT_DIR, profile["output_dir"] + "_fixed")
trainer.save_model(save_dir)
processor.save_pretrained(save_dir)

print(f"\nSaved model to: {save_dir}")