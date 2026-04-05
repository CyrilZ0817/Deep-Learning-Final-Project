import io
import os
import random
import numpy as np
import soundfile as sf
import torch

from dataclasses import dataclass
from typing import Dict, List, Union

from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from jiwer import cer


# 1. settings

MODEL_NAME = "facebook/wav2vec2-base-960h"
TRAIN_SPLIT = "train.100[:1%]"
VALID_SPLIT = "validation[:1%]"
OUTPUT_DIR = "./wav2vec2-multi-noise-augmented"

# Multiple noise files
NOISE_FILES = {
    "stationary": "noises/AirConditioner_2.wav",
    "nonstationary": "noises/AirportAnnouncements_1.wav",
    "multispeaker": "noises/Babble_1.wav",
}

# Random SNR choices
SNR_LEVELS = [20, 10, 0]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 2. load dataset

train_dataset = load_dataset("librispeech_asr", "clean", split=TRAIN_SPLIT)
valid_dataset = load_dataset("librispeech_asr", "clean", split=VALID_SPLIT)

train_dataset = train_dataset.cast_column("audio", Audio(decode=False))
valid_dataset = valid_dataset.cast_column("audio", Audio(decode=False))


# 3. load processor
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)


# 4. noise utils
def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-8)


def load_noise(path):
    noise, sr = sf.read(path)

    if len(noise.shape) > 1:
        noise = noise.mean(axis=1)

    if sr != 16000:
        import librosa
        noise = librosa.resample(noise.astype("float32"), orig_sr=sr, target_sr=16000)

    return noise.astype("float32")


def mix(clean, noise, snr_db):
    if len(noise) < len(clean):
        repeat_times = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeat_times)

    noise = noise[:len(clean)]

    clean_rms = rms(clean)
    noise_rms = rms(noise)

    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / (noise_rms + 1e-8))

    mixed = clean + noise

    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed.astype("float32")


# 5. preload noises
loaded_noises = {}

for noise_name, noise_path in NOISE_FILES.items():
    if not os.path.exists(noise_path):
        print(f"Warning: missing noise file: {noise_path}")
        continue

    # Skip suspiciously tiny / broken files
    if os.path.getsize(noise_path) < 1000:
        print(f"Warning: invalid or tiny noise file skipped: {noise_path}")
        continue

    try:
        loaded_noises[noise_name] = load_noise(noise_path)
        print(f"Loaded noise: {noise_name} -> {noise_path}")
    except Exception as e:
        print(f"Warning: failed to load noise file {noise_path}: {e}")

if len(loaded_noises) == 0:
    raise RuntimeError("No valid noise files were loaded. Please check the noises/ folder.")

available_noise_names = list(loaded_noises.keys())
print("Available noise types:", available_noise_names)


# 6. load and mix audio
def load_audio_from_record(batch):
    audio_info = batch["audio"]

    if audio_info["bytes"] is not None:
        audio, sr = sf.read(io.BytesIO(audio_info["bytes"]))
    else:
        audio, sr = sf.read(audio_info["path"])

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=16000)
        sr = 16000

    audio = audio.astype("float32")

    # Randomly choose noise type and SNR
    chosen_noise_name = random.choice(available_noise_names)
    chosen_noise = loaded_noises[chosen_noise_name]
    chosen_snr = random.choice(SNR_LEVELS)

    noisy_audio = mix(audio, chosen_noise, chosen_snr)

    batch["speech"] = noisy_audio
    batch["sampling_rate"] = sr
    batch["target_text"] = batch["text"]
    batch["chosen_noise"] = chosen_noise_name
    batch["chosen_snr"] = chosen_snr
    return batch


train_dataset = train_dataset.map(load_audio_from_record)
valid_dataset = valid_dataset.map(load_audio_from_record)

print("Example augmentation:")
print("train noise =", train_dataset[0]["chosen_noise"])
print("train snr   =", train_dataset[0]["chosen_snr"])


# 7. prepare features
def prepare_dataset(batch):
    batch["input_values"] = processor(
        batch["speech"],
        sampling_rate=batch["sampling_rate"]
    ).input_values[0]

    batch["labels"] = processor.tokenizer(batch["target_text"]).input_ids
    return batch


train_dataset = train_dataset.map(
    prepare_dataset,
    remove_columns=train_dataset.column_names
)

valid_dataset = valid_dataset.map(
    prepare_dataset,
    remove_columns=valid_dataset.column_names
)


# 8. data collator
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


# 10. load model
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

model.freeze_feature_encoder()


# 11. training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    num_train_epochs=1,
    learning_rate=1e-5,
    warmup_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

# 12. trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 13. train
trainer.train()

# 14. final eval
metrics = trainer.evaluate()
print("\nFinal evaluation:")
print(metrics)

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\nSaved model to: {OUTPUT_DIR}")