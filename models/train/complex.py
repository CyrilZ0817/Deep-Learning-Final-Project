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


# ---------------------------------------------------------------------------
# Config & seeds
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ACTIVE_TYPE = "complex"
profile = config["training"]["types"][ACTIVE_TYPE]

SEED = config["training"]["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(SCRIPT_DIR, "data/librispeech_clean_16k/train")
full_ds = load_from_disk(DATA_PATH)
print(f"Loaded dataset: {len(full_ds)} samples.")

split_ds = full_ds.train_test_split(test_size=0.1, seed=SEED)
train_raw = split_ds["train"]
valid_raw = split_ds["test"]
print(f"Split — train: {len(train_raw)}, valid: {len(valid_raw)}")

train_dataset = train_raw.to_iterable_dataset().shuffle(buffer_size=1000, seed=SEED)
valid_dataset = valid_raw.to_iterable_dataset()

# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------
processor = Wav2Vec2Processor.from_pretrained(config["model"]["name"])
processor.feature_extractor.do_normalize = True

# ---------------------------------------------------------------------------
# Noise loading
# ---------------------------------------------------------------------------
def load_noises() -> dict:
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
        # Normalise noise to unit RMS so SNR scaling is well-conditioned
        noise_rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
        audio = audio / noise_rms
        loaded[fname] = audio
    return loaded


LOADED_NOISES = load_noises()
NOISE_NAMES = list(LOADED_NOISES.keys())
if len(NOISE_NAMES) == 0:
    raise RuntimeError("No valid noise files found.")
print(f"Loaded {len(NOISE_NAMES)} noise files.")

# ---------------------------------------------------------------------------
# Audio mixing helpers
# ---------------------------------------------------------------------------
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float32) ** 2) + 1e-8))


def align_noise(noise: np.ndarray, target_len: int) -> np.ndarray:
    if len(noise) < target_len:
        noise = np.tile(noise, int(np.ceil(target_len / len(noise))))
    start = random.randint(0, len(noise) - target_len)
    return noise[start : start + target_len]


def mix_with_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean = np.asarray(clean, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)

    noise_aligned = align_noise(noise, len(clean))
    clean_rms_val = rms(clean)
    noise_rms_val = rms(noise_aligned)

    target_noise_rms = clean_rms_val / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms_val + 1e-8)

    mixed = clean + noise_aligned * scale
    mixed = np.nan_to_num(mixed, nan=0.0, posinf=1.0, neginf=-1.0)

    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)

# ---------------------------------------------------------------------------
# CTC validity helpers  (must match wav2vec2-base CNN stack exactly)
# ---------------------------------------------------------------------------
def ctc_output_length(input_len: int) -> int:
    out = input_len
    for kernel, stride in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
        out = (out - kernel) // stride + 1
    return out


CTC_SAFETY_MARGIN = 2  # output frames must be >= label_len * this


def is_ctc_valid(example) -> bool:
    label_len = len(example["labels"])
    if label_len == 0:
        return False
    # input_values here is the processor output (float array, length = raw samples)
    out_len = ctc_output_length(len(example["input_values"]))
    return out_len >= label_len * CTC_SAFETY_MARGIN

# ---------------------------------------------------------------------------
# On-the-fly augmentation map
# ---------------------------------------------------------------------------
def mix_on_the_fly(batch):
    clean = np.asarray(batch["clean_audio"], dtype=np.float32)
    text  = str(batch["clean_text"]).upper().strip()

    # Guard: skip silent or corrupted clean audio
    if np.max(np.abs(clean)) < 0.01 or np.isnan(clean).any() or np.isinf(clean).any():
        batch["input_values"] = np.zeros(1, dtype=np.float32)
        batch["labels"]       = []
        batch["chosen_noise"] = ""
        batch["chosen_snr"]   = 0
        batch["clean_text"]   = text
        return batch

    noise_name = random.choice(NOISE_NAMES)
    noise = LOADED_NOISES[noise_name]
    snr   = random.randint(profile["snr_range"]["min"], profile["snr_range"]["max"])

    mixed = mix_with_snr(clean, noise, snr)

    input_values = processor(
        mixed,
        sampling_rate=16000,
        return_tensors="np",
    ).input_values[0].astype(np.float32)

    # Fix: NaN guard must be set BEFORE we assign to batch["labels"]
    if np.isnan(input_values).any() or np.isinf(input_values).any():
        input_values = np.zeros(1, dtype=np.float32)  # is_ctc_valid will drop this
        labels = []
    else:
        labels = processor.tokenizer(text).input_ids

    batch["input_values"] = input_values
    batch["labels"]       = labels
    batch["chosen_noise"] = noise_name
    batch["chosen_snr"]   = snr
    batch["clean_text"]   = text
    return batch


train_dataset = train_dataset.map(mix_on_the_fly).filter(is_ctc_valid)
valid_dataset = valid_dataset.map(mix_on_the_fly).filter(is_ctc_valid)

# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]}          for f in features]

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

        batch["labels"]         = labels
        batch["attention_mask"] = batch["attention_mask"].long()
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred):
    pred_ids  = np.argmax(pred.predictions, axis=-1)
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer_scores = [cer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
    return {"cer": float(np.mean(cer_scores))}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = Wav2Vec2ForCTC.from_pretrained(
    config["model"]["name"],
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    # Fix: True silences -inf paths early in training, preventing NaN loss.
    # Once training is stable you can set this back to False for a final run.
    ctc_zero_infinity=True,
)

# Freeze the CNN feature encoder for the first phase of training.
#
# Why: the encoder is pre-trained on 960h of clean speech and is already
# excellent at extracting features.  Fine-tuning it with a noisy task-specific
# loss at lr=3e-4 will corrupt those representations and cause divergence.
# We only want to adapt the transformer layers and the CTC head.
# If you later want to unfreeze it (e.g. for a second-phase run), do so at a
# much lower lr (1e-5) after the rest of the model has converged.
model.freeze_feature_encoder()

# ---------------------------------------------------------------------------
# Training arguments
#
# Learning rate recommendation: 3e-4
#
# Reasoning:
#   - With the feature encoder frozen, only ~85M of the 94M parameters are
#     being updated (transformer layers + CTC projection).
#   - wav2vec2-base was pre-trained with Adam at 1e-3; fine-tuning on a
#     downstream CTC task typically uses 1e-4 to 5e-4.
#   - 3e-4 with a linear warmup of 500 steps gives fast convergence without
#     instability.  The original wav2vec2 fine-tuning paper (Baevski et al.)
#     used 1e-4 on LibriSpeech 100h clean; we go slightly higher because
#     noise augmentation acts as a regulariser and allows a more aggressive lr.
#   - If you see loss spiking after warmup, drop to 1e-4.
#   - If you later unfreeze the encoder for phase 2, use 1e-5 for everything.
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4

training_args = TrainingArguments(
    output_dir=os.path.join(SCRIPT_DIR, profile["output_dir"] + "_fixed"),

    # Batch / steps
    per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
    per_device_eval_batch_size=4,
    max_steps=int(config["training"]["max_steps"]),

    # Optimiser
    learning_rate=LEARNING_RATE,
    warmup_steps=config["training"]["warmup_steps"],
    # Linear decay to 0 over training gives a clean convergence curve
    lr_scheduler_type="linear",

    # Gradient stability
    max_grad_norm=1.0,
    fp16=False,  # keep off; wav2vec2 CTC can produce inf in fp16 without careful scaling

    # Logging / saving
    logging_steps=50,
    logging_strategy="steps",
    eval_strategy="steps",
    eval_steps=int(config["training"]["eval_steps"]),
    save_steps=int(config["training"]["save_steps"]),
    save_total_limit=2,

    # Best-model tracking
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,

    report_to="none",
    remove_unused_columns=False,
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset.take(100),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ---------------------------------------------------------------------------
# Pre-flight sanity check
# ---------------------------------------------------------------------------
print("\n--- Pre-flight batch check ---")
first_batch = next(iter(trainer.get_train_dataloader()))
iv   = first_batch["input_values"]
labs = first_batch["labels"]

print(f"input_values  shape : {iv.shape}")
print(f"labels        shape : {labs.shape}")
print(f"attention_mask dtype: {first_batch['attention_mask'].dtype}")
print(f"input_values  NaN={torch.isnan(iv).any()}, Inf={torch.isinf(iv).any()}")
print(f"input_values  range : [{iv.min():.3f}, {iv.max():.3f}]")

non_pad_per_sample = (labs != -100).sum(dim=-1).tolist()
print(f"label lengths (non-padding): {non_pad_per_sample}")

# Verify every sample in the batch satisfies the CTC constraint
for i, label_len in enumerate(non_pad_per_sample):
    frame_len = ctc_output_length(iv.shape[-1])
    ratio = frame_len / max(label_len, 1)
    status = "OK" if frame_len >= label_len * CTC_SAFETY_MARGIN else "FAIL"
    print(f"  sample {i}: frames={frame_len}, labels={label_len}, ratio={ratio:.2f} [{status}]")

# Run this right before trainer.train() and paste the output
print("\n--- Deep probe ---")

# 1. Decode the labels to see what transcripts look like
labs_decoded = []
for i in range(labs.shape[0]):
    ids = labs[i][labs[i] != -100].tolist()
    text = processor.tokenizer.decode(ids)
    print(f"  sample {i} transcript ({len(ids)} tokens): '{text[:120]}'")

# 2. Check logits for NaN/Inf
model.eval()
with torch.no_grad():
    probe_batch_dev = {k: v.to(model.device) for k, v in first_batch.items()}
    # Forward WITHOUT labels to check if logits themselves are clean
    logits = model(
        input_values=probe_batch_dev["input_values"],
        attention_mask=probe_batch_dev["attention_mask"],
    ).logits
    print(f"\nLogits shape : {logits.shape}")
    print(f"Logits NaN   : {torch.isnan(logits).any()}")
    print(f"Logits Inf   : {torch.isinf(logits).any()}")
    print(f"Logits range : [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Log-softmax (what CTC actually receives)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    print(f"Log-probs NaN: {torch.isnan(log_probs).any()}")
    print(f"Log-probs Inf: {torch.isinf(log_probs).any()}")
    print(f"Log-probs min: {log_probs.min():.3f}  (very negative = collapsed logits)")

# 3. Check CTC loss manually with exact lengths
import torch.nn.functional as F
log_probs_t = log_probs.transpose(0, 1)  # (T, N, C)
input_lengths = probe_batch_dev["attention_mask"].sum(dim=-1)
input_lengths_ctc = torch.tensor(
    [ctc_output_length(l.item()) for l in input_lengths], dtype=torch.long
)
target_lengths = (probe_batch_dev["labels"] != -100).sum(dim=-1)
targets = probe_batch_dev["labels"].clone()
targets[targets == -100] = 0  # CTC doesn't accept -100

print(f"\nCTC manual check:")
print(f"  input_lengths_ctc : {input_lengths_ctc.tolist()}")
print(f"  target_lengths    : {target_lengths.tolist()}")

ctc_loss = F.ctc_loss(
    log_probs_t,
    targets,
    input_lengths_ctc,
    target_lengths,
    blank=processor.tokenizer.pad_token_id,
    reduction="none",
    zero_infinity=False,
)
print(f"  per-sample CTC loss: {ctc_loss.tolist()}")
ctc_loss_zi = F.ctc_loss(
    log_probs_t,
    targets,
    input_lengths_ctc,
    target_lengths,
    blank=processor.tokenizer.pad_token_id,
    reduction="none",
    zero_infinity=True,
)
print(f"  per-sample CTC loss (zero_infinity=True): {ctc_loss_zi.tolist()}")

# Probe forward pass
model.train()
probe_out = model(**{k: v.to(model.device) for k, v in first_batch.items()})
probe_loss = probe_out.loss.item()
print(f"\nProbe loss: {probe_loss:.4f}")
if np.isnan(probe_loss):
    raise RuntimeError(
        "Probe loss is NaN even after all fixes. "
        "Re-run prepare_clean_data.py with the updated script before training."
    )
if probe_loss == 0.0:
    print("WARNING: probe loss is 0.0 — check that labels are not all padding.")
print("Pre-flight passed. Starting training...\n")



# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
trainer.train()

# ---------------------------------------------------------------------------
# Final eval & save
# ---------------------------------------------------------------------------
metrics = trainer.evaluate()
print("\nFinal evaluation:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

save_dir = os.path.join(SCRIPT_DIR, profile["output_dir"] + "_fixed")
trainer.save_model(save_dir)
processor.save_pretrained(save_dir)
print(f"\nModel saved to: {save_dir}")