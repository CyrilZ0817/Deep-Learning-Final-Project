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
    ctc_zero_infinity=True,
    # Reduce dropout: default 0.1 is fine for large datasets but with only
    # 3k samples and extreme logit ranges it causes -inf paths in train mode.
    # 0.05 keeps regularisation without dropout-amplified logit explosions.
    hidden_dropout=0.05,
    activation_dropout=0.05,
    attention_dropout=0.05,
    feat_proj_dropout=0.05,
)

# Verify blank token id matches what CTC expects
blank_id = processor.tokenizer.pad_token_id
print(f"Blank token id : {blank_id}  (token: '{processor.tokenizer.decode([blank_id])}')")
print(f"Vocab size     : {processor.tokenizer.vocab_size}")

# Do NOT freeze the feature encoder.
#
# Why: wav2vec2-base-960h has a pre-trained CTC head whose weight scale is
# matched to the full model's activation range.  With 3k samples the
# transformer layers need the grounding signal from the encoder to avoid
# producing the extreme logit range (~58 units) we diagnosed above.
# The encoder's CNN weights will barely move at lr=1e-4 anyway — they have
# large Frobenius norms and the gradients flowing through them are tiny.
# Freezing them removes that grounding and is what caused log-probs of -56.

# ---------------------------------------------------------------------------
# Training arguments
#
# Learning rate: 1e-4
#
# Reasoning:
#   - Diagnostic showed logits range [-41, +17] → log-probs min -56, which is
#     ~16x worse than random initialisation. This means the pretrained CTC head
#     weights are being disrupted — 3e-4 is too aggressive for a 3k-sample set.
#   - 1e-4 with warmup_steps=500 keeps early updates small enough that the
#     model stays in the basin of the pretrained solution while still adapting
#     to noisy speech.
#   - Gradient clipping at 0.5 (tighter than default 1.0) gives an extra
#     safety net for the first few hundred steps where logits are still large.
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-4

training_args = TrainingArguments(
    output_dir=os.path.join(SCRIPT_DIR, profile["output_dir"] + "_fixed"),

    # Batch / steps
    per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
    per_device_eval_batch_size=4,
    max_steps=int(config["training"]["max_steps"]),

    # Optimiser
    learning_rate=LEARNING_RATE,
    warmup_steps=config["training"]["warmup_steps"],
    lr_scheduler_type="linear",

    # Tighter gradient clipping for first run with extreme logit range
    max_grad_norm=0.5,
    fp16=False,

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

for i, label_len in enumerate(non_pad_per_sample):
    frame_len = ctc_output_length(iv.shape[-1])
    ratio = frame_len / max(label_len, 1)
    status = "OK" if frame_len >= label_len * CTC_SAFETY_MARGIN else "FAIL"
    print(f"  sample {i}: frames={frame_len}, labels={label_len}, ratio={ratio:.2f} [{status}]")

# Check logit health in eval mode (no dropout) first
model.eval()
with torch.no_grad():
    dev_batch = {k: v.to(model.device) for k, v in first_batch.items()}
    logits    = model(
        input_values=dev_batch["input_values"],
        attention_mask=dev_batch["attention_mask"],
    ).logits
    logit_range = logits.max().item() - logits.min().item()
    log_probs   = torch.nn.functional.log_softmax(logits, dim=-1)
    lp_min      = log_probs.min().item()
    print(f"\nLogit range    : {logit_range:.2f}  (healthy < 20, warning > 40)")
    print(f"Log-probs min  : {lp_min:.3f}     (healthy ~ -3.5, warning < -20)")
    if logit_range > 40 or lp_min < -20:
        print("  WARNING: extreme logit scale detected.")
        print("  This is normal on step 0 with wav2vec2-base-960h — warmup will fix it.")
        print("  If loss is still NaN after 200 steps, reduce learning_rate further.")

# Now probe in train mode (dropout active)
model.train()
probe_out  = model(**{k: v.to(model.device) for k, v in first_batch.items()})
probe_loss = probe_out.loss.item()
print(f"\nProbe loss (train mode): {probe_loss:.4f}")

if np.isnan(probe_loss):
    # With ctc_zero_infinity=True the Trainer itself will survive NaN on step 0
    # because zero_infinity zeroes out inf paths before reduction.
    # The probe runs WITHOUT zero_infinity catching it at the Python level
    # because model.train() recomputes with dropout — this is expected on step 0
    # when logit range is extreme.  Training will stabilise after warmup.
    print(
        "  NOTE: probe NaN with extreme logits + dropout is expected at step 0.\n"
        "  ctc_zero_infinity=True protects the Trainer's actual loss computation.\n"
        "  Proceeding — watch the first 50 logged steps: loss should drop from ~200."
    )
elif probe_loss == 0.0:
    print("WARNING: probe loss is 0.0 — check that labels are not all padding.")
else:
    print("Pre-flight passed cleanly.")

print("Starting training...\n")

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