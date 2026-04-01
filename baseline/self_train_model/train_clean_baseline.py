import io
import numpy as np
import soundfile as sf
import torch

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

# 1. settings

MODEL_NAME = "facebook/wav2vec2-base-960h"
TRAIN_SPLIT = "train.100[:1%]"
VALID_SPLIT = "validation[:1%]"
OUTPUT_DIR = "./wav2vec2-clean-baseline"


# 2. load dataset

train_dataset = load_dataset("librispeech_asr", "clean", split=TRAIN_SPLIT)
valid_dataset = load_dataset("librispeech_asr", "clean", split=VALID_SPLIT)

train_dataset = train_dataset.cast_column("audio", Audio(decode=False))
valid_dataset = valid_dataset.cast_column("audio", Audio(decode=False))


# 3. load processor

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# 4. helper: decode audio safely
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

    batch["speech"] = audio.astype("float32")
    batch["sampling_rate"] = sr
    batch["target_text"] = batch["text"]
    return batch


train_dataset = train_dataset.map(load_audio_from_record)
valid_dataset = valid_dataset.map(load_audio_from_record)


# 5. prepare features
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


# 6. data collator
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

# 7. metrics
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer_scores = [cer(ref, hyp) for ref, hyp in zip(label_str, pred_str)]
    avg_cer = float(np.mean(cer_scores))
    return {"cer": avg_cer}


# 8. load model
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

model.freeze_feature_encoder()


# 9. training args
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    num_train_epochs=2,
    learning_rate=1e-4,
    warmup_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)


# 10. trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# 11. train

trainer.train()

# 12. final eval
metrics = trainer.evaluate()
print("\nFinal evaluation:")
print(metrics)

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\nSaved model to: {OUTPUT_DIR}")