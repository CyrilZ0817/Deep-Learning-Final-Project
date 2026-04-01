# Deep-Learning-Final-Project
Cyril Zhang,Seoli Kim,Yi Wang, Final project

model initial  : Hugging Face + wav2vec2（Transformer-based

need to install 
pip install torch torchaudio transformers datasets jiwer librosa soundfile

noise model from
https://github.com/microsoft/MS-SNSD

##  Key Findings

### 1. Noise significantly degrades ASR performance
- Lower SNR → higher error (CER)
- Performance drops sharply at **0 dB**

### 2. Multi-speaker noise is the hardest
- Causes the largest degradation
- Much worse than stationary or environmental noise

### 3. Noise-specific training improves robustness

| Model | Clean | Stationary | Nonstationary | Multispeaker |
|------|------:|-----------:|--------------:|-------------:|
| Clean Baseline | 0.0066 | 0.0098 | 0.0123 | 0.0676 |
| Multispeaker-trained | 0.0060 | 0.0087 | 0.0090 | 0.0473 |

 **~30% CER reduction on multispeaker noise**

### 4. No degradation on clean data
- Noise-trained model maintains clean performance

---

##  Methods

### Model
- :contentReference[oaicite:0]{index=0} (facebook/wav2vec2-base-960h)

### Dataset
- LibriSpeech (clean subset)

### Noise Dataset
- MS-SNSD noise dataset

### Evaluation Metric
- Character Error Rate (CER)

---

### 1. Baseline Evaluation
- Pretrained model tested on:
  - Clean audio
  - Noisy audio (3 noise types)

### 2. SNR Analysis
- Tested at:
  - 20 dB
  - 10 dB
  - 0 dB

### 3. Training Strategies
- Clean-only fine-tuning
- Multispeaker noise training (10 dB)

---

## Results

### SNR Impact
- CER increases as SNR decreases
- Multi-speaker noise shows the steepest degradation

### Noise Type Comparison
- Stationary noise: minimal impact
- Non-stationary noise: moderate impact
- Multi-speaker noise: severe impact

### Training Impact
- Clean training: no significant robustness improvement
- Multispeaker training: significant improvement under matched condition

---
```text
## Project Structure
final project/
├── train_clean_baseline.py
├── train_multispeaker_baseline.py
├── test_trained_clean_baseline.py
├── test_multispeaker_trained_model.py
├── snr_analysis.py
├── noises/
│ ├── AirConditioner_2.wav
│ ├── AirportAnnouncements_1.wav
│ └── Babble_1.wav
├── wav2vec2-clean-baseline/
├── wav2vec2-multispeaker-10db/
└── README.md
```
