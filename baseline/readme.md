
We evaluate ASR performance under three noise types (stationary, non-stationary, and multi-speaker) across different SNR levels (20 dB, 10 dB, and 0 dB).

As shown in Figure X, the Character Error Rate (CER) increases as SNR decreases for all noise types, confirming that lower signal quality leads to degraded recognition performance.

Among all noise types, multi-speaker noise has the most significant impact. At 0 dB, the CER reaches approximately 0.43, which is more than five times higher than that of stationary and non-stationary noise. This suggests that competing speech signals are far more disruptive than environmental noise.

In contrast, stationary and non-stationary noise result in relatively minor degradation, even at lower SNR levels. This indicates that the model is capable of filtering out noise that does not share speech-like characteristics.

Overall, these results demonstrate that speech-like interference is the primary challenge for robust speech recognition systems.

## Code Structure

This folder contains baseline evaluation scripts for analyzing the performance of a pretrained ASR model (wav2vec2) under clean and noisy conditions. It includes simple tests, noise robustness comparisons, and systematic SNR analysis.

- `test_asr.py`: Runs basic ASR inference on clean LibriSpeech samples using a pretrained wav2vec2 model. This script verifies the pipeline and provides a clean baseline CER for comparison.

- `test_clean_vs_noise.py`: Evaluates ASR performance on a single clean audio sample and its noisy versions (stationary, non-stationary, and multi-speaker noise) at a fixed SNR level. This script demonstrates how different noise types affect recognition.

- `batch_noise_test.py`: Performs batch evaluation on multiple samples to compare ASR performance across clean and different noise types at a fixed SNR (10 dB). Outputs average CER and generates a bar chart for comparison.

- `snr_analysis.py`: Conducts systematic evaluation of ASR performance under different noise types and SNR levels (20, 10, 0 dB). Computes average CER, saves results to JSON, and generates performance curves.

## Self-Train Model

This folder contains scripts for fine-tuning the pretrained wav2vec2 model and evaluating the trained models under different noise conditions.

- `train_clean_baseline.py`: Fine-tunes the model on clean LibriSpeech data to obtain a clean-only baseline model.

- `train_multispeaker_baseline.py`: Fine-tunes the model on speech mixed with multi-speaker (babble) noise at 10 dB to improve robustness under speech-like interference.

- `test_trained_clean_baseline.py`: Evaluates the clean-trained model on clean and noisy data (stationary, non-stationary, and multi-speaker noise) at 10 dB.

- `test_multispeaker_trained_model.py`: Evaluates the multi-speaker-trained model under the same conditions to compare robustness improvements.

## Figures

This folder contains visualization and result files generated from baseline experiments.

- `cer_comparison.png`: Bar chart comparing ASR performance across noise types at 10 dB.

- `snr_analysis.png`: Line plot showing CER trends across different SNR levels for each noise type.

- `snr_results.json`: Numerical results used for plotting SNR analysis.
