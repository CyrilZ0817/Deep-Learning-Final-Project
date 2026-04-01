
We evaluate ASR performance under three noise types (stationary, non-stationary, and multi-speaker) across different SNR levels (20 dB, 10 dB, and 0 dB).

As shown in Figure X, the Character Error Rate (CER) increases as SNR decreases for all noise types, confirming that lower signal quality leads to degraded recognition performance.

Among all noise types, multi-speaker noise has the most significant impact. At 0 dB, the CER reaches approximately 0.43, which is more than five times higher than that of stationary and non-stationary noise. This suggests that competing speech signals are far more disruptive than environmental noise.

In contrast, stationary and non-stationary noise result in relatively minor degradation, even at lower SNR levels. This indicates that the model is capable of filtering out noise that does not share speech-like characteristics.

Overall, these results demonstrate that speech-like interference is the primary challenge for robust speech recognition systems.
