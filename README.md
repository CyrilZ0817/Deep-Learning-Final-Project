# Research Project: Impact of Different Noises on Speech Recognition using Transformers

This project explores how differences in noise types impact model robustness and cross-domain performance, specifically using Wav2Vec2.0 models. We fine-tune the Wav2Vec2.0 models on the LibriSpeech dataset and evaluate their performance on the Common Voice dataset, which contains various noise types. The project includes data processing, model training, and evaluation steps, with results stored in the `results` folder. 

## Installation
Complete the following steps:
1. Clone the repository to your desired location using ```[git clone https://github.com/CyrilZ0817/Deep-Learning-Final-Project.git```.
2. Create python virtual environment using *requirements.txt*

## Training
1. Download the following database for training:
   - **LibriSpeech**: [LibriSpeech Dataset 100 Hr](https://www.openslr.org/12)
2. Run the training script in `training/` to train the model. The training script will automatically save the trained model in the `models/` directory.
  - The training for this project is conducted using slurm system, and the training scripts are located in `training/scripts/`. You can run the training script using the following command:
  ```sbatch training/scripts/baseline.sh```
3. To see the training results including loss changes and validation results, please refer to the `training/logs/` folder. 
  - To output the graph of training loss, you can run the jupyter notebook `training/logs/graph.ipynb`. This would allow you to upload an output log file and visualize the training loss changes.


## Evaluation
We evaluate our models using controlled benchmark noise settings based on the LibriSpeech dataset.

### Metrics
We use:
- Character Error Rate (CER) as the main metric to measure transcription accuracy
- Word Error Rate (WER) as a secondary reference

### Benchmark Noise Evaluation
We test model performance by adding different types of noise to clean LibriSpeech audio:
- Noise types:
  - stationary (e.g., air conditioner)
  - non-stationary (e.g., environmental sounds)
  - multi-speaker (babble noise)
- SNR levels:
  - 20 dB, 10 dB, 0 dB

This setup allows us to compare how different noise types and noise levels affect model performance under controlled conditions.

### Training Strategy Evaluation
We also compare models trained under different settings:
- clean-trained models
- noise-trained models (e.g., multispeaker noise)

This helps us analyze how training with noise improves robustness, especially under matching noise conditions.


## Results

### SNR Impact
We observe a clear trend across all experiments: as the signal-to-noise ratio (SNR) decreases, model performance degrades significantly.

- Higher SNR (e.g., 20 dB) → better recognition performance  
- Lower SNR (e.g., 0 dB) → much higher CER and WER  
- Performance drop is most severe under multi-speaker noise  

This confirms that speech recognition models are highly sensitive to noise intensity, especially when the noise overlaps with speech content.

---

### Noise Type Comparison
Different noise types affect the model in very different ways:

- **Stationary noise** (e.g., air conditioner, fan)  
  → minimal impact  
  → model handles this type of noise relatively well  

- **Non-stationary noise** (e.g., environmental sounds, traffic)  
  → moderate performance degradation  
  → variability makes it harder to filter  

- **Multi-speaker noise (babble)**  
  → most severe degradation  
  → competing speech signals are hardest to distinguish  

Overall, noise that resembles speech has the largest negative impact on recognition accuracy.

---

### Training Strategy Impact
We compare models trained under different conditions:

- **Clean-trained models**  
  → perform well on clean data  
  → limited robustness improvement under noisy conditions  

- **Noise-trained models (e.g., multi-speaker training)**  
  → improved performance under matching noise conditions  
  → little to no degradation on clean data  

This suggests that **training and testing noise alignment is critical** for robustness.

---

### Key Findings
- Not all noise types are equally difficult  
- Multi-speaker noise is the most challenging  
- Stationary noise is the easiest to handle  
- Noise-aware training improves robustness, but mainly under matched conditions  

---

### Limitations

Despite the promising results, several limitations should be noted:

- **Model capacity and training duration**  
  Models are relatively small and trained for limited steps due to computational constraints. Larger models and longer training may improve performance.

- **Limited diversity of noise types**  
  The noise samples used may not fully represent real-world acoustic complexity.

- **Synthetic evaluation setup**  
  Noise is artificially added to clean speech, which may not perfectly reflect real-world environments.

---

### Conclusion

This project shows that the impact of noise on speech recognition depends strongly on the type of noise, not just its intensity.

In particular:
- Stationary noise is easier for models to handle  
- Multi-speaker noise remains a major challenge  
- Training with specific noise types improves robustness under matching conditions  

These findings suggest that **carefully designed data augmentation strategies are essential for real-world speech recognition systems**, especially when computational resources are limited.

