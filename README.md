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

