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


## Evaluation
 


## Results

