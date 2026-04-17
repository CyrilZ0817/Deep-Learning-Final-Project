#!/bin/bash

# --- SLURM Resource Request ---
#SBATCH --job-name=clean_data
#SBATCH --output=logs/clean_data.out
#SBATCH --error=logs/clean_data.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# SET THE DIRECTORY
PROJECT_ROOT="/common/users/sk2779/Deep-Learning-Final-Project"
cd "$PROJECT_ROOT"

# Create a virtual environment if it doesn't exist
if [ ! -d "venv462" ]; then
    python3 -m venv venv462
    source venv462/bin/activate
    pip install requirements.txt --upgrade
fi

# Activate the environment
source venv462/bin/activate


# --- 2. Training Execution ---
echo "Starting training at: $(date)"

# Run with -u to get real-time log updates in your .out file
python3 "${PROJECT_ROOT}/models/train/prepare_clean_data.py"

echo "Job finished at: $(date)"