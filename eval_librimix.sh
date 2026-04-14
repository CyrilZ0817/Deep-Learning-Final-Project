#!/bin/bash

# --- SLURM Resource Request ---
#SBATCH --job-name=eval_librimix
#SBATCH --output=logs/eval.out
#SBATCH --error=logs/eval.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# SET THE DIRECTORY
PROJECT_ROOT=""
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
mkdir -p "logs/static"

# Run with -u to get real-time log updates in your .out file
python3 "${PROJECT_ROOT}/models/train/static.py"

echo "Job finished at: $(date)"