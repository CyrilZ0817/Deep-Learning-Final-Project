import os
import torch
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer, cer

# --- Configuration ---
BASE_DIR = "results"
# Path to your MiniLibriMix metadata
METADATA_CSV = "test/MiniLibriMix/metadata/mixture_train_mix_both.csv"
# Path to original LibriSpeech (needed to find transcripts)
LIBRISPEECH_SRC = "path/to/original/LibriSpeech/train-clean-100" 
OUTPUT_FILE = "evaluation_results.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_NAME = "facebook/wav2vec2-base-960h"

def get_transcript_from_id(mixture_id):
    """
    Parses '5400-34479-0005_4973-24515-0007' to find the 
    original ground truth for the primary speaker (source 1).
    """
    try:
        # MiniLibriMix IDs are usually: [spk1-chapter-utterance]_[spk2-chapter-utterance]
        s1_id = mixture_id.split('_')[0] 
        spk, chapter, utt = s1_id.split('-')
        
        # Path to the .trans.txt file in LibriSpeech
        trans_path = os.path.join(LIBRISPEECH_SRC, spk, chapter, f"{spk}-{chapter}.trans.txt")
        
        with open(trans_path, "r") as f:
            for line in f:
                if line.startswith(s1_id):
                    return line.replace(s1_id, "").strip()
    except Exception:
        return None
    return None

def evaluate_models():
    results = []
    df_test = pd.read_csv(METADATA_CSV)
    
    # 1. Identify checkpoints
    checkpoint_paths = []
    for item in os.listdir(BASE_DIR):
        full_path = os.path.join(BASE_DIR, item)
        
        # Check if it's a directory and matches your naming
        if os.path.isdir(full_path):
            if "checkpoint" in item.lower() or "wav2vec2" in item.lower():
                checkpoint_paths.append(full_path)

    if not checkpoint_paths:
        print(f"No checkpoint folders found directly in {BASE_DIR}")
        return

    # 2. Loop through checkpoints
    # 2. Loop through checkpoints
    for ckpt in checkpoint_paths:
        print(f"\nEvaluating: {ckpt}")
        try:
            # Use only the processor
            processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL_NAME)
            model = Wav2Vec2ForCTC.from_pretrained(ckpt).to(DEVICE)
            
            predictions = []
            references = []

            for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Testing"):
                audio_path = os.path.join("test", row['mixture_path'])
                target_text = get_transcript_from_id(row['mixture_ID'])
                
                if target_text:
                    # Load and normalize
                    speech, _ = librosa.load(audio_path, sr=16000)
                    
                    # The processor handles the feature extraction internally
                    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
                    
                    with torch.no_grad():
                        logits = model(inputs.input_values.to(DEVICE)).logits
                    
                    # Decode to text
                    pred_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(pred_ids)[0]
                    
                    predictions.append(transcription.upper())
                    references.append(target_text.upper())

            if predictions:
                error_wer = wer(references, predictions)
                error_cer = cer(references, predictions)
                
                res_str = f"Model: {ckpt}\nWER: {error_wer:.4f} | CER: {error_cer:.4f}\n" + ("-"*30)
                print(res_str)
                results.append(res_str)

        except Exception as e:
            print(f"Error at {ckpt}: {e}")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results))

if __name__ == "__main__":
    evaluate_models()