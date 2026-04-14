import os
import torch
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer, cer

# --- Configuration ---
BASE_DIR = "models/train/results_old"
# Path to your MiniLibriMix metadata
METADATA_CSV = "test/MiniLibriMix/metadata/mixture_train_mix_both.csv"
# Path to original LibriSpeech (needed to find transcripts)
LIBRISPEECH_SRC = "path/to/original/LibriSpeech/train-clean-100" 
OUTPUT_FILE = "evaluation_results.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    for root, dirs, files in os.walk(BASE_DIR):
        if "checkpoint-" in root or "config.json" in files:
            if "wav2vec2" in root.lower():
                checkpoint_paths.append(root)

    if not checkpoint_paths:
        print("No wav2vec2 checkpoints found.")
        return

    # 2. Loop through checkpoints
    for ckpt in checkpoint_paths:
        print(f"\nEvaluating: {ckpt}")
        try:
            processor = Wav2Vec2Processor.from_pretrained(ckpt)
            model = Wav2Vec2ForCTC.from_pretrained(ckpt).to(DEVICE)
            
            predictions = []
            references = []

            for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Testing"):
                # Construct absolute path to the mixture
                # 'test/' is the prefix based on your directory description
                audio_path = os.path.join("test", row['mixture_path'])
                
                # Fetch ground truth transcript
                target_text = get_transcript_from_id(row['mixture_ID'])
                
                if target_text:
                    speech, _ = librosa.load(audio_path, sr=16000)
                    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
                    
                    with torch.no_grad():
                        logits = model(inputs.input_values.to(DEVICE)).logits
                    
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