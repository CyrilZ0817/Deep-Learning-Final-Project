import os
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer

# --- Configuration ---
BASE_DIR = "results"
WAV_DIR = "test/MiniLibriMix/train/mix_both"
OUTPUT_FILE = "evaluation_whisper_ground_truth.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models
W2V2_BASE_NAME = "facebook/wav2vec2-base-960h"
WHISPER_MODEL_NAME = "openai/whisper-large-v3"

def evaluate_with_whisper():
    results = []
    
    # 1. Initialize Whisper for Ground Truth Generation
    print(f"Loading Whisper ({WHISPER_MODEL_NAME}) to generate ground truth...")
    whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME).to(DEVICE)
    
    # 2. Initialize Wav2Vec2 Processor
    w2v2_processor = Wav2Vec2Processor.from_pretrained(W2V2_BASE_NAME)

    # 3. Get all wav files
    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith(".wav")]
    if not wav_files:
        print(f"No wav files found in {WAV_DIR}")
        return

    # 4. Generate Whisper "Ground Truth" Transcripts first (to save time)
    print(f"Generating ground truth for {len(wav_files)} files...")
    ground_truth_map = {}
    for filename in tqdm(wav_files, desc="Whisper Transcribing"):
        path = os.path.join(WAV_DIR, filename)
        speech, _ = librosa.load(path, sr=16000)
        
        # Inside the Whisper loop
        input_features = whisper_processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)

        # ADD THIS LINE to match precision
        input_features = input_features.to(whisper_model.dtype) 

        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)
        
        transcript = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        ground_truth_map[filename] = transcript.upper().strip()

    # Clear Whisper from GPU memory to make room for Wav2Vec2
    del whisper_model
    torch.cuda.empty_cache()

    # 5. Identify Wav2Vec2 Checkpoints
    checkpoint_paths = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) 
                        if os.path.isdir(os.path.join(BASE_DIR, d)) and ("checkpoint" in d.lower() or "wav2vec2" in d.lower())]

    # 6. Evaluate Each Checkpoint
    for ckpt in checkpoint_paths:
        print(f"\nEvaluating Wav2Vec2 Checkpoint: {ckpt}")
        try:
            model = Wav2Vec2ForCTC.from_pretrained(ckpt).to(DEVICE)
            model.eval()
            
            predictions = []
            references = []

            for filename in tqdm(wav_files, desc="Wav2Vec2 Inference"):
                path = os.path.join(WAV_DIR, filename)
                speech, _ = librosa.load(path, sr=16000)
                
                inputs = w2v2_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    logits = model(inputs.input_values.to(DEVICE)).logits
                
                pred_ids = torch.argmax(logits, dim=-1)
                transcription = w2v2_processor.batch_decode(pred_ids)[0]
                
                predictions.append(transcription.upper().strip())
                references.append(ground_truth_map[filename])

            # Calculate Metrics
            if predictions:
                res_wer = wer(references, predictions)
                res_cer = cer(references, predictions)
                
                output = f"Model: {ckpt}\nWER: {res_wer:.4f} | CER: {res_cer:.4f}\n" + ("-"*30)
                print(output)
                results.append(output)
            
            # Clean up memory for next checkpoint
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {ckpt}: {e}")

    # 7. Final Output
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results))
    print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate_with_whisper()