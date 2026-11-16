import torch
import glob
import os
import argparse
import soundfile as sf
import numpy as np
from tqdm import tqdm

def load_utmos():
    print("[INFO] Loading UTMOS from TorchHub...")
    model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0",
        "utmos22_strong",
        trust_repo=True
    )
    model.eval()
    return model

def load_wav(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = torch.from_numpy(wav).float().unsqueeze(0)
    return wav, sr

def score_wav(model, wav_path):
    wav, sr = load_wav(wav_path)
    with torch.no_grad():
        score = model(wav, sr)
    return score.item()

def evaluate_folder(folder):
    wavs = sorted(glob.glob(os.path.join(folder, "*.wav")))
    if len(wavs) == 0:
        print("No wav files found.")
        return

    model = load_utmos()

    scores = []
    for wav in tqdm(wavs):
        try:
            s = score_wav(model, wav)
            scores.append(s)
        except Exception as e:
            print(f"Error on {wav}: {e}")

    if len(scores) == 0:
        print("No valid scores computed.")
        return

    print("\n==== UTMOS RESULT ====")
    print("Files:", len(scores))
    print("Mean MOS:", sum(scores) / len(scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", required=True)
    args = parser.parse_args()

    evaluate_folder(args.wav_dir)