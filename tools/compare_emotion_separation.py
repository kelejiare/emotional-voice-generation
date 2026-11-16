import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH = "/data/lizhen/EmoVoice/microsoftwavlm-large"


def load_model():
    extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    model.to(DEVICE)
    model.eval()
    return extractor, model


def extract_embedding(wav_path, extractor, model):
    speech, sr = torchaudio.load(wav_path)

    if speech.shape[0] > 1:
        speech = torch.mean(speech, dim=0)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech)

    inputs = extractor(
        speech.numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb


def compute_inter_distance(wav_dir):

    extractor, model = load_model()

    wav_files = sorted([
        os.path.join(wav_dir, f)
        for f in os.listdir(wav_dir)
        if f.endswith(".wav")
    ])

    embeddings = []

    for wav in wav_files:
        emb = extract_embedding(wav, extractor, model)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    inter = []

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            inter.append(dist)

    return np.mean(inter)


if __name__ == "__main__":

    BASE = "outputs/runs/ljh_en_Aplus_strong"

    models = {
        "CosyVoice3": f"{BASE}/cosyvoice3_instruct/pred_audio",
        "EmoVoice": f"{BASE}/emovoice_Aplus/pred_audio/neutral_prompt_speech",
        "EmoVoicePP": f"{BASE}/emovoicePP_Aplus/pred_audio/neutral_prompt_speech",
        "EmoVoicePP+VC": f"{BASE}/emovoicePP_Aplus/pred_audio_vc/neutral_prompt_speech"
    }

    results = {}

    for name, path in models.items():
        print(f"\n[INFO] Evaluating {name}")
        dist = compute_inter_distance(path)
        results[name] = dist
        print("Mean Inter-class Distance:", dist)

    print("\n===== FINAL COMPARISON =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")