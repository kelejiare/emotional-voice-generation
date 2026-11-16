import os
import argparse
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


def get_label(path):
    return os.path.basename(path).split("_")[-1].replace(".wav", "")


def extract_embeddings(model_path, wav_dir):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading model: {model_path}")

    extractor = AutoFeatureExtractor.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True
    )

    model.to(device)
    model.eval()

    embeddings = []
    labels = []

    wav_files = sorted([
        os.path.join(wav_dir, f)
        for f in os.listdir(wav_dir)
        if f.endswith(".wav")
    ])

    print(f"[INFO] Found {len(wav_files)} wav files")

    for wav in tqdm(wav_files):

        speech, sr = torchaudio.load(wav)

        if speech.shape[0] > 1:
            speech = torch.mean(speech, dim=0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(speech)
            sr = 16000

        inputs = extractor(
            speech.numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # 🔥 通用 embedding 提取方式
        if hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state.mean(dim=1)
        else:
            emb = outputs[0].mean(dim=1)

        emb = emb.squeeze().cpu().numpy()

        embeddings.append(emb)
        labels.append(get_label(wav))

    return np.array(embeddings), labels


def compute_separation(embeddings, labels):

    intra = []
    inter = []

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):

            dist = np.linalg.norm(embeddings[i] - embeddings[j])

            if labels[i] == labels[j]:
                intra.append(dist)
            else:
                inter.append(dist)

    intra_mean = np.mean(intra) if len(intra) > 0 else 0
    inter_mean = np.mean(inter) if len(inter) > 0 else 0

    sep = inter_mean / intra_mean if intra_mean != 0 else 0

    print("\n===== Embedding Analysis =====")
    print("Intra-class distance:", intra_mean)
    print("Inter-class distance:", inter_mean)
    print("Separation Score:", sep)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, required=True)
    args = parser.parse_args()

    embeddings, labels = extract_embeddings(
        args.model_path,
        args.wav_dir
    )

    compute_separation(embeddings, labels)