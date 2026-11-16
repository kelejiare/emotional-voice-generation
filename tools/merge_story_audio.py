import os
import argparse
import torch
import torchaudio

ORDER = [
    "story_0001_neutral.wav",
    "story_0002_happy.wav",
    "story_0003_surprise.wav",
    "story_0004_sad.wav",
    "story_0005_angry.wav",
    "story_0006_calm.wav",
]


def load_wav(path, target_sr=None):

    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if target_sr and sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr

    return wav, sr


def normalize_rms(wav, target_db=-20):

    rms = torch.sqrt(torch.mean(wav ** 2))

    db = 20 * torch.log10(rms + 1e-9)

    gain = 10 ** ((target_db - db.item()) / 20)

    return wav * gain


def crossfade(a, b, sr, ms=30):

    n = int(sr * ms / 1000)

    if n == 0:
        return torch.cat([a, b], dim=1)

    fade_out = torch.linspace(1, 0, n)
    fade_in = torch.linspace(0, 1, n)

    mid = a[:, -n:] * fade_out + b[:, :n] * fade_in

    return torch.cat([a[:, :-n], mid, b[:, n:]], dim=1)


def merge_audio(in_dir, out_path):

    wavs = []

    sr = None

    for name in ORDER:

        path = os.path.join(in_dir, name)

        if not os.path.exists(path):
            raise RuntimeError(f"Missing {path}")

        w, sr = load_wav(path, sr)

        w = normalize_rms(w)

        wavs.append(w)

    out = wavs[0]

    for w in wavs[1:]:
        out = crossfade(out, w, sr)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torchaudio.save(out_path, out, sr)

    print("Saved:", out_path)
    print("Duration:", out.shape[1] / sr, "seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)

    parser.add_argument("--output_wav", required=True)

    args = parser.parse_args()

    merge_audio(args.input_dir, args.output_wav)