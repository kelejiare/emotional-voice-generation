import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


@dataclass
class FileMetric:
    path: str
    duration_s: float
    rms_db: float
    silence_ratio: float
    speaker_cosine: float
    cer: Optional[float]


def list_eval_wavs(audio_dir: str) -> List[str]:
    all_wavs = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")])
    if not all_wavs:
        return []
    base_set = set(all_wavs)
    chosen = []
    for f in all_wavs:
        if f.endswith("_0.wav"):
            base = f[:-6] + ".wav"
            if base in base_set:
                continue
        chosen.append(os.path.join(audio_dir, f))
    return chosen


def load_mono(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def rms_db(wav: torch.Tensor, eps: float = 1e-12) -> float:
    x = wav.float()
    v = torch.mean(x * x).item()
    return 10.0 * math.log10(max(v, eps))


def silence_ratio(wav: torch.Tensor, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0, thr_db: float = -40.0) -> float:
    x = wav.squeeze(0).float()
    frame = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if x.numel() < frame:
        return 0.0
    n = 1 + (x.numel() - frame) // hop
    silent = 0
    for i in range(n):
        seg = x[i * hop : i * hop + frame]
        v = torch.mean(seg * seg).item()
        db = 10.0 * math.log10(max(v, 1e-12))
        if db < thr_db:
            silent += 1
    return float(silent) / float(n)


def normalize_zh(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    s = "".join(ch for ch in s if ("\u4e00" <= ch <= "\u9fff") or ch.isalnum())
    return s


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def cer(gt: str, pred: str) -> float:
    g = normalize_zh(gt)
    p = normalize_zh(pred)
    if len(g) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return float(levenshtein(g, p)) / float(len(g))


def build_ort_session(onnx_path: str) -> onnxruntime.InferenceSession:
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(onnx_path, sess_options=opt, providers=["CPUExecutionProvider"])


def campplus_embed(ort: onnxruntime.InferenceSession, wav_path: str) -> np.ndarray:
    wav, sr = load_mono(wav_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
    feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    inp = ort.get_inputs()[0].name
    emb = ort.run(None, {inp: feat.unsqueeze(0).cpu().numpy()})[0].reshape(-1)
    return emb.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def read_lines(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [x.rstrip("\n") for x in f]


def mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / max(1, len(xs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--ref_wav", required=True)
    ap.add_argument("--campplus_onnx", required=True)
    ap.add_argument("--gt_text", default="")
    ap.add_argument("--pred_text", default="")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    wavs = list_eval_wavs(args.audio_dir)
    if not wavs:
        raise SystemExit(f"[ERR] no wav under {args.audio_dir}")

    gt = read_lines(args.gt_text)
    pred = read_lines(args.pred_text)

    ort = build_ort_session(args.campplus_onnx)
    ref_emb = campplus_embed(ort, args.ref_wav)

    items: List[FileMetric] = []
    for i, w in enumerate(wavs):
        wav, sr = load_mono(w)
        dur = float(wav.shape[1]) / float(sr)
        m_rms = rms_db(wav)
        m_sil = silence_ratio(wav, sr)

        emb = campplus_embed(ort, w)
        spk = cosine(ref_emb, emb)

        m_cer = None
        if i < len(gt) and i < len(pred):
            m_cer = cer(gt[i], pred[i])

        items.append(FileMetric(w, dur, m_rms, m_sil, spk, m_cer))

    summary = {
        "n": len(items),
        "mean_duration_s": mean([x.duration_s for x in items]),
        "mean_rms_db": mean([x.rms_db for x in items]),
        "mean_silence_ratio": mean([x.silence_ratio for x in items]),
        "mean_speaker_cosine": mean([x.speaker_cosine for x in items]),
        "mean_CER": mean([x.cer for x in items if x.cer is not None]),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary,
             "detail": [
                 {"path": x.path, "duration_s": x.duration_s, "rms_db": x.rms_db,
                  "silence_ratio": x.silence_ratio, "speaker_cosine": x.speaker_cosine, "CER": x.cer}
                 for x in items
             ]},
            f, ensure_ascii=False, indent=2
        )

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Stable ZH Emotional Clone Metrics\n\n")
        f.write("## Summary\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Per-file\n\n")
        f.write("| idx | wav | dur(s) | rms(dB) | silence | spk_cos | CER |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for i, x in enumerate(items):
            name = os.path.basename(x.path)
            f.write(f"| {i} | {name} | {x.duration_s:.2f} | {x.rms_db:.2f} | {x.silence_ratio:.3f} | {x.speaker_cosine:.4f} | {'' if x.cer is None else f'{x.cer:.4f}'} |\n")

    print("[DONE]", args.out_json)
    print("[DONE]", args.out_md)


if __name__ == "__main__":
    main()