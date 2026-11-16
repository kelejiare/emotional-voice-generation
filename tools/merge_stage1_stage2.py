import os
import re
import glob
import argparse
import torch
import torchaudio


def _sort_key(path: str) -> int:
    # demo_0001_xxx.wav -> 1
    base = os.path.basename(path)
    m = re.search(r"demo_(\d{4})_", base)
    return int(m.group(1)) if m else 10**9


def _is_segment_file(path: str) -> bool:
    # 过滤 demo_0001_xxx_0.wav / _1.wav 这种分段文件
    base = os.path.basename(path)
    return re.search(r"_[0-9]+\.wav$", base) is not None


def load_mono(path: str):
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int):
    if sr == target_sr:
        return wav
    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav


def rms_normalize(wav: torch.Tensor, target_db: float = -20.0):
    # 简单 RMS 归一化到 target_db（dBFS 近似）
    rms = torch.sqrt(torch.mean(wav ** 2) + 1e-9)
    db = 20.0 * torch.log10(rms + 1e-9)
    gain = 10 ** ((target_db - db.item()) / 20.0)
    return wav * gain


def crossfade(a: torch.Tensor, b: torch.Tensor, sr: int, fade_ms: int):
    n = int(sr * fade_ms / 1000)
    if n <= 0:
        return torch.cat([a, b], dim=1)
    n = min(n, a.size(1), b.size(1))
    if n <= 0:
        return torch.cat([a, b], dim=1)

    fade_out = torch.linspace(1.0, 0.0, n).unsqueeze(0)
    fade_in = torch.linspace(0.0, 1.0, n).unsqueeze(0)
    mid = a[:, -n:] * fade_out + b[:, :n] * fade_in
    return torch.cat([a[:, :-n], mid, b[:, n:]], dim=1)


def merge_dir(
    in_dir: str,
    out_wav: str,
    pattern: str = "demo_*.wav",
    gap_ms: int = 140,
    fade_ms: int = 12,
    target_db: float = -20.0,
    do_rms_norm: bool = True,
):
    wav_paths = glob.glob(os.path.join(in_dir, pattern))
    wav_paths = [p for p in wav_paths if os.path.isfile(p) and (not _is_segment_file(p))]
    wav_paths = sorted(wav_paths, key=_sort_key)

    if not wav_paths:
        raise RuntimeError(f"[ERR] No wav matched: dir={in_dir}, pattern={pattern}")

    # 以第一条的采样率为准
    first_wav, sr = load_mono(wav_paths[0])
    target_sr = sr

    gap = torch.zeros(1, int(target_sr * gap_ms / 1000))

    out = None
    for p in wav_paths:
        w, sr0 = load_mono(p)
        w = resample_if_needed(w, sr0, target_sr)
        if do_rms_norm:
            w = rms_normalize(w, target_db)

        if out is None:
            out = w
        else:
            out = torch.cat([out, gap], dim=1)
            out = crossfade(out, w, target_sr, fade_ms)

    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    torchaudio.save(out_wav, out.cpu(), target_sr)

    dur = out.size(1) / target_sr
    print(f"[DONE] merged {len(wav_paths)} files -> {out_wav}")
    print(f"[DONE] duration = {dur:.2f}s, sr = {target_sr}")
    return out_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_dir", required=True)
    ap.add_argument("--stage2_dir", required=True)
    ap.add_argument("--out_stage1", required=True)
    ap.add_argument("--out_stage2", required=True)
    ap.add_argument("--pattern", default="demo_*.wav")
    ap.add_argument("--gap_ms", type=int, default=140)
    ap.add_argument("--fade_ms", type=int, default=12)
    ap.add_argument("--target_db", type=float, default=-20.0)
    ap.add_argument("--no_rms_norm", action="store_true")
    args = ap.parse_args()

    do_rms = not args.no_rms_norm

    print("[INFO] Merge Stage1 ...")
    merge_dir(
        in_dir=args.stage1_dir,
        out_wav=args.out_stage1,
        pattern=args.pattern,
        gap_ms=args.gap_ms,
        fade_ms=args.fade_ms,
        target_db=args.target_db,
        do_rms_norm=do_rms,
    )

    print("[INFO] Merge Stage2 ...")
    merge_dir(
        in_dir=args.stage2_dir,
        out_wav=args.out_stage2,
        pattern=args.pattern,
        gap_ms=args.gap_ms,
        fade_ms=args.fade_ms,
        target_db=args.target_db,
        do_rms_norm=do_rms,
    )


if __name__ == "__main__":
    main()