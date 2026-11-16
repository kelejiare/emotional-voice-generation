import argparse, os, re
import torch
import torchaudio

def list_wavs(in_dir: str, pattern: str):
    import glob
    wavs = sorted(glob.glob(os.path.join(in_dir, pattern)))
    # 按 demo_0000_xxx 的数字排序
    def key_fn(p):
        m = re.search(r"_(\d{4})_", os.path.basename(p))
        return int(m.group(1)) if m else 10**9
    return sorted(wavs, key=key_fn)

def load_mono_16k(path, target_sr):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav, sr

def trim_silence(wav: torch.Tensor, sr: int, thr_db: float = -35.0):
    # 简单能量阈值裁剪
    eps = 1e-9
    frame = int(sr * 0.02)  # 20ms
    hop = int(sr * 0.01)    # 10ms
    x = wav[0]
    if x.numel() < frame:
        return wav
    frames = x.unfold(0, frame, hop)
    rms = torch.sqrt(torch.mean(frames**2, dim=1) + eps)
    db = 20 * torch.log10(rms + eps)
    idx = torch.where(db > thr_db)[0]
    if idx.numel() == 0:
        return wav
    start = int(idx[0].item() * hop)
    end = int(min(x.numel(), idx[-1].item() * hop + frame))
    return wav[:, start:end]

def rms_normalize(wav: torch.Tensor, target_dbfs: float = -20.0):
    eps = 1e-9
    rms = torch.sqrt(torch.mean(wav**2) + eps)
    db = 20 * torch.log10(rms + eps)
    gain = 10 ** ((target_dbfs - db.item()) / 20)
    return wav * gain

def crossfade(a: torch.Tensor, b: torch.Tensor, n: int):
    # a,b: [1,T]
    if n <= 0:
        return torch.cat([a, b], dim=1)
    n = min(n, a.size(1), b.size(1))
    fa = torch.linspace(1, 0, n)
    fb = torch.linspace(0, 1, n)
    mid = a[:, -n:] * fa + b[:, :n] * fb
    return torch.cat([a[:, :-n], mid, b[:, n:]], dim=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--pattern", default="demo_*.wav")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--fade_ms", type=int, default=25)   # crossfade 时长
    ap.add_argument("--gap_ms", type=int, default=30)    # 段间微停顿
    args = ap.parse_args()

    wavs = list_wavs(args.in_dir, args.pattern)
    if not wavs:
        raise SystemExit(f"[ERR] no wav in {args.in_dir} pattern={args.pattern}")

    fade_n = int(args.sr * args.fade_ms / 1000)
    gap_n = int(args.sr * args.gap_ms / 1000)
    gap = torch.zeros(1, gap_n)

    out = None
    for p in wavs:
        w, sr = load_mono_16k(p, args.sr)
        w = trim_silence(w, args.sr, thr_db=-35.0)
        w = rms_normalize(w, target_dbfs=-20.0)
        if out is None:
            out = w
        else:
            out = torch.cat([out, gap], dim=1)
            out = crossfade(out, w, fade_n)

    os.makedirs(os.path.dirname(args.out_wav), exist_ok=True)
    torchaudio.save(args.out_wav, out.cpu(), args.sr)
    print(f"[DONE] wrote {args.out_wav} (sec={out.size(1)/args.sr:.2f})")

if __name__ == "__main__":
    main()