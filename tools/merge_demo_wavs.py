import os, re, glob, argparse
import torch
import torchaudio

def load_mono(path, target_sr=None):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr and sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav, sr

def rms_norm(wav, target_db=-20.0):
    rms = torch.sqrt(torch.mean(wav**2) + 1e-9)
    db = 20 * torch.log10(rms + 1e-9)
    gain = 10 ** ((target_db - db.item()) / 20)
    return wav * gain

def crossfade(a, b, sr, ms=30):
    n = int(sr * ms / 1000)
    if n <= 0:
        return torch.cat([a, b], dim=1)
    n = min(n, a.size(1), b.size(1))
    fa = torch.linspace(1, 0, n)
    fb = torch.linspace(0, 1, n)
    mid = a[:, -n:] * fa + b[:, :n] * fb
    return torch.cat([a[:, :-n], mid, b[:, n:]], dim=1)

def sort_key(p):
    m = re.search(r"_(\d{4})_", os.path.basename(p))
    return int(m.group(1)) if m else 10**9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--pattern", default="demo_*.wav")
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--gap_ms", type=int, default=40)
    ap.add_argument("--fade_ms", type=int, default=30)
    ap.add_argument("--target_db", type=float, default=-20.0)
    args = ap.parse_args()

    wavs = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)), key=sort_key)
    if not wavs:
        raise SystemExit(f"[ERR] no wavs in {args.in_dir} pattern={args.pattern}")

    out = None
    sr = None

    for p in wavs:
        w, sr = load_mono(p, sr)
        w = rms_norm(w, args.target_db)
        if out is None:
            out = w
        else:
            gap = torch.zeros(1, int(sr * args.gap_ms / 1000))
            out = torch.cat([out, gap], dim=1)
            out = crossfade(out, w, sr, args.fade_ms)

    os.makedirs(os.path.dirname(args.out_wav), exist_ok=True)
    torchaudio.save(args.out_wav, out.cpu(), sr)
    print("[DONE] wrote:", args.out_wav)
    print("[DONE] duration(sec):", out.size(1) / sr)

if __name__ == "__main__":
    main()