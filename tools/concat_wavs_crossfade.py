import argparse, os, glob
import torch
import torchaudio

def pick_wavs(d: str, pattern: str) -> list[str]:
    wavs = sorted(glob.glob(os.path.join(d, pattern)))
    base_set = set(os.path.basename(x) for x in wavs)
    chosen = []
    for w in wavs:
        bn = os.path.basename(w)
        if bn.endswith("_0.wav"):
            base = bn[:-6] + ".wav"
            if base in base_set:
                continue
        chosen.append(w)
    return chosen

def fade(wav: torch.Tensor, sr: int, fade_ms: int):
    if fade_ms <= 0:
        return wav
    n = int(sr * fade_ms / 1000)
    n = min(n, wav.shape[1] // 2)
    if n <= 1:
        return wav
    ramp = torch.linspace(0.0, 1.0, n, device=wav.device).unsqueeze(0)
    wav[:, :n] *= ramp
    wav[:, -n:] *= ramp.flip(dims=[1])
    return wav

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--pattern", default="xling_*.wav")
    ap.add_argument("--gap_ms", type=int, default=120)   # 段间静音
    ap.add_argument("--fade_ms", type=int, default=10)   # 段首尾淡入淡出
    args = ap.parse_args()

    wavs = pick_wavs(args.in_dir, args.pattern)
    if not wavs:
        raise SystemExit(f"[ERR] no wav in {args.in_dir} with {args.pattern}")

    pieces = []
    out_sr = None

    for w in wavs:
        x, sr = torchaudio.load(w)
        if x.size(0) > 1:
            x = x.mean(dim=0, keepdim=True)
        if out_sr is None:
            out_sr = sr
        elif sr != out_sr:
            x = torchaudio.transforms.Resample(sr, out_sr)(x)

        x = fade(x, out_sr, args.fade_ms)
        pieces.append(x)

        if args.gap_ms > 0:
            gap_n = int(out_sr * args.gap_ms / 1000)
            pieces.append(torch.zeros((1, gap_n), dtype=x.dtype))

    y = torch.cat(pieces, dim=1)
    os.makedirs(os.path.dirname(args.out_wav), exist_ok=True)
    torchaudio.save(args.out_wav, y, out_sr)
    print("[DONE] wrote:", args.out_wav)

if __name__ == "__main__":
    main()
