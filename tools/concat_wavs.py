import argparse, os, glob
import torch
import torchaudio

def pick_wavs(d: str, pattern: str) -> list[str]:
    wavs = sorted(glob.glob(os.path.join(d, pattern)))
    # 如果同时存在 base.wav 与 base_0.wav，优先 base.wav
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--pattern", default="*.wav")
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
        pieces.append(x)

    y = torch.cat(pieces, dim=1)
    os.makedirs(os.path.dirname(args.out_wav), exist_ok=True)
    torchaudio.save(args.out_wav, y, out_sr)
    print("[DONE] wrote:", args.out_wav)

if __name__ == "__main__":
    main()