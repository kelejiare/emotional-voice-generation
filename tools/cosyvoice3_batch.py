# tools/cosyvoice3_batch.py
import os, sys, argparse, glob, re, json
from typing import List, Tuple, Optional
import torch
import torchaudio

END = "<|endofprompt|>"

def add_paths(cosyvoice_repo: str):
    sys.path.insert(0, cosyvoice_repo)
    sys.path.insert(0, os.path.join(cosyvoice_repo, "third_party", "Matcha-TTS"))

def load_model(model_dir: str):
    from cosyvoice.cli.cosyvoice import AutoModel
    return AutoModel(model_dir=model_dir)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _load_mono(wav_path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(wav_path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr

def _save_wav(path: str, wav: torch.Tensor, sr: int):
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)
    torchaudio.save(path, wav.cpu(), sr)

def prepare_wav(in_wav: str, out_wav: str, target_sr: int) -> str:
    wav, sr = _load_mono(in_wav)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
        sr = target_sr
    _save_wav(out_wav, wav, sr)
    return out_wav

def save_outputs(outputs, out_prefix: str, sample_rate: int, concat: bool = False, no_segments: bool = False):
    segs: List[torch.Tensor] = []
    if outputs is None:
        return
    for i, j in enumerate(outputs):
        wav = j["tts_speech"].detach().cpu()
        segs.append(wav)
        if not no_segments:
            _save_wav(f"{out_prefix}_{i}.wav", wav, sample_rate)
    if concat and len(segs) > 0:
        full = torch.cat(segs, dim=1)
        _save_wav(f"{out_prefix}.wav", full, sample_rate)

def run_vc(model_dir: str, src_dir: str, prompt_wav: str, out_dir: str, pattern: str, concat: bool):
    cosyvoice = load_model(model_dir)
    ensure_dir(out_dir)

    wavs = sorted(glob.glob(os.path.join(src_dir, pattern)))
    if not wavs:
        raise SystemExit(f"[ERR] no wav found under {src_dir} with pattern={pattern}")

    tmp_dir = os.path.join(out_dir, "_tmp_wavs")
    ensure_dir(tmp_dir)

    prompt_prepared = prepare_wav(prompt_wav, os.path.join(tmp_dir, "prompt_prepared.wav"), cosyvoice.sample_rate)

    for w in wavs:
        base = os.path.splitext(os.path.basename(w))[0]
        out_prefix = os.path.join(out_dir, base)
        src_prepared = prepare_wav(w, os.path.join(tmp_dir, f"{base}_src_prepared.wav"), cosyvoice.sample_rate)

        print(f"[VC] {w} -> {out_prefix}_*.wav" + (" + concat" if concat else ""))
        try:
            with torch.no_grad():
                outputs = cosyvoice.inference_vc(src_prepared, prompt_prepared, stream=False)
            save_outputs(outputs, out_prefix, cosyvoice.sample_rate, concat=concat)
        except Exception as e:
            print(f"[VC][ERR] {w}: {e}")

def _normalize_prompt_text(prompt: str) -> str:
    p = (prompt or "").strip()
    if END not in p:
        p = p.rstrip() + END
    # 更贴官方示例：给一个极短 system prefix（通常不会被朗读）
    # 官方示例里大量出现 "You are a helpful assistant.<|endofprompt|>..." 的格式 :contentReference[oaicite:2]{index=2}
    if "You are a helpful assistant" not in p:
        p = "You are a helpful assistant. " + p
    # 保证 END 在最后（避免 prompt 后面又粘别的）
    if not p.endswith(END):
        # 若 prompt 内部已有 END，强制截到最后一个 END
        idx = p.rfind(END)
        p = p[: idx + len(END)]
    return p

def run_instruct2(model_dir: str, jsonl_path: str, prompt_wav: str, out_dir: str, concat: bool):
    """
    保留你原先的 instruct2，但注意：如果你把“很长的中文提示词”放在 prompt 里，仍可能被模型念出来。
    更推荐用下面的 xling_tts（同样走 cross_lingual，但更严格约束 prompt 结构）。
    """
    cosyvoice = load_model(model_dir)
    ensure_dir(out_dir)

    tmp_dir = os.path.join(out_dir, "_tmp_wavs")
    ensure_dir(tmp_dir)
    prompt_prepared = prepare_wav(prompt_wav, os.path.join(tmp_dir, "prompt_prepared.wav"), cosyvoice.sample_rate)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            d=json.loads(line)
            key = d["key"]
            text = (d.get("target_text") or d.get("source_text") or "").strip()
            instruct = (d.get("emotion_text_prompt") or END).strip()
            instruct = _normalize_prompt_text(instruct)
            combined = f"{instruct}{text}"

            out_prefix = os.path.join(out_dir, key)
            print(f"[INS] {key} -> {out_prefix}.wav" + (" + concat" if concat else ""))
            try:
                with torch.no_grad():
                    outputs = cosyvoice.inference_cross_lingual(combined, prompt_prepared, stream=False)
                save_outputs(outputs, out_prefix, cosyvoice.sample_rate, concat=concat)
            except Exception as e:
                print(f"[INS][ERR] {key}: {e}")

def run_xling_tts(model_dir: str, jsonl_path: str, default_prompt_wav: str,
                 out_dir: str, concat: bool, no_segments: bool):
    """
    ✅ 推荐使用：完全按 CosyVoice3 官方范式组织输入：
      combined = "You are a helpful assistant. <emotion instruction><|endofprompt|>" + <tts_text>
    然后调用 inference_cross_lingual(combined, prompt_wav)

    这样既满足 <|endofprompt|> 强校验，也能最大化降低“提示词被念出来”的概率。:contentReference[oaicite:3]{index=3}
    """
    cosyvoice = load_model(model_dir)
    ensure_dir(out_dir)

    tmp_dir = os.path.join(out_dir, "_tmp_wavs")
    ensure_dir(tmp_dir)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            d=json.loads(line)
            key = d["key"]
            text = (d.get("target_text") or d.get("source_text") or "").strip()
            if len(text) < 6:
                print(f"[XLING][SKIP] {key} text too short: {text!r}")
                continue

            prompt_text = _normalize_prompt_text(d.get("emotion_text_prompt") or END)
            combined = f"{prompt_text}{text}"

            prompt_wav = (d.get("prompt_wav") or default_prompt_wav)
            prompt_wav = os.path.abspath(prompt_wav)

            # prepare wav per-sample（不同情绪参考音频）
            prompt_prepared = prepare_wav(
                prompt_wav,
                os.path.join(tmp_dir, f"{key}_prompt_prepared.wav"),
                cosyvoice.sample_rate
            )

            out_prefix = os.path.join(out_dir, key)
            print(f"[XLING] {key} -> {out_prefix}.wav" + (" + concat" if concat else ""))
            print(f"[XLING] prompt_wav={prompt_wav}")
            try:
                with torch.no_grad():
                    outputs = cosyvoice.inference_cross_lingual(combined, prompt_prepared, stream=False)
                save_outputs(outputs, out_prefix, cosyvoice.sample_rate, concat=concat, no_segments=no_segments)
            except Exception as e:
                print(f"[XLING][ERR] {key}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice_repo", required=True)
    ap.add_argument("--model_dir", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_vc = sub.add_parser("vc")
    ap_vc.add_argument("--src_dir", required=True)
    ap_vc.add_argument("--prompt_wav", required=True)
    ap_vc.add_argument("--out_dir", required=True)
    ap_vc.add_argument("--pattern", default="*.wav")
    ap_vc.add_argument("--concat", action="store_true")

    ap_ins = sub.add_parser("instruct2")
    ap_ins.add_argument("--jsonl", required=True)
    ap_ins.add_argument("--prompt_wav", required=True)
    ap_ins.add_argument("--out_dir", required=True)
    ap_ins.add_argument("--concat", action="store_true")

    ap_x = sub.add_parser("xling_tts")
    ap_x.add_argument("--jsonl", required=True)
    ap_x.add_argument("--default_prompt_wav", required=True,
                      help="Stage1 情绪/韵律参考音频（若 jsonl 行内没写 prompt_wav 则用它）")
    ap_x.add_argument("--out_dir", required=True)
    ap_x.add_argument("--concat", action="store_true")
    ap_x.add_argument("--no_segments", action="store_true",
                      help="不保存 *_0.wav 段落文件，仅保存合并后的 key.wav（需要 --concat）")

    args = ap.parse_args()
    add_paths(args.cosyvoice_repo)

    if args.cmd == "vc":
        run_vc(args.model_dir, args.src_dir, args.prompt_wav, args.out_dir, args.pattern, args.concat)
    elif args.cmd == "instruct2":
        run_instruct2(args.model_dir, args.jsonl, args.prompt_wav, args.out_dir, args.concat)
    else:
        run_xling_tts(args.model_dir, args.jsonl, args.default_prompt_wav, args.out_dir, args.concat, args.no_segments)

if __name__ == "__main__":
    main()