import argparse
import json
import os
import re
from pathlib import Path

PUNCTS = "，。！？；：、"

def norm_emo(e: str) -> str:
    e = (e or "neutral").strip().lower()
    alias = {"joy":"happy","happiness":"happy","anger":"angry","sadness":"sad","sorrow":"sad"}
    return alias.get(e, e)

def normalize_zh_punct(s: str) -> str:
    s = (s or "").strip()
    s = s.replace(",", "，").replace(".", "。").replace("!", "！").replace("?", "？").replace(";", "；").replace(":", "：")
    s = re.sub(r"\s+", "，", s)
    if not re.search(r"[。！？]$", s):
        s += "。"
    return s

def split_text_segments(s: str, max_chars: int, min_chars: int) -> list[str]:
    s = normalize_zh_punct(s)

    # 强标点先切
    sentences, buf = [], ""
    for ch in s:
        buf += ch
        if ch in "。！？":
            sentences.append(buf.strip()); buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    # 句内弱标点切（够长才切，避免过碎）
    chunks = []
    for sent in sentences:
        buf = ""
        for ch in sent:
            buf += ch
            if ch in "，；：、" and len(buf) >= min_chars:
                chunks.append(buf.strip()); buf = ""
        if buf.strip():
            chunks.append(buf.strip())

    # 合并到 <= max_chars
    merged, cur = [], ""
    for p in chunks:
        if not cur:
            cur = p; continue
        if len(cur) + len(p) <= max_chars:
            cur += p
        else:
            merged.append(cur); cur = p
    if cur:
        merged.append(cur)

    # 过短段合并
    final = []
    i = 0
    while i < len(merged):
        seg = merged[i]
        if len(seg) < min_chars and final:
            final[-1] += seg
        elif len(seg) < min_chars and i + 1 < len(merged):
            merged[i+1] = seg + merged[i+1]
        else:
            final.append(seg)
        i += 1

    return [x.strip() for x in final if x.strip()]

def read_text(text: str | None, text_file: str | None) -> str:
    if text_file:
        p = Path(text_file)
        if not p.exists():
            raise SystemExit(f"[ERR] --text_file not found: {text_file}")
        raw = p.read_text(encoding="utf-8").strip("\n\r")
        if not raw.strip():
            raise SystemExit(f"[ERR] --text_file is empty: {text_file}")
        return raw
    if text is None:
        raise SystemExit("[ERR] need --text or --text_file")
    if os.path.exists(text):
        raise SystemExit(f"[ERR] --text looks like an existing path: {text}")
    return text

def cosyvoice_instruct_template(emo: str) -> str:
    """
    用 CosyVoice 官方 common.py instruct_list 同风格的短指令（避免被念出来）
    关键：必须以 <|endofprompt|> 结尾
    """
    table = {
        "happy":  "You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>",
        "sad":    "You are a helpful assistant. 请非常伤心地说一句话。<|endofprompt|>",
        "angry":  "You are a helpful assistant. 请非常生气地说一句话。<|endofprompt|>",
        "neutral":"You are a helpful assistant.<|endofprompt|>",
    }
    return table.get(emo, table["neutral"])

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", default=None)
    g.add_argument("--text_file", default=None)
    ap.add_argument("--emotion", default="neutral")
    ap.add_argument("--ref_wav", required=True)
    ap.add_argument("--out_fixed", required=True)
    ap.add_argument("--max_chars", type=int, default=36)
    ap.add_argument("--min_chars", type=int, default=14)
    ap.add_argument("--key_prefix", default="xling")
    ap.add_argument("--no_split", action="store_true")
    ap.add_argument("--keep_exact", action="store_true")
    args = ap.parse_args()

    emo = norm_emo(args.emotion)
    text = read_text(args.text, args.text_file).strip()

    if args.keep_exact:
        full = text
        if not re.search(r"[。！？]$", full):
            full += "。"
    else:
        full = normalize_zh_punct(text)

    segs = [full] if args.no_split else split_text_segments(full, args.max_chars, args.min_chars)

    out_fixed = Path(args.out_fixed)
    out_fixed.parent.mkdir(parents=True, exist_ok=True)

    instruct = cosyvoice_instruct_template(emo)

    with out_fixed.open("w", encoding="utf-8") as fout:
        for i, seg in enumerate(segs):
            d = {
                "key": f"{args.key_prefix}_{i:04d}_{emo}",
                "emotion": emo,
                # ✅ 这里现在是“短指令 + <|endofprompt|>”，不会再放长中文说明
                "emotion_text_prompt": instruct,
                "neutral_speaker_wav": args.ref_wav,
                "source_text": seg,
                "target_text": seg,
                "answer_cosyvoice_speech_token": [],
                "target_len": 0,
                "source_len": max(1, len(seg)),
            }
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[INFO] wrote fixed jsonl: {out_fixed} (n={len(segs)})")
    print(f"[INFO] instruct: {instruct}")
    print(f"[INFO] first_seg: {segs[0]}")

if __name__ == "__main__":
    main()
